from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

from .metrics import evaluate_regression
from .preprocess import normalize_text


@dataclass
class EnsembleArtifacts:
    tfidf: TfidfVectorizer
    ridge: Ridge
    isotonic: IsotonicRegression
    data_min: float
    data_max: float
    reg_weight: float
    sim_weight: float
    val_metrics: Dict[str, float]
    used_sbert: bool


def clean_text(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def gibberish_score_hint(text: str) -> bool:
    s = str(text or "").strip()
    if len(s) == 0:
        return True
    words = re.findall(r"[A-Za-z0-9']+", s)
    if len(words) < 2 and len(s) < 5:
        return True
    alpha_ratio = sum(1 for ch in s if ch.isalpha()) / max(1, len(s))
    if alpha_ratio < 0.3:
        return True
    if not re.search(r"[aeiouAEIOU]", s):
        return True
    return False


def _normalize(arr: np.ndarray, amin: float, amax: float) -> np.ndarray:
    if amax == amin:
        return np.zeros_like(arr, dtype=float) + 0.5
    return (arr - amin) / (amax - amin)


def _tfidf_cosine_similarity(vectorizer: TfidfVectorizer, students: pd.Series, references: pd.Series) -> np.ndarray:
    vecs_s = vectorizer.transform(students.astype(str).tolist())
    vecs_r = vectorizer.transform(references.astype(str).tolist())
    dot = vecs_s.multiply(vecs_r).sum(axis=1).A1
    s_norm = np.sqrt(vecs_s.multiply(vecs_s).sum(axis=1).A1)
    r_norm = np.sqrt(vecs_r.multiply(vecs_r).sum(axis=1).A1)
    denom = s_norm * r_norm
    sims = np.zeros_like(dot, dtype=float)
    nonzero = denom > 0
    sims[nonzero] = dot[nonzero] / denom[nonzero]
    return np.clip(sims, 0.0, 1.0)


def _sbert_similarity(students: pd.Series, references: pd.Series, model_name: str) -> Tuple[np.ndarray, bool]:
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return np.zeros(len(students), dtype=float), False

    try:
        model = SentenceTransformer(model_name)
    except Exception:
        return np.zeros(len(students), dtype=float), False

    st = [str(x or "") for x in students]
    rf = [str(x or "") for x in references]
    if all(r.strip() == "" for r in rf):
        return np.zeros(len(st), dtype=float), True

    sims = []
    batch_size = 64
    for i in range(0, len(st), batch_size):
        s_batch = st[i : i + batch_size]
        r_batch = rf[i : i + batch_size]
        s_emb = model.encode(s_batch, convert_to_tensor=True, show_progress_bar=False)
        r_emb = model.encode(r_batch, convert_to_tensor=True, show_progress_bar=False)
        cos_mat = util.cos_sim(s_emb, r_emb).cpu().numpy()
        sims.extend(np.diag(cos_mat).tolist())
    sims = np.asarray(sims, dtype=float)
    return np.clip((sims + 1.0) / 2.0, 0.0, 1.0), True


def train_similarity_ensemble(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Dict) -> EnsembleArtifacts:
    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["student_answer"] = train_df["student_answer"].map(clean_text)
    train_df["reference_answer"] = train_df["reference_answer"].map(clean_text)
    val_df["student_answer"] = val_df["student_answer"].map(clean_text)
    val_df["reference_answer"] = val_df["reference_answer"].map(clean_text)

    data_min = float(min(train_df["score"].min(), val_df["score"].min()))
    data_max = float(max(train_df["score"].max(), val_df["score"].max()))

    tfidf = TfidfVectorizer(
        max_features=int(cfg.get("max_tfidf_features", 10000)),
        ngram_range=(1, 2),
    )
    tfidf.fit(pd.concat([train_df["student_answer"], train_df["reference_answer"]]).astype(str))

    X_train = tfidf.transform(train_df["student_answer"].astype(str).tolist())
    X_val = tfidf.transform(val_df["student_answer"].astype(str).tolist())

    ridge = Ridge(alpha=float(cfg.get("ridge_alpha", 1.0)))
    ridge.fit(X_train, train_df["score"].to_numpy())
    val_pred_reg = ridge.predict(X_val)

    use_sbert = bool(cfg.get("use_sbert", False))
    used_sbert = False

    if use_sbert:
        train_sims, used_sbert = _sbert_similarity(
            train_df["student_answer"],
            train_df["reference_answer"],
            model_name=str(cfg.get("sbert_model_name", "all-mpnet-base-v2")),
        )
        val_sims, used_sbert_val = _sbert_similarity(
            val_df["student_answer"],
            val_df["reference_answer"],
            model_name=str(cfg.get("sbert_model_name", "all-mpnet-base-v2")),
        )
        used_sbert = used_sbert and used_sbert_val
        if not used_sbert:
            train_sims = _tfidf_cosine_similarity(tfidf, train_df["student_answer"], train_df["reference_answer"])
            val_sims = _tfidf_cosine_similarity(tfidf, val_df["student_answer"], val_df["reference_answer"])
    else:
        train_sims = _tfidf_cosine_similarity(tfidf, train_df["student_answer"], train_df["reference_answer"])
        val_sims = _tfidf_cosine_similarity(tfidf, val_df["student_answer"], val_df["reference_answer"])

    power = float(cfg.get("similarity_contrast_power", 1.8))
    train_sims = np.clip(np.power(train_sims, power), 0.0, 1.0)
    val_sims = np.clip(np.power(val_sims, power), 0.0, 1.0)

    train_score_norm = _normalize(train_df["score"].to_numpy(), data_min, data_max)

    # synthetic extremes for isotonic robustness
    n_synth = min(int(cfg.get("synthetic_extremes", 50)), len(train_df))
    refs = train_df["reference_answer"].astype(str).tolist()[:n_synth]
    synth_perfect_s = refs
    synth_perfect_r = refs
    synth_bad_s = [""] * n_synth
    synth_bad_r = refs

    if n_synth > 0:
        sims_perfect = _tfidf_cosine_similarity(tfidf, pd.Series(synth_perfect_s), pd.Series(synth_perfect_r))
        sims_bad = _tfidf_cosine_similarity(tfidf, pd.Series(synth_bad_s), pd.Series(synth_bad_r))
        aug_train_sims = np.concatenate([train_sims, sims_perfect, sims_bad])
        aug_train_labels = np.concatenate([train_score_norm, np.ones_like(sims_perfect), np.zeros_like(sims_bad)])
    else:
        aug_train_sims = train_sims
        aug_train_labels = train_score_norm

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(aug_train_sims, aug_train_labels)

    val_calib_norm = isotonic.predict(val_sims)
    val_calib_score = val_calib_norm * (data_max - data_min) + data_min

    val_reg_norm = _normalize(val_pred_reg, data_min, data_max)
    val_sim_norm = _normalize(val_calib_score, data_min, data_max)

    min_sim_weight = float(cfg.get("min_similarity_weight", 0.6))
    grid_steps = int(cfg.get("weight_grid_steps", 21))
    best = None
    for w in np.linspace(0.0, 1.0, grid_steps):
        reg_w = float(w)
        sim_w = float(1.0 - w)
        if sim_w < min_sim_weight:
            continue
        final_norm = val_reg_norm * reg_w + val_sim_norm * sim_w
        final_score = final_norm * (data_max - data_min) + data_min
        metrics = evaluate_regression(
            val_df["score"].to_numpy(),
            final_score,
            min_score=data_min,
            max_score=data_max,
        )
        key = metrics["qwk"]
        if best is None or key > best[2]:
            best = (reg_w, sim_w, key, metrics)

    if best is None:
        best = (0.05, 0.95, -1.0, {"mse": float("inf"), "qwk": -1.0})

    return EnsembleArtifacts(
        tfidf=tfidf,
        ridge=ridge,
        isotonic=isotonic,
        data_min=data_min,
        data_max=data_max,
        reg_weight=best[0],
        sim_weight=best[1],
        val_metrics=best[3],
        used_sbert=used_sbert,
    )


def predict_ensemble(artifacts: EnsembleArtifacts, reference: str, student: str) -> Tuple[float, Dict[str, float]]:
    reference = clean_text(reference)
    student = clean_text(student)

    # Hard guardrail: exact normalized match should always receive full marks.
    # This prevents weighted blending from reducing a perfect answer.
    ref_norm = normalize_text(reference)
    stu_norm = normalize_text(student)
    if ref_norm != "" and ref_norm == stu_norm:
        return float(artifacts.data_max), {
            "reg_score": float(artifacts.data_max),
            "sim_score": float(artifacts.data_max),
            "reg_weight": float(artifacts.reg_weight),
            "sim_weight": float(artifacts.sim_weight),
            "used_sbert": bool(artifacts.used_sbert),
            "rule_applied": "exact_match_full_score",
        }

    reg_score = float(artifacts.ridge.predict(artifacts.tfidf.transform([student]))[0])

    sim = _tfidf_cosine_similarity(artifacts.tfidf, pd.Series([student]), pd.Series([reference]))[0]
    sim_norm = float(artifacts.isotonic.predict([sim])[0])
    sim_score = sim_norm * (artifacts.data_max - artifacts.data_min) + artifacts.data_min

    reg_n = _normalize(np.asarray([reg_score]), artifacts.data_min, artifacts.data_max)[0]
    sim_n = _normalize(np.asarray([sim_score]), artifacts.data_min, artifacts.data_max)[0]

    total = artifacts.reg_weight + artifacts.sim_weight
    final_norm = (reg_n * artifacts.reg_weight + sim_n * artifacts.sim_weight) / max(total, 1e-9)
    final_score = final_norm * (artifacts.data_max - artifacts.data_min) + artifacts.data_min

    if gibberish_score_hint(student):
        final_score = artifacts.data_min

    final_score = float(np.clip(final_score, artifacts.data_min, artifacts.data_max))
    details = {
        "reg_score": float(reg_score),
        "sim_score": float(sim_score),
        "reg_weight": float(artifacts.reg_weight),
        "sim_weight": float(artifacts.sim_weight),
        "used_sbert": bool(artifacts.used_sbert),
    }
    return final_score, details


def save_ensemble(artifacts: EnsembleArtifacts, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts.tfidf, out_dir / "ensemble_tfidf.joblib")
    joblib.dump(artifacts.ridge, out_dir / "ensemble_ridge.joblib")
    joblib.dump(artifacts.isotonic, out_dir / "ensemble_isotonic.joblib")

    with open(out_dir / "ensemble_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_min": artifacts.data_min,
                "data_max": artifacts.data_max,
                "reg_weight": artifacts.reg_weight,
                "sim_weight": artifacts.sim_weight,
                "val_metrics": artifacts.val_metrics,
                "used_sbert": artifacts.used_sbert,
            },
            f,
            indent=2,
        )


def load_ensemble(out_dir: str | Path) -> EnsembleArtifacts:
    out_dir = Path(out_dir)
    tfidf = joblib.load(out_dir / "ensemble_tfidf.joblib")
    ridge = joblib.load(out_dir / "ensemble_ridge.joblib")
    isotonic = joblib.load(out_dir / "ensemble_isotonic.joblib")
    with open(out_dir / "ensemble_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return EnsembleArtifacts(
        tfidf=tfidf,
        ridge=ridge,
        isotonic=isotonic,
        data_min=float(meta["data_min"]),
        data_max=float(meta["data_max"]),
        reg_weight=float(meta["reg_weight"]),
        sim_weight=float(meta["sim_weight"]),
        val_metrics=meta.get("val_metrics", {}),
        used_sbert=bool(meta.get("used_sbert", False)),
    )
