from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pandas as pd

from .preprocess import normalize_text


@dataclass
class AdaptedDataset:
    df: pd.DataFrame
    source_name: str


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl"}:
        return pd.read_json(path, lines=True)
    if suffix in {".json"}:
        return pd.read_json(path)
    return pd.read_csv(path)


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def _map_text_score(df: pd.DataFrame, col: str, mapping: Dict[str, float]) -> pd.Series:
    s = df[col].astype(str).str.strip().str.lower()
    return s.map(mapping)


def adapt_asap(path: str | Path) -> AdaptedDataset:
    raw = _read_table(path)

    student_col = _first_existing(raw, ["essay", "student_answer", "answer", "response"])
    score_col = _first_existing(raw, ["domain1_score", "score", "label", "grade"])
    ref_col = _first_existing(raw, ["prompt", "prompt_text", "reference_answer", "model_answer", "question"])

    if student_col is None or score_col is None:
        raise ValueError("ASAP adapter requires essay/answer and score columns")

    if ref_col is None:
        prompt_id_col = _first_existing(raw, ["essay_set", "prompt_id", "set_id"])
        if prompt_id_col is not None:
            reference = raw[prompt_id_col].astype(str).map(lambda x: f"prompt {x}")
        else:
            reference = pd.Series(["general writing prompt"] * len(raw))
    else:
        reference = raw[ref_col]

    df = pd.DataFrame(
        {
            "reference_answer": reference,
            "student_answer": raw[student_col],
            "score": pd.to_numeric(raw[score_col], errors="coerce"),
            "dataset": "asap",
        }
    )
    return AdaptedDataset(df=df, source_name="asap")


def adapt_scientsbank(path: str | Path) -> AdaptedDataset:
    raw = _read_table(path)

    ref_col = _first_existing(raw, ["reference_answer", "model_answer", "answer_key", "expected_answer"])
    student_col = _first_existing(raw, ["student_answer", "student_response", "answer", "response"])
    score_col = _first_existing(raw, ["score", "label", "grade", "gold_label"])

    if ref_col is None or student_col is None or score_col is None:
        raise ValueError("SciEntsBank adapter requires reference, student, and score columns")

    score = pd.to_numeric(raw[score_col], errors="coerce")
    if score.isna().all():
        score = _map_text_score(
            raw,
            score_col,
            {
                "incorrect": 0.0,
                "contradictory": 0.0,
                "partially_correct_incomplete": 1.0,
                "partially correct": 1.0,
                "partially_correct": 1.0,
                "correct": 2.0,
            },
        )

    df = pd.DataFrame(
        {
            "reference_answer": raw[ref_col],
            "student_answer": raw[student_col],
            "score": pd.to_numeric(score, errors="coerce"),
            "dataset": "scientsbank",
        }
    )
    return AdaptedDataset(df=df, source_name="scientsbank")


def adapt_mohler(path: str | Path) -> AdaptedDataset:
    raw = _read_table(path)

    ref_col = _first_existing(raw, ["reference_answer", "model_answer", "expected_answer", "answer_key"])
    student_col = _first_existing(raw, ["student_answer", "student_response", "answer", "response"])
    score_col = _first_existing(raw, ["score", "label", "grade"])

    if ref_col is None or student_col is None or score_col is None:
        raise ValueError("Mohler adapter requires reference, student, and score columns")

    score = pd.to_numeric(raw[score_col], errors="coerce")
    if score.isna().all():
        score = _map_text_score(
            raw,
            score_col,
            {
                "incorrect": 0.0,
                "partially_correct": 2.0,
                "partially correct": 2.0,
                "correct": 5.0,
            },
        )

    df = pd.DataFrame(
        {
            "reference_answer": raw[ref_col],
            "student_answer": raw[student_col],
            "score": pd.to_numeric(score, errors="coerce"),
            "dataset": "mohler",
        }
    )
    return AdaptedDataset(df=df, source_name="mohler")


def adapt_sts(path: str | Path) -> AdaptedDataset:
    raw = _read_table(path)

    s1_col = _first_existing(raw, ["sentence1", "sent1", "text1", "reference_answer"])
    s2_col = _first_existing(raw, ["sentence2", "sent2", "text2", "student_answer"])
    score_col = _first_existing(raw, ["score", "label", "similarity_score", "sts_score"])

    if s1_col is None or s2_col is None or score_col is None:
        raise ValueError("STS adapter requires sentence1, sentence2, and score columns")

    df = pd.DataFrame(
        {
            "reference_answer": raw[s1_col],
            "student_answer": raw[s2_col],
            "score": pd.to_numeric(raw[score_col], errors="coerce"),
            "dataset": "sts",
        }
    )
    return AdaptedDataset(df=df, source_name="sts")


def _normalize_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reference_answer"] = out["reference_answer"].fillna("").map(normalize_text)
    out["student_answer"] = out["student_answer"].fillna("").map(normalize_text)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score"])
    out = out[(out["reference_answer"] != "") | (out["student_answer"] != "")]
    return out.reset_index(drop=True)


def infer_adapter_key(path: str | Path, columns: Iterable[str]) -> str:
    p = str(path).lower()
    cols = {c.lower() for c in columns}

    if "asap" in p or "essay_set" in cols or "domain1_score" in cols:
        return "asap"
    if "scientsbank" in p or "gold_label" in cols:
        return "scientsbank"
    if "mohler" in p or "mrc" in p:
        return "mohler"
    if "sts" in p or ({"sentence1", "sentence2"}.issubset(cols)):
        return "sts"

    # fallback based on available columns
    if {"sentence1", "sentence2"}.issubset(cols):
        return "sts"
    if "domain1_score" in cols or "essay" in cols:
        return "asap"
    if "gold_label" in cols:
        return "scientsbank"
    return "mohler"


def merge_datasets(paths: List[str | Path], drop_duplicates: bool = False) -> pd.DataFrame:
    adapters: Dict[str, Callable[[str | Path], AdaptedDataset]] = {
        "asap": adapt_asap,
        "scientsbank": adapt_scientsbank,
        "mohler": adapt_mohler,
        "sts": adapt_sts,
    }

    frames = []
    for path in paths:
        raw = _read_table(path)
        key = infer_adapter_key(path, raw.columns)
        adapted = adapters[key](path)
        frames.append(_normalize_output(adapted.df))

    if not frames:
        raise ValueError("No datasets were provided to merge")

    merged = pd.concat(frames, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates(subset=["reference_answer", "student_answer", "score"]).reset_index(drop=True)
    return merged
