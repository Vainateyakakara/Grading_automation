from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .cross_encoder import load_cross_encoder_predictor
from .ensemble import EnsembleArtifacts, load_ensemble, predict_ensemble


@dataclass
class BestModelArtifacts:
    final_model: str
    ensemble_metrics: Dict[str, float]
    cross_encoder_metrics: Dict[str, float] | None


def save_model_selection(
    out_dir: str | Path,
    ensemble_metrics: Dict[str, float],
    cross_encoder_metrics: Dict[str, float] | None,
) -> BestModelArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ce_qwk = cross_encoder_metrics["qwk"] if cross_encoder_metrics else float("-inf")
    ens_qwk = ensemble_metrics["qwk"]

    final_model = "cross_encoder" if ce_qwk > ens_qwk else "ensemble"

    payload = {
        "final_model": final_model,
        "ensemble_metrics": ensemble_metrics,
        "cross_encoder_metrics": cross_encoder_metrics,
    }

    with open(out_dir / "model_selection.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return BestModelArtifacts(
        final_model=final_model,
        ensemble_metrics=ensemble_metrics,
        cross_encoder_metrics=cross_encoder_metrics,
    )


class BestModelPredictor:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        with open(self.out_dir / "model_selection.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.final_model = str(meta["final_model"])
        self.ensemble_artifacts: EnsembleArtifacts = load_ensemble(self.out_dir / "ensemble")
        self.cross_encoder_predict = None

        ce_dir = self.out_dir / "cross_encoder"
        if ce_dir.exists() and (ce_dir / "cross_encoder_meta.json").exists():
            try:
                self.cross_encoder_predict = load_cross_encoder_predictor(ce_dir)
            except Exception:
                self.cross_encoder_predict = None

        if self.final_model == "cross_encoder" and self.cross_encoder_predict is None:
            self.final_model = "ensemble"

    def predict(self, reference: str, student: str) -> Dict:
        ens_score, ens_details = predict_ensemble(self.ensemble_artifacts, reference, student)

        ce_score = None
        if self.cross_encoder_predict is not None:
            try:
                ce_score = float(self.cross_encoder_predict(reference, student))
            except Exception:
                ce_score = None

        if self.final_model == "cross_encoder" and ce_score is not None:
            final_score = ce_score
            source = "cross_encoder"
        else:
            final_score = ens_score
            source = "ensemble"

        return {
            "predicted_score": float(final_score),
            "source_model": source,
            "ensemble_score": float(ens_score),
            "cross_encoder_score": ce_score,
            "details": ens_details,
            "min_score": self.ensemble_artifacts.data_min,
            "max_score": self.ensemble_artifacts.data_max,
        }
