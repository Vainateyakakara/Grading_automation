from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from asag.best_model import BestModelPredictor
from asag.inference import BaselinePredictor


DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "baseline"
DEFAULT_BEST_MODEL_DIR = PROJECT_ROOT / "artifacts" / "best_model"
FRONTEND_DIR = PROJECT_ROOT / "web" / "frontend"


class PredictRequest(BaseModel):
    reference_answer: str = Field(min_length=1)
    student_answer: str = Field(min_length=1)


class PredictResponse(BaseModel):
    predicted_score: float
    min_score: float
    max_score: float
    source_model: Optional[str] = None


app = FastAPI(title="ASAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_predictor(model_dir: Path = DEFAULT_MODEL_DIR) -> BaselinePredictor:
    if not model_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model directory not found: {model_dir}. Train baseline first.",
        )
    try:
        return BaselinePredictor.load(model_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")


def load_best_predictor(model_dir: Path = DEFAULT_BEST_MODEL_DIR) -> Optional[BestModelPredictor]:
    if not model_dir.exists():
        return None
    selection = model_dir / "model_selection.json"
    if not selection.exists():
        return None
    try:
        return BestModelPredictor(model_dir)
    except Exception:
        return None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info() -> dict:
    best = load_best_predictor()
    if best is not None:
        return {
            "model_type": "best_model_selector",
            "model_dir": str(DEFAULT_BEST_MODEL_DIR),
            "final_model": best.final_model,
            "min_score": best.ensemble_artifacts.data_min,
            "max_score": best.ensemble_artifacts.data_max,
        }

    predictor = load_predictor()
    return {
        "model_type": "baseline_ridge",
        "model_dir": str(DEFAULT_MODEL_DIR),
        "min_score": predictor.min_score,
        "max_score": predictor.max_score,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    best = load_best_predictor()
    if best is not None:
        out = best.predict(payload.reference_answer, payload.student_answer)
        return PredictResponse(
            predicted_score=float(out["predicted_score"]),
            min_score=float(out["min_score"]),
            max_score=float(out["max_score"]),
            source_model=str(out["source_model"]),
        )

    predictor = load_predictor()
    score = predictor.predict(payload.reference_answer, payload.student_answer)
    return PredictResponse(
        predicted_score=score,
        min_score=predictor.min_score,
        max_score=predictor.max_score,
        source_model="baseline",
    )


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)) -> dict:
    best = load_best_predictor()
    predictor = load_predictor() if best is None else None

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")

    required = {"reference_answer", "student_answer"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    if best is not None:
        preds = [
            best.predict(ref, stu)["predicted_score"]
            for ref, stu in zip(df["reference_answer"].astype(str), df["student_answer"].astype(str))
        ]
        min_score = best.ensemble_artifacts.data_min
        max_score = best.ensemble_artifacts.data_max
    else:
        preds = [
            predictor.predict(ref, stu)
            for ref, stu in zip(df["reference_answer"].astype(str), df["student_answer"].astype(str))
        ]
        min_score = predictor.min_score
        max_score = predictor.max_score

    out = df.copy()
    out["predicted_score"] = preds

    out_path = PROJECT_ROOT / "artifacts" / "batch_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    return {
        "rows": int(len(out)),
        "min_score": min_score,
        "max_score": max_score,
        "output_file": str(out_path),
    }


@app.get("/")
def root() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index)


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
