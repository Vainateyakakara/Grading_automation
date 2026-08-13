from __future__ import annotations

from pathlib import Path

import streamlit as st

import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from asag.best_model import BestModelPredictor
from asag.inference import BaselinePredictor


st.set_page_config(page_title="ASAG Grader", page_icon="A", layout="wide")

st.title("Automatic Short Answer Grader")
st.caption("Baseline model: TF-IDF + similarity features + Ridge Regression")

with st.sidebar:
    st.header("Model")
    model_dir = st.text_input("Model directory", value=str(ROOT / "artifacts" / "best_model"))
    load_clicked = st.button("Load model")

if "predictor" not in st.session_state:
    st.session_state.predictor = None

if load_clicked:
    try:
        sel = Path(model_dir) / "model_selection.json"
        if sel.exists():
            st.session_state.predictor = BestModelPredictor(model_dir)
            st.success("Best-model selector loaded")
        else:
            st.session_state.predictor = BaselinePredictor.load(model_dir)
            st.success("Baseline model loaded")
    except Exception as e:
        st.session_state.predictor = None
        st.error(f"Failed to load model: {e}")

col1, col2 = st.columns(2)

with col1:
    reference = st.text_area(
        "Reference answer",
        value="Photosynthesis is the process by which plants convert light energy into chemical energy.",
        height=180,
    )

with col2:
    student = st.text_area(
        "Student answer",
        value="Plants use sunlight to make food through photosynthesis.",
        height=180,
    )

if st.button("Predict score", type="primary"):
    if st.session_state.predictor is None:
        st.warning("Load a baseline model first from the sidebar.")
    else:
        if isinstance(st.session_state.predictor, BestModelPredictor):
            out = st.session_state.predictor.predict(reference, student)
            st.metric("Predicted score", f"{out['predicted_score']:.3f}")
            st.write(
                {
                    "source_model": out["source_model"],
                    "ensemble_score": out["ensemble_score"],
                    "cross_encoder_score": out["cross_encoder_score"],
                    "min_score": out["min_score"],
                    "max_score": out["max_score"],
                }
            )
        else:
            pred = st.session_state.predictor.predict(reference, student)
            st.metric("Predicted score", f"{pred:.3f}")
            st.write(
                {
                    "min_score": st.session_state.predictor.min_score,
                    "max_score": st.session_state.predictor.max_score,
                }
            )

st.divider()
st.subheader("Quick commands")
st.code(
    "\n".join(
        [
            "python3 scripts/prepare_demo_sources.py --out-dir data/raw/demo_sources --replicate 50",
            "python3 scripts/merge_datasets.py --data-dir data/raw/demo_sources --out data/processed/merged_asag.csv",
            "python3 scripts/train_best_model.py --data data/processed/merged_asag.csv --ensemble-config configs/ensemble.yaml --cross-config configs/cross_encoder.yaml --out artifacts/best_model",
            "streamlit run streamlit_app.py",
        ]
    ),
    language="bash",
)
