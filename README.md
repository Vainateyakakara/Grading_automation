# Automatic Short Answer Grading (ASAG)

Unified best-version project combining both implementations:
- Notebook-style similarity ensemble (TF-IDF Ridge + calibrated similarity + synthetic extremes + gibberish guard)
- DeBERTa cross-encoder training path (normalized labels + synthetic extremes)
- Automatic best-model selection by validation QWK (`cross_encoder` vs `ensemble`)
- Fullstack website (FastAPI + polished frontend)
- Streamlit app

## Setup

```bash
cd Automatic_Short_Answer_Grading
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r web/backend/requirements.txt
```

## Train Best Combined Model

Demo run:

```bash
python3 scripts/run_best_pipeline.py
```

If your environment cannot train cross-encoder now:

```bash
python3 scripts/run_best_pipeline.py --skip-cross-encoder
```

Real data run:

```bash
python3 scripts/run_best_pipeline.py --real-data-dir /absolute/path/to/your/datasets
```

Artifacts:
- `artifacts/best_model/ensemble/`
- `artifacts/best_model/cross_encoder/` (if trained)
- `artifacts/best_model/model_selection.json`

## Fullstack Website

```bash
./scripts/run_fullstack.sh
```

Open `http://127.0.0.1:8000`

Backend automatically uses `artifacts/best_model` if available, otherwise falls back to baseline.

## Streamlit

```bash
streamlit run streamlit_app.py
```

## Main Commands

Merge data:

```bash
python3 scripts/merge_datasets.py --data-dir data/raw/demo_sources --out data/processed/merged_asag.csv
```

Train combined best model:

```bash
python3 scripts/train_best_model.py \
  --data data/processed/merged_asag.csv \
  --ensemble-config configs/ensemble.yaml \
  --cross-config configs/cross_encoder.yaml \
  --out artifacts/best_model
```

Predict with selected best model:

```bash
python3 scripts/predict_best_model.py \
  --model-dir artifacts/best_model \
  --reference "The capital of France is Paris." \
  --student "Paris is the capital city of France."
```
