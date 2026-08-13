# Project Structure

```text
Automatic_Short_Answer_Grading/
  LICENSE
  README.md
  CONTRIBUTING.md
  PROJECT_STRUCTURE.md
  requirements.txt
  .gitignore

  configs/
    baseline.yaml
    bert.yaml
    ensemble.yaml
    cross_encoder.yaml

  data/
    raw/
    processed/

  artifacts/
    baseline/
    best_model/

  scripts/
    prepare_sample_data.py
    prepare_demo_sources.py
    merge_datasets.py
    train_baseline.py
    train_bert.py
    train_best_model.py
    evaluate.py
    predict_baseline.py
    predict_best_model.py
    run_full_pipeline.py
    run_best_pipeline.py
    run_fullstack.sh

  src/asag/
    __init__.py
    config.py
    preprocess.py
    data.py
    dataset_adapters.py
    features.py
    baseline.py
    bert.py
    ensemble.py
    cross_encoder.py
    best_model.py
    inference.py
    metrics.py

  streamlit_app.py

  web/
    backend/
      requirements.txt
      app/
        __init__.py
        main.py
    frontend/
      index.html
      styles.css
      app.js
```

## Architecture Summary

- Data ingestion and normalization are handled in `src/asag/data.py` and `src/asag/dataset_adapters.py`.
- Baseline model training is in `scripts/train_baseline.py`.
- Best-model training (ensemble + optional cross-encoder + model selection) is in `scripts/train_best_model.py`.
- API and website are served through FastAPI (`web/backend/app/main.py`) and static frontend assets (`web/frontend/`).
- Streamlit UI is available in `streamlit_app.py`.
