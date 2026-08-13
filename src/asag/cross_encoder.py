from __future__ import annotations

import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .metrics import evaluate_regression, qwk_numpy


@dataclass
class CrossEncoderArtifacts:
    model_dir: Path
    val_metrics: Dict[str, float]
    data_min: float
    data_max: float


def _build_ce_dataframe(df: pd.DataFrame, data_min: float, data_max: float) -> pd.DataFrame:
    out = df[["reference_answer", "student_answer", "score"]].copy()
    out = out.rename(columns={"reference_answer": "reference", "student_answer": "student"})
    out["score_norm"] = (out["score"] - data_min) / max((data_max - data_min), 1e-9)
    out["score_norm"] = out["score_norm"].clip(0.0, 1.0)
    return out


def _add_synthetic_extremes(df_ce: pd.DataFrame, n_synthetic: int, random_state: int) -> pd.DataFrame:
    n = min(int(n_synthetic), len(df_ce))
    refs = df_ce["reference"].astype(str).tolist()[:n]
    if len(refs) == 0:
        refs = df_ce["student"].astype(str).tolist()[:n]

    synth_perfect = pd.DataFrame({"reference": refs, "student": refs, "score_norm": [1.0] * len(refs)})
    synth_bad = pd.DataFrame({"reference": refs, "student": [""] * len(refs), "score_norm": [0.0] * len(refs)})

    aug = pd.concat([df_ce, synth_perfect, synth_bad], ignore_index=True)
    return aug.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_cross_encoder(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Dict,
    out_dir: str | Path,
) -> CrossEncoderArtifacts:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Cross-encoder dependencies missing. Install transformers/datasets/torch to train it."
        ) from exc

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    model_name = str(cfg.get("model_name", "microsoft/deberta-v3-small"))
    epochs = int(cfg.get("num_train_epochs", 5))
    batch_size = int(cfg.get("per_device_train_batch_size", 16))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    lr = float(cfg.get("learning_rate", 2e-5))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_length = int(cfg.get("max_length", 256))
    random_state = int(cfg.get("random_state", 42))
    n_synthetic = int(cfg.get("synthetic_extremes", 500))
    fp16 = bool(cfg.get("fp16", False))
    save_total_limit = int(cfg.get("save_total_limit", 2))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_min = float(min(train_df["score"].min(), val_df["score"].min()))
    data_max = float(max(train_df["score"].max(), val_df["score"].max()))

    ce_train = _build_ce_dataframe(train_df, data_min, data_max)
    ce_val = _build_ce_dataframe(val_df, data_min, data_max)
    ce_train = _add_synthetic_extremes(ce_train, n_synthetic=n_synthetic, random_state=random_state)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds_train = Dataset.from_pandas(ce_train[["reference", "student", "score_norm"]])
    ds_val = Dataset.from_pandas(ce_val[["reference", "student", "score_norm"]])

    def preprocess_fn(examples):
        tokenized = tokenizer(
            examples["reference"],
            examples["student"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        tokenized["labels"] = [float(x) for x in examples["score_norm"]]
        return tokenized

    ds_train = ds_train.map(preprocess_fn, batched=True)
    ds_val = ds_val.map(preprocess_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    config = AutoConfig.from_pretrained(model_name, problem_type="regression", num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    train_examples = len(ds_train)
    steps_per_epoch = max(1, math.ceil(train_examples / batch_size))
    updates_per_epoch = max(1, math.ceil(steps_per_epoch / grad_accum))
    total_updates = max(1, epochs * updates_per_epoch)

    kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=int(total_updates),
        weight_decay=weight_decay,
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        fp16=fp16,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    argnames = list(sig.parameters.keys())
    if "evaluation_strategy" in argnames:
        kwargs["evaluation_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"
    elif "eval_strategy" in argnames:
        kwargs["eval_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"
    else:
        kwargs["do_eval"] = True

    if "report_to" in argnames:
        kwargs["report_to"] = "none"

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds).reshape(-1)
        labels = np.asarray(labels).reshape(-1)

        preds_orig = np.clip(preds, 0.0, 1.0) * (data_max - data_min) + data_min
        labels_orig = np.clip(labels, 0.0, 1.0) * (data_max - data_min) + data_min

        metrics = evaluate_regression(
            labels_orig,
            preds_orig,
            min_score=data_min,
            max_score=data_max,
        )
        return metrics

    args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # validation in original scale
    pred_out = trainer.predict(ds_val)
    raw_preds = np.asarray(pred_out.predictions).reshape(-1)
    val_preds = np.clip(raw_preds, 0.0, 1.0) * (data_max - data_min) + data_min
    val_true = ce_val["score_norm"].to_numpy() * (data_max - data_min) + data_min

    val_metrics = evaluate_regression(val_true, val_preds, min_score=data_min, max_score=data_max)

    with open(out_dir / "cross_encoder_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_min": data_min,
                "data_max": data_max,
                "val_metrics": val_metrics,
                "model_name": model_name,
                "max_length": max_length,
            },
            f,
            indent=2,
        )

    return CrossEncoderArtifacts(
        model_dir=out_dir,
        val_metrics=val_metrics,
        data_min=data_min,
        data_max=data_max,
    )


def load_cross_encoder_predictor(model_dir: str | Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    with open(model_dir / "cross_encoder_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    data_min = float(meta["data_min"])
    data_max = float(meta["data_max"])
    max_length = int(meta.get("max_length", 256))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def predict(reference: str, student: str) -> float:
        inputs = tokenizer(
            str(reference),
            str(student),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            pred = out.logits.squeeze().detach().cpu().numpy()
        pred = np.asarray(pred).reshape(-1)
        pred = np.clip(pred, 0.0, 1.0)
        pred_orig = pred * (data_max - data_min) + data_min
        return float(pred_orig[0])

    return predict
