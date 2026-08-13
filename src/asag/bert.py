from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class BertArtifacts:
    model_dir: Path


def _build_text(df: pd.DataFrame) -> list[str]:
    return [f"reference: {r} [SEP] student: {s}" for r, s in zip(df["reference_answer"], df["student_answer"])]


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "text": _build_text(df),
        "labels": df["score"].astype(float).tolist(),
    })


def _compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.squeeze(preds)
    mse = mean_squared_error(labels, preds)
    return {"mse": float(mse)}


def train_bert_regressor(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Dict, out_dir: str) -> BertArtifacts:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = _to_hf_dataset(train_df)
    val_ds = _to_hf_dataset(val_df)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=int(cfg["max_length"]))

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )

    args = TrainingArguments(
        output_dir=str(out_path),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=int(cfg["logging_steps"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(cfg["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    return BertArtifacts(model_dir=out_path)
