from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import normalize_text


COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "reference_answer": ("reference_answer", "model_answer", "teacher_answer", "expected_answer", "answer_key"),
    "student_answer": ("student_answer", "answer", "student_response", "response"),
    "score": ("score", "label", "grade", "target"),
}


@dataclass
class DataSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def _resolve_column(df: pd.DataFrame, canonical: str) -> str:
    for c in COLUMN_ALIASES[canonical]:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required column for '{canonical}'. Supported aliases: {COLUMN_ALIASES[canonical]}")


def load_and_standardize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ref_col = _resolve_column(df, "reference_answer")
    stu_col = _resolve_column(df, "student_answer")
    score_col = _resolve_column(df, "score")

    keep_cols = [ref_col, stu_col, score_col]
    if "dataset" in df.columns:
        keep_cols.append("dataset")

    out = df[keep_cols].copy()
    rename_map = {
        ref_col: "reference_answer",
        stu_col: "student_answer",
        score_col: "score",
    }
    out = out.rename(columns=rename_map)

    out["reference_answer"] = out["reference_answer"].fillna("").map(normalize_text)
    out["student_answer"] = out["student_answer"].fillna("").map(normalize_text)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score"])
    out = out[(out["reference_answer"] != "") | (out["student_answer"] != "")]

    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].fillna("unknown").astype(str)

    return out.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float, val_size: float, random_state: int) -> DataSplits:
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    effective_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=effective_val_size, random_state=random_state)

    return DataSplits(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )
