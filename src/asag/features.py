from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FeatureBundle:
    X: np.ndarray
    vectorizer: TfidfVectorizer


def _build_pair_features(ref_tfidf, stu_tfidf, ref_text: pd.Series, stu_text: pd.Series) -> np.ndarray:
    cos_sim = cosine_similarity(ref_tfidf, stu_tfidf).diagonal()
    len_diff = ref_text.str.len().to_numpy() - stu_text.str.len().to_numpy()
    len_ratio = (stu_text.str.len().to_numpy() + 1.0) / (ref_text.str.len().to_numpy() + 1.0)
    return np.vstack([cos_sim, len_diff, len_ratio]).T


def fit_transform_features(df: pd.DataFrame, max_features: int, ngram_range: tuple[int, int]) -> FeatureBundle:
    corpus = pd.concat([df["reference_answer"], df["student_answer"]], axis=0).tolist()
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(corpus)

    ref_tfidf = vectorizer.transform(df["reference_answer"])
    stu_tfidf = vectorizer.transform(df["student_answer"])

    X = _build_pair_features(ref_tfidf, stu_tfidf, df["reference_answer"], df["student_answer"])
    return FeatureBundle(X=X, vectorizer=vectorizer)


def transform_features(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> np.ndarray:
    ref_tfidf = vectorizer.transform(df["reference_answer"])
    stu_tfidf = vectorizer.transform(df["student_answer"])
    return _build_pair_features(ref_tfidf, stu_tfidf, df["reference_answer"], df["student_answer"])
