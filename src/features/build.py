"""Pure feature functions. Every function here gets a unit test."""

import pandas as pd

from src.config import CATEGORICAL, FEATURES, REGIONS


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features and fix categorical dtypes (same categories at train and serve time)."""
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=REGIONS)
    return X
