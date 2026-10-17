"""Data contract, feature and leakage tests. Rewrite for the real data in hour 1."""

import pandas as pd
import pandera.errors
import pytest

from contracts.data import StubSchema
from src.config import FEATURES, ID_COL, TARGET
from src.data.stub import make_stub
from src.features.build import to_model_frame
from src.models.explain import fmt_kzt


def test_stub_matches_contract():
    StubSchema.validate(make_stub(200))


def test_contract_rejects_bad_rows():
    df = make_stub(10)
    df.loc[0, "age"] = 5
    with pytest.raises(pandera.errors.SchemaError):
        StubSchema.validate(df)


def test_no_leakage_columns_in_features():
    assert TARGET not in FEATURES and ID_COL not in FEATURES


def test_model_frame_categories_fixed():
    X = to_model_frame(pd.DataFrame([make_stub(1).iloc[0]]))
    assert list(X.columns) == FEATURES
    assert len(X["region"].cat.categories) == 6


def test_fmt_kzt_thin_space():
    assert fmt_kzt(1250000) == "1 250 000 ₸"
