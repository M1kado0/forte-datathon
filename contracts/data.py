"""Data contracts (pandera). P3 rewrites these for the real dataset in hour 1."""

import pandera.pandas as pa

# Stub schema for the synthetic demo data. Replace columns once the case data is known.
StubSchema = pa.DataFrameSchema(
    {
        "client_id": pa.Column(str, unique=True),
        "age": pa.Column(int, pa.Check.in_range(18, 90)),
        "income_kzt": pa.Column(float, pa.Check.ge(0)),
        "debt_ratio": pa.Column(float, pa.Check.in_range(0, 5)),
        "days_overdue_max": pa.Column(int, pa.Check.ge(0)),
        "tenure_months": pa.Column(int, pa.Check.ge(0)),
        "region": pa.Column(str),
        "target": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)
