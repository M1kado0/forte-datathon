"""Synthetic credit-like data so the whole pipeline runs tonight. Delete once real data is wired."""

import numpy as np
import pandas as pd

from src.config import REGIONS, SEED


def make_stub(n: int = 5000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = np.round(rng.lognormal(mean=np.log(450_000), sigma=0.5, size=n), -3)
    debt_ratio = np.clip(rng.gamma(2.0, 0.22, n), 0, 4.9)
    overdue = np.where(rng.random(n) < 0.7, 0, rng.integers(1, 120, n))
    tenure = rng.integers(0, 180, n)
    region = rng.choice(REGIONS, n, p=[0.28, 0.22, 0.12, 0.1, 0.08, 0.2])

    logit = (
        -2.6
        + 1.4 * debt_ratio
        + 0.025 * overdue
        - 0.006 * tenure
        - 0.6 * (income > 600_000)
        + 0.3 * (age < 25)
    )
    p = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < p).astype(int)

    return pd.DataFrame(
        {
            "client_id": [f"KZ-{100000 + i}" for i in range(n)],
            "age": age,
            "income_kzt": income.astype(float),
            "debt_ratio": debt_ratio.round(3),
            "days_overdue_max": overdue,
            "tenure_months": tenure,
            "region": region,
            "target": target,
        }
    )
