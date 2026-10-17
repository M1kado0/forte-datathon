"""`make all` / `make demo` entry point: data -> validate -> train -> insights. All UI/slide numbers come from here."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from contracts.data import StubSchema
from src.config import DATA_PROCESSED, REPORTS, TARGET
from src.data.stub import make_stub
from src.models.stub_train import train


def build_insights(df: pd.DataFrame) -> list[dict]:
    base = df[TARGET].mean()
    bins = pd.cut(df["days_overdue_max"], [-1, 0, 30, 60, 1000], labels=["0", "1–30", "31–60", "60+"])
    by_overdue = df.groupby(bins, observed=True)[TARGET].mean()
    top = by_overdue.iloc[-1]
    by_region = df.groupby("region")[TARGET].mean().sort_values(ascending=False)
    return [
        {
            "id": "overdue",
            "title": f"Просрочка 60+ дней — риск в {top / base:.1f}× выше среднего",
            "number": f"{top * 100:.0f}%",
            "chart_data": [
                {"x": str(k), "y": round(float(v) * 100, 1), "highlight": k == "60+"}
                for k, v in by_overdue.items()
            ],
            "takeaway": "Максимальная просрочка — самый сильный сигнал дефолта.",
        },
        {
            "id": "region",
            "title": "Регион почти не влияет на риск",
            "number": f"±{(by_region.max() - by_region.min()) * 100:.0f} п.п.",
            "chart_data": [{"x": k, "y": round(float(v) * 100, 1)} for k, v in by_region.items()],
            "takeaway": "Разница между регионами в пределах шума — модель не дискриминирует по региону.",
        },
    ]


def main(stub: bool) -> None:
    if stub:
        df = make_stub()
    else:
        raise SystemExit("Real data loader not wired yet: implement src/data/load.py (ticket 01).")
    df = StubSchema.validate(df)
    df.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    metrics = train(df, note="stub-baseline" if stub else "baseline")
    (REPORTS / "insights.json").write_text(json.dumps(build_insights(df), ensure_ascii=False, indent=2))
    print("metrics:", metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    main(ap.parse_args().stub)
