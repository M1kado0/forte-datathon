"""FastAPI app. Serves the API under /api and the built front-end (web/dist) on the same port."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from contracts.schemas import (
    FieldMeta,
    Health,
    Insight,
    Meta,
    PredictRequest,
    PredictResponse,
    Reason,
)
from src.config import (
    DATA_PROCESSED,
    FEATURE_LABELS,
    FEATURES,
    ID_COL,
    PRODUCT_NAME,
    REGIONS,
    REPORTS,
    VERSION,
    WEB_DIST,
)
from src.features.build import to_model_frame
from src.models.explain import impact_kzt, reason_text, verdict
from src.models.train import load_artifacts, predict_with_reasons

app = FastAPI(title=PRODUCT_NAME, version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: dict = {"art": None, "clients": None}


@app.on_event("startup")
def _load() -> None:
    STATE["art"] = load_artifacts()
    path = DATA_PROCESSED / "train.parquet"
    if path.exists():
        STATE["clients"] = pd.read_parquet(path).set_index(ID_COL)


DEFAULTS = {
    "age": 35, "income_kzt": 450_000.0, "debt_ratio": 0.4,
    "days_overdue_max": 0, "tenure_months": 24, "region": "Алматы",
}


@app.get("/api/health", response_model=Health)
def health() -> Health:
    return Health(version=VERSION, model_loaded=STATE["art"] is not None)


@app.get("/api/meta", response_model=Meta)
def meta() -> Meta:
    fields = [
        FieldMeta(name="age", label=FEATURE_LABELS["age"], type="number", min=18, max=90, step=1),
        FieldMeta(name="income_kzt", label=FEATURE_LABELS["income_kzt"], type="number", min=0, max=5_000_000, step=10_000),
        FieldMeta(name="debt_ratio", label=FEATURE_LABELS["debt_ratio"], type="number", min=0, max=3, step=0.05),
        FieldMeta(name="days_overdue_max", label=FEATURE_LABELS["days_overdue_max"], type="number", min=0, max=120, step=1, whatif=True),
        FieldMeta(name="tenure_months", label=FEATURE_LABELS["tenure_months"], type="number", min=0, max=240, step=1),
        FieldMeta(name="region", label=FEATURE_LABELS["region"], type="category", options=REGIONS),
    ]
    for f in fields:
        f.default = DEFAULTS[f.name]
    return Meta(product_name=PRODUCT_NAME, primary_field=ID_COL, fields=fields)


@app.get("/api/clients/{client_id}")
def client(client_id: str) -> dict:
    df = STATE["clients"]
    if df is None or client_id not in df.index:
        raise HTTPException(404, "Клиент не найден")
    row = df.loc[client_id, FEATURES].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    art = STATE["art"]
    if art is None:
        raise HTTPException(503, "Модель не загружена: запустите `make demo`")
    data = dict(DEFAULTS)
    cid = req.input.get(ID_COL)
    if cid:
        data.update(client(str(cid)))
    data.update({k: v for k, v in req.input.items() if k in FEATURES and v is not None})

    X = to_model_frame(pd.DataFrame([data]))
    p, contrib = predict_with_reasons(art, X)
    p0 = float(p[0])
    order = sorted(range(len(FEATURES)), key=lambda i: -abs(contrib[0, i]))[:5]
    reasons = [
        Reason(feature=FEATURES[i], effect=round(float(contrib[0, i]), 4),
               text=reason_text(FEATURES[i], data[FEATURES[i]], float(contrib[0, i])))
        for i in order
    ]
    label, action = verdict(p0)
    return PredictResponse(
        result=round(p0, 4), label=label, confidence=round(max(p0, 1 - p0), 4),
        reasons=reasons, action=action, impact_kzt=round(impact_kzt(p0, float(data["income_kzt"])), 0),
    )


@app.get("/api/insights", response_model=list[Insight])
def insights() -> list[Insight]:
    path = REPORTS / "insights.json"
    return json.loads(path.read_text()) if path.exists() else []


# ---- front-end (must be last) ----
if (WEB_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = WEB_DIST / path
        return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")
