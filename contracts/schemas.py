"""API contract. Change this file first, then tell P1 — the front-end mirrors it in web/src/api.ts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    model_loaded: bool


class PredictRequest(BaseModel):
    input: dict[str, Any] = Field(..., description="Feature name -> value, or {'client_id': ...}")


class Reason(BaseModel):
    feature: str
    effect: float = Field(..., description="Signed contribution (e.g. SHAP value); >0 pushes towards the label")
    text: str = Field(..., description="Plain-language Russian explanation for the UI")


class PredictResponse(BaseModel):
    result: float = Field(..., description="Main score, e.g. probability 0..1")
    label: str = Field(..., description="Human verdict, e.g. 'Высокий риск'")
    confidence: float = Field(..., ge=0, le=1)
    reasons: list[Reason] = Field(default_factory=list, max_length=5)
    action: str = Field(..., description="Recommended next step for the employee")
    impact_kzt: float = Field(..., description="Expected ₸ effect of the action")


class ChartPoint(BaseModel):
    x: str | float
    y: float
    highlight: bool = False


class Insight(BaseModel):
    id: str
    title: str = Field(..., description="Takeaway title, not a variable name")
    number: str = Field(..., description="The big number, already formatted")
    chart_data: list[ChartPoint] = Field(default_factory=list)
    takeaway: str


class FieldMeta(BaseModel):
    name: str
    label: str
    type: Literal["number", "category", "text"]
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    default: Any = None
    whatif: bool = False


class Meta(BaseModel):
    product_name: str
    primary_field: str
    fields: list[FieldMeta]
