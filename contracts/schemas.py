"""Video-analysis wire contract from brief section 10."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Health(Contract):
    status: Literal["ok"] = "ok"
    version: str
    model_loaded: bool


class EvidenceFrame(Contract):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str = Field(min_length=1)


class Violation(Contract):
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def matching_timecode(self):
        # Decoding snaps evidence to an actual frame; allow sub-second rounding.
        if abs(self.timecode_sec - self.evidence_frames[0].timestamp_sec) >= 1.0:
            raise ValueError("First evidence frame must be less than one second from the violation timecode")
        return self


class AnalyzeResponse(Contract):
    job_id: str


class JobResponse(AnalyzeResponse):
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: list[Violation] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_result(self):
        expected = ("violation" if self.violations else "no_violation") if self.status == "done" else None
        if self.verdict != expected or (self.status != "done" and self.violations):
            raise ValueError("Verdict and violations must match job status")
        return self
