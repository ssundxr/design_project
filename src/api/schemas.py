from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "privara-redaction-api"
    version: str = "1.0.0"


class RedactionResponse(BaseModel):
    filename: str
    content_type: str
    output_file: str
    output_url: str
    audit_file: Optional[str] = None
    audit_url: Optional[str] = None
    processing_time: float = Field(default=0.0)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    risk_assessment: Optional[str] = None
    detections: int = 0