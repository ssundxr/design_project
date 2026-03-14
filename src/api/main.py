import re
from io import BytesIO
from pathlib import Path
from typing import Final

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from ..pdf_redactor import PDFRedactor
from ..redactor import PIIRedactor
from .schemas import HealthResponse, RedactionResponse


APP_TITLE: Final = "Privara Redaction API"
APP_VERSION: Final = "1.0.0"
OUTPUT_DIR: Final = Path("output")
UPLOAD_DIR: Final = OUTPUT_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Backend-only PII redaction service for image and PDF documents. "
        "Swagger UI is available at /docs."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

image_redactor = PIIRedactor(output_dir=str(OUTPUT_DIR))
pdf_redactor = PDFRedactor(output_dir=str(OUTPUT_DIR))


def _safe_filename(filename: str, fallback: str) -> str:
    candidate = filename or fallback
    candidate = Path(candidate).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
    return sanitized or fallback


def _audit_url(audit_file: str | None) -> str | None:
    if not audit_file:
        return None
    if audit_file.startswith("pdf_"):
        return f"/outputs/pdf_audit_logs/{audit_file}"
    return f"/outputs/audit_logs/{audit_file}"


@app.get("/", response_model=HealthResponse, tags=["system"])
def root() -> HealthResponse:
    return HealthResponse(version=APP_VERSION)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(version=APP_VERSION)


@app.post("/api/v1/redact/image", response_model=RedactionResponse, tags=["redaction"])
async def redact_image(file: UploadFile = File(...)) -> RedactionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing image filename")

    safe_name = _safe_filename(file.filename, "document.png")
    try:
        payload = await file.read()
        image = Image.open(BytesIO(payload))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported image file") from exc

    image = image.convert("RGB")
    redacted_image, audit_data = image_redactor.redact_image(image, filename=safe_name)
    output_filename = f"redacted_{Path(safe_name).stem}.png"
    output_path = image_redactor.save_redacted_image(redacted_image, output_filename)

    return RedactionResponse(
        filename=safe_name,
        content_type=file.content_type or "image/png",
        output_file=output_path.name,
        output_url=f"/outputs/redacted/{output_path.name}",
        audit_file=audit_data.get("audit_file"),
        audit_url=_audit_url(audit_data.get("audit_file")),
        processing_time=audit_data.get("processing_time", 0.0),
        statistics=audit_data.get("statistics", {}),
        risk_assessment=audit_data.get("risk_assessment"),
        detections=len(audit_data.get("detections", [])),
    )


@app.post("/api/v1/redact/pdf", response_model=RedactionResponse, tags=["redaction"])
async def redact_pdf(file: UploadFile = File(...)) -> RedactionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing PDF filename")

    safe_name = _safe_filename(file.filename, "document.pdf")
    if Path(safe_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported on this endpoint")

    upload_path = UPLOAD_DIR / safe_name
    upload_path.write_bytes(await file.read())

    try:
        output_path, audit_data = pdf_redactor.redact_pdf(str(upload_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_file = Path(output_path).name
    return RedactionResponse(
        filename=safe_name,
        content_type=file.content_type or "application/pdf",
        output_file=output_file,
        output_url=f"/outputs/pdf_redacted/{output_file}",
        audit_file=audit_data.get("audit_file"),
        audit_url=_audit_url(audit_data.get("audit_file")),
        processing_time=audit_data.get("processing_time", 0.0),
        statistics=audit_data.get("statistics", {}),
        detections=audit_data.get("total_detections", 0),
    )