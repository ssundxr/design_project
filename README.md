# Privara Redaction API

Pure FastAPI backend service for image and PDF PII redaction.

## Overview

This repository is now backend-only:

- FastAPI service with Swagger UI
- Image redaction endpoint
- Multi-page PDF redaction endpoint
- JSON audit logs written to output folders

No separate website or Node backend is required.

## Features

- OCR-driven text detection
- Visual PII detection (signature, QR, face)
- PDF page-by-page redaction
- Output and audit files served from `/outputs`

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## API Docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Main Endpoints

- `GET /health`
- `POST /api/v1/redact/image`
- `POST /api/v1/redact/pdf`

## Notes

- Tesseract is used as OCR fallback and should be installed on host machines.
- PDF processing uses PyMuPDF (`pymupdf`) for page rendering and word-level extraction, then Gemini selects only PII tokens for redaction.
- Generated files are written under `output/`.

## Project Structure

```text
src/
  api/
    main.py
    schemas.py
  detector.py
  gemini_orchestrator.py
  layoutlm_detector.py
  ocr.py
  pdf_redactor.py
  redactor.py
main.py
requirements.txt
```
