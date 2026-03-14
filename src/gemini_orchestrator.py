"""
Gemini Orchestrator - Privacy-Safe Document Classification & XAI Audit
Ensures DPDP Act 2023 compliance by only using Gemini for non-PII metadata analysis.
"""

import json
import logging
import os
import time
import re as _re
from typing import Dict, Any, List
from io import BytesIO
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class GeminiOrchestrator:
    """
    Handles zero-PII interactions with Gemini Vision API.
    1. Document Classification (using heavily downscaled thumbnails)
    2. Explainable AI Audit Generation (using anonymized metadata)
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key and GENAI_AVAILABLE:
            logger.warning("GEMINI_API_KEY not set. Orchestrator offline.")
        
        self.client = genai.Client(api_key=self.api_key) if self.api_key and GENAI_AVAILABLE else None
        self.model_name = model_name
        self.last_error = None

    def _get_pii_models(self) -> List[str]:
        """Resolve model list for token-level PII detection.

        Default to the configured model only. Extra fallbacks can be provided via
        GEMINI_PII_MODELS as a comma-separated list.
        """
        raw = os.environ.get("GEMINI_PII_MODELS", "")
        extras = [m.strip() for m in raw.split(",") if m.strip()]
        models: List[str] = []
        for m in [self.model_name] + extras:
            if m and m not in models:
                models.append(m)
        return models

    def classify_document(self, image: Image.Image) -> str:
        """
        Classify document type using a low-res, blurred thumbnail ensuring PII is illegible.
        Returns: Document category (e.g., 'AADHAAR', 'PAN', 'MEDICAL_RECORD', 'COURT_ORDER', 'OTHER')
        """
        if not self.client:
            return "UNKNOWN"

        try:
            # Downscale and blur to destroy PII but keep layout structure
            thumb = image.copy()
            thumb.thumbnail((256, 256))
            from PIL import ImageFilter
            thumb = thumb.filter(ImageFilter.GaussianBlur(radius=2))
            
            img_bytes = BytesIO()
            thumb.convert('RGB').save(img_bytes, format='JPEG', quality=60)
            img_bytes.seek(0)
            
            prompt = (
                "Analyze the general layout and visible non-sensitive structural elements of this document thumbnail. "
                "Classify it into EXACTLY ONE of these categories: AADHAAR, PAN, PASSPORT, MEDICAL_RECORD, "
                "BANK_STATEMENT, TAX_FORM, COURT_DOCUMENT, INVOICE, DRIVING_LICENSE, VOTER_ID, OTHER. "
                "Reply with ONLY the category name. No explanations."
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes.read(), mime_type="image/jpeg"),
                    prompt
                ]
            )
            
            classification = response.text.strip().upper()
            logger.info(f"Document classified as: {classification}")
            return classification
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "UNKNOWN"

    def generate_audit_explanation(self, entity_type: str, context_metadata: str, confidence: float) -> str:
        """
        Generate human-readable DPDP compliance explanation using ANONYMIZED metadata.
        Never sends actual PII string.
        """
        if not self.client:
            return f"Redacted {entity_type} (Conf: {confidence:.2f})"

        try:
            prompt = (
                f"You are a DPDP Act compliance auditor. Explain why a detection system redacted an entity based on the following metadata.\n"
                f"Entity Type: {entity_type}\n"
                f"Context Found: {context_metadata}\n"
                f"Confidence Score: {confidence:.2f}\n"
                f"\nWrite a single, formal sentence explaining the redaction reason for audit logs. "
                f"Example: 'Redacted a 12-digit sequence consistent with Aadhaar format detected near the keyword \"Govt of India\" with high confidence.'"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Audit generation failed: {e}")
            return f"Redacted {entity_type} based on localized heuristic patterns."

    def detect_pii_tokens(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify PII tokens from OCR/PDF extracted text.

        Args:
            tokens: List like [{'id': 1, 'text': 'John', 'bbox': [x1,y1,x2,y2]}, ...]

        Returns:
            List of detections: [{'id': int, 'entity': str, 'risk': str, 'confidence': float}]
        """
        self.last_error = None
        if not self.client or not tokens:
            return []

        try:
            compact_tokens = [{'id': t.get('id'), 'text': str(t.get('text', ''))[:120]} for t in tokens if t.get('text')]

            prompt = (
                "You are a strict PII detection engine for document redaction. "
                "Given OCR/PDF tokens, identify ONLY sensitive personal data that must be redacted. "
                "Allowed entity labels: PERSON_NAME, EMAIL, PHONE, AADHAAR, PAN, PASSPORT, SSN, CREDIT_CARD, ACCOUNT, ADDRESS, SIGNATURE_TEXT, BIOMETRIC_TEXT. "
                "Do NOT mark generic headers, labels, or non-sensitive words. "
                "Return JSON only with this schema: "
                "{\"detections\":[{\"id\":<int>,\"entity\":<label>,\"risk\":\"HIGH|MEDIUM|LOW\",\"confidence\":<0-1 float>}]}"
            )

            normalized = []
            chunk_size = 120
            models = self._get_pii_models()

            for i in range(0, len(compact_tokens), chunk_size):
                chunk = compact_tokens[i:i + chunk_size]
                payload = json.dumps(chunk, ensure_ascii=False)
                response = None
                last_exc = None

                for model in models:
                    try:
                        response = self.client.models.generate_content(
                            model=model,
                            contents=[prompt, "TOKENS:", payload]
                        )
                        break
                    except Exception as e:
                        last_exc = e
                        err = str(e).upper()
                        # Quota/auth failures will not improve by trying a second model.
                        if "429" in err or "RESOURCE_EXHAUSTED" in err or "401" in err or "PERMISSION_DENIED" in err:
                            break
                        continue

                if response is None:
                    if last_exc is not None:
                        self.last_error = str(last_exc)
                        logger.warning(f"Gemini token chunk failed; returning partial detections: {last_exc}")
                        return normalized
                    continue

                text = (response.text or "").strip()
                start = text.find('{')
                end = text.rfind('}')
                if start == -1 or end == -1 or end <= start:
                    continue

                parsed = json.loads(text[start:end + 1])
                detections = parsed.get('detections', [])
                if not isinstance(detections, list):
                    continue

                for d in detections:
                    try:
                        token_id = int(d.get('id'))
                        entity = str(d.get('entity', 'PII')).upper()
                        risk = str(d.get('risk', 'MEDIUM')).upper()
                        confidence = float(d.get('confidence', 0.8))
                        confidence = max(0.0, min(1.0, confidence))
                        if risk not in {'HIGH', 'MEDIUM', 'LOW'}:
                            risk = 'MEDIUM'
                        normalized.append({
                            'id': token_id,
                            'entity': entity,
                            'risk': risk,
                            'confidence': confidence,
                        })
                    except Exception:
                        continue

            return normalized
        except Exception as e:
            logger.error(f"Gemini PII detection failed: {e}")
            self.last_error = str(e)
            return []

    def analyze_document_for_pii(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Send the full document image to Gemini Vision to identify all PII text.

        Returns a list of dicts:
            [{'text': 'SHYAM SUNDER', 'entity': 'PERSON_NAME', 'risk': 'HIGH', 'confidence': 0.95}, ...]
        """
        self.last_error = None
        if not self.client:
            return []

        try:
            # Convert image to JPEG bytes for the API
            img_bytes = BytesIO()
            image.convert('RGB').save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)

            prompt = (
                "You are a strict PII detection engine for document redaction under India's DPDP Act 2023.\n"
                "Analyze this document image and identify ALL personally identifiable information (PII).\n\n"
                "For each PII item found, return the EXACT text as it appears in the document.\n\n"
                "PII categories to detect:\n"
                "- PERSON_NAME: Full names, first names, last names, father's name, mother's name, spouse name\n"
                "- DATE_OF_BIRTH: Any date that represents a birth date\n"
                "- EMAIL: Email addresses\n"
                "- PHONE: Phone numbers, mobile numbers\n"
                "- AADHAAR: Aadhaar numbers (12 digits, may be partially masked)\n"
                "- PAN: PAN card numbers\n"
                "- PASSPORT: Passport numbers\n"
                "- ADDRESS: Full addresses, street, city, pin code, district\n"
                "- PLACE_OF_BIRTH: Birth place/city\n"
                "- ACCOUNT: Bank account numbers\n"
                "- CREDIT_CARD: Credit/debit card numbers\n"
                "- SSN: Social security numbers\n"
                "- ID_NUMBER: Any government ID numbers (CSE, ARN, counter numbers with personal linkage)\n\n"
                "Do NOT flag:\n"
                "- Document titles, headers, form labels, instructions\n"
                "- Government body names (e.g. 'Ministry of External Affairs')\n"
                "- Generic words like 'FRESH', 'NORMAL', 'MALE', 'SINGLE', 'BIRTH', 'STUDENT', 'YES', 'NO'\n"
                "- Non-personal categorical values (gender, marital status, employment type, etc.)\n\n"
                "Return ONLY valid JSON with this schema:\n"
                "{\"pii_items\":[{\"text\":\"<exact text>\",\"entity\":\"<label>\",\"risk\":\"HIGH|MEDIUM|LOW\",\"confidence\":<0-1 float>}]}\n"
                "If no PII is found, return: {\"pii_items\":[]}"
            )

            models = self._get_pii_models()
            response = None
            last_exc = None
            max_retries = 3

            for attempt in range(max_retries):
                for model in models:
                    try:
                        img_bytes.seek(0)
                        response = self.client.models.generate_content(
                            model=model,
                            contents=[
                                types.Part.from_bytes(data=img_bytes.read(), mime_type="image/jpeg"),
                                prompt,
                            ],
                        )
                        break
                    except Exception as e:
                        last_exc = e
                        err = str(e).upper()
                        if "401" in err or "PERMISSION_DENIED" in err:
                            # Auth errors won't recover with retry
                            response = None
                            break
                        if "429" in err or "RESOURCE_EXHAUSTED" in err:
                            # Extract retry delay from error message
                            delay_match = _re.search(r'retry in ([\d.]+)', str(e), _re.IGNORECASE)
                            wait = float(delay_match.group(1)) if delay_match else (2 ** attempt * 5)
                            wait = min(wait, 30)  # cap at 30s
                            logger.info(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait:.1f}s...")
                            time.sleep(wait)
                            break  # break model loop to retry
                        continue

                if response is not None:
                    break
                # If auth error, don't retry
                if last_exc and ("401" in str(last_exc).upper() or "PERMISSION_DENIED" in str(last_exc).upper()):
                    break

            if response is None:
                if last_exc:
                    self.last_error = str(last_exc)
                    logger.warning(f"Gemini Vision PII analysis failed after {max_retries} attempts: {last_exc}")
                return []

            text = (response.text or "").strip()
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1 or end <= start:
                logger.warning(f"Gemini Vision returned non-JSON: {text[:200]}")
                return []

            parsed = json.loads(text[start:end + 1])
            pii_items = parsed.get('pii_items', [])
            if not isinstance(pii_items, list):
                return []

            normalized = []
            for item in pii_items:
                try:
                    pii_text = str(item.get('text', '')).strip()
                    if not pii_text:
                        continue
                    entity = str(item.get('entity', 'PII')).upper()
                    risk = str(item.get('risk', 'HIGH')).upper()
                    confidence = float(item.get('confidence', 0.90))
                    confidence = max(0.0, min(1.0, confidence))
                    if risk not in {'HIGH', 'MEDIUM', 'LOW'}:
                        risk = 'HIGH'
                    normalized.append({
                        'text': pii_text,
                        'entity': entity,
                        'risk': risk,
                        'confidence': confidence,
                    })
                except Exception:
                    continue

            logger.info(f"Gemini Vision identified {len(normalized)} PII items")
            return normalized

        except Exception as e:
            logger.error(f"Gemini Vision PII analysis failed: {e}")
            self.last_error = str(e)
            return []

