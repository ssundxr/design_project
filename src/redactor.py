"""
PII Redactor - Enterprise Edition v4.0
Powered by Gemini Vision API for intelligent PII detection
Only redacts DETECTED PII - Nothing else
"""
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw
import hashlib
import numpy as np
import re


try:
    from .ocr import extract_words_with_boxes
except ImportError:
    from ocr import extract_words_with_boxes

try:
    from .local_pii_detector import LocalPIIDetector
except ImportError:
    from local_pii_detector import LocalPIIDetector
try:
    from .layoutlm_detector import LayoutLMDetector, LAYOUTLM_AVAILABLE
except ImportError:
    try:
        from layoutlm_detector import LayoutLMDetector, LAYOUTLM_AVAILABLE
    except ImportError:
        LAYOUTLM_AVAILABLE = False

# The original code had `logger = logging.getLogger(__name__)` and `logging.basicConfig(...)`
# These should remain.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Try importing Gemini detector
# This block is removed as per instructions.


class PIIRedactor:
    """
    Smart PII Redactor for Government Documents.
    
    Redacts ONLY:
    - Names (detected through context)
    - Aadhaar numbers (12 digits)
    - Phone numbers
    - Addresses (multi-line text blocks)
    - Photos (face detection)
    - QR codes
    - Signatures
    
    Does NOT redact:
    - Headers/titles
    - Government logos
    - General text
    """
    
    def __init__(self, output_dir: str = "output"):
        """Initialize redactor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.redacted_dir = self.output_dir / "redacted"
        self.redacted_dir.mkdir(exist_ok=True)
        
        self.audit_dir = self.output_dir / "audit_logs"
        self.audit_dir.mkdir(exist_ok=True)
        
        # Local PII Detector (spaCy NER + context + regex)
        self.local_detector = LocalPIIDetector()
        
        # Initialize LayoutLM Detector (Local PII extraction)
        self.layoutlm = None
        self._init_layoutlm()
        
        # Initialize visual detectors
        self._init_detectors()

    def _init_layoutlm(self):
        """Initialize LayoutLMv3 for context-aware structured detection."""
        if not LAYOUTLM_AVAILABLE:
            logger.warning("LayoutLMv3 unavailable")
            return
            
        try:
            self.layoutlm = LayoutLMDetector()
            logger.info("✓ LayoutLMv3 ready (Local PII Extraction)")
        except Exception as e:
            logger.warning(f"LayoutLM init failed: {e}.")
    
    def _init_detectors(self):
        """Initialize visual detectors (lazy — YOLO loads on first use)."""
        self.visual_detector = None
        self._visual_detector_initialized = False
    
    def _get_visual_detector(self):
        """Lazy-load visual detector on first use."""
        if not self._visual_detector_initialized:
            self._visual_detector_initialized = True
            try:
                try:
                    from .detector import VisualPIIDetector
                except ImportError:
                    from detector import VisualPIIDetector
                self.visual_detector = VisualPIIDetector()
                logger.info("✓ Visual detector ready (lazy loaded)")
            except Exception as e:
                self.visual_detector = None
                logger.warning(f"Visual detector unavailable: {e}")
        return self.visual_detector
    
    def redact_image(
        self,
        image: Image.Image,
        filename: str = "document.jpg",
        generate_audit: bool = True,
        words_with_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Intelligently redact ONLY PII from document.
        """
        start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING REDACTION: {filename}")
        logger.info(f"{'='*60}")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            logger.info("Converting numpy array to PIL Image")
            image = Image.fromarray(image)
        
        # Audit data
        audit_data = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'image_size': image.size,
            'detections': [],
            'statistics': {},
            'nlp_explanation': '',
            'risk_assessment': '',
            'processing_time': 0
        }
        
        # Create redacted copy
        redacted = image.copy()
        draw = ImageDraw.Draw(redacted)
        
        # Collect all PII detections
        all_detections = []

        # 1. Extract ALL text first
        logger.info(f"\n[1/3] Extracting text...")
        words_with_boxes = words_with_boxes or extract_words_with_boxes(image)
        words_with_boxes = self._normalize_words_with_boxes(words_with_boxes)
        logger.info(f"✓ Extracted {len(words_with_boxes)} words")

        # 2. Text PII Detection (fully local)
        logger.info("\n[2/3] Detecting Text PII (Local: spaCy + Context + Regex)...")
        text_detections = []

        # PRIMARY: Local PII detector (spaCy NER + context heuristics + regex)
        local_detections = self.local_detector.detect_pii(words_with_boxes)
        text_detections.extend(local_detections)
        logger.info(f"✓ Local detector found {len(local_detections)} PII items")

        # SECONDARY: LayoutLMv3 structured detection
        if self.layoutlm:
            layout_detections = self.layoutlm.detect_structured_pii(image)
            text_detections.extend(layout_detections)
            logger.info(f"✓ LayoutLM found {len(layout_detections)} context PII items")

        all_detections.extend(self._dedupe_detections(text_detections))

        # 3. Detect visual PII (photos, QR codes, signatures)
        logger.info("\n[3/3] Detecting visual PII (YOLOv8 + Signature + QR)...")
        visual_detections = self._detect_visual_pii_precise(image)
        all_detections.extend(visual_detections)
        logger.info(f"✓ Found {len(visual_detections)} visual PII items")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL PII ITEMS TO REDACT: {len(all_detections)}")
        logger.info(f"{'='*60}\n")
        
        # Redact each detected PII
        for i, detection in enumerate(all_detections, 1):
            self._redact_single_item(draw, detection, i)
        
        # Finalize
        processing_time = time.time() - start_time
        audit_data['detections'] = all_detections
        audit_data['statistics'] = self._calculate_statistics(all_detections)
        audit_data['processing_time'] = round(processing_time, 2)
        
        if generate_audit:
            audit_data['nlp_explanation'] = self._generate_explanation(all_detections)
            audit_data['risk_assessment'] = self._assess_risk(all_detections)
            self._save_audit(audit_data)
        
        logger.info(f"\n✓ REDACTION COMPLETE in {processing_time:.2f}s\n")
        
        return redacted, audit_data
    
    def _fallback_regex_detection(self, words_with_boxes: List[Tuple[str, Tuple]]) -> List[Dict]:
        """
        Fallback: Detect PII using regex patterns (used when Gemini is unavailable).
        Only uses high-precision patterns to minimize false positives.
        """
        detections = []
        
        # Use only high-precision patterns (skip ACCOUNT, DATE, PASSPORT)
        import re
        safe_patterns = {
            "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            "PHONE": re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d"),
            "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
            "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
            "CREDIT_CARD": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
        }
        
        for word, bbox in words_with_boxes:
            for label, pattern in safe_patterns.items():
                if pattern.search(word):
                    x, y, w, h = bbox
                    detections.append({
                        'type': 'PATTERN',
                        'entity': label,
                        'text': word,
                        'bbox': [int(x), int(y), int(x + w), int(y + h)],
                        'confidence': 0.90,
                        'risk': self._map_risk(label)
                    })
                    logger.info(f"  → {label} (regex): {word}")

        # Contextual fallback: detect person name and address near label tokens.
        cleaned = []
        for idx, (word, bbox) in enumerate(words_with_boxes):
            token = re.sub(r"[^A-Za-z0-9]", "", str(word or ""))
            cleaned.append((idx, token, word, bbox))

        field_labels = {
            "name", "fullname", "candidate", "applicant",
            "address", "residence", "location",
            "phone", "mobile", "contact", "tel", "telephone",
            "email", "mail", "emailid", "emailaddress",
        }
        field_prefixes = ("emai", "mail", "phon", "mobi", "cont", "addr", "tele", "tel")

        def is_name_token(tok: str) -> bool:
            return bool(re.fullmatch(r"[A-Z][a-z]{1,20}", tok))

        phone_re = safe_patterns["PHONE"]
        email_re = safe_patterns["EMAIL"]

        for i, token, raw_word, bbox in cleaned:
            key = token.lower()

            if key in {"name", "fullname", "candidate", "applicant"}:
                name_parts = []
                box_parts = []
                for j in range(i + 1, min(i + 6, len(cleaned))):
                    _, ntok, nraw, nbbox = cleaned[j]
                    ltok = ntok.lower()
                    if ltok in field_labels or ltok.startswith(field_prefixes):
                        break
                    if is_name_token(ntok):
                        name_parts.append(nraw)
                        box_parts.append(nbbox)
                    elif name_parts:
                        break

                if len(name_parts) >= 2 and box_parts:
                    x1 = min(b[0] for b in box_parts)
                    y1 = min(b[1] for b in box_parts)
                    x2 = max(b[0] + b[2] for b in box_parts)
                    y2 = max(b[1] + b[3] for b in box_parts)
                    detections.append({
                        'type': 'PATTERN',
                        'entity': 'PERSON_NAME',
                        'text': ' '.join(name_parts),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': 0.80,
                        'risk': 'HIGH'
                    })
                    logger.info(f"  → PERSON_NAME (context): {' '.join(name_parts)}")

            if key in {"address", "residence", "location"}:
                box_parts = []
                text_parts = []
                for j in range(i + 1, min(i + 14, len(cleaned))):
                    _, ntok, nraw, nbbox = cleaned[j]
                    if not ntok:
                        continue
                    text_parts.append(nraw)
                    box_parts.append(nbbox)
                    if any(ch in str(nraw) for ch in ['|', ';']):
                        break

                if len(text_parts) >= 4 and box_parts:
                    x1 = min(b[0] for b in box_parts)
                    y1 = min(b[1] for b in box_parts)
                    x2 = max(b[0] + b[2] for b in box_parts)
                    y2 = max(b[1] + b[3] for b in box_parts)
                    detections.append({
                        'type': 'PATTERN',
                        'entity': 'ADDRESS',
                        'text': ' '.join(text_parts[:12]),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': 0.70,
                        'risk': 'HIGH'
                    })
                    logger.info("  → ADDRESS (context)")

            if key in {"phone", "mobile", "contact", "tel", "telephone"}:
                cand_parts = []
                box_parts = []
                for j in range(i + 1, min(i + 6, len(cleaned))):
                    _, ntok, nraw, nbbox = cleaned[j]
                    if not ntok:
                        continue
                    cand_parts.append(str(nraw))
                    box_parts.append(nbbox)
                    candidate = " ".join(cand_parts)
                    if phone_re.search(candidate):
                        x1 = min(b[0] for b in box_parts)
                        y1 = min(b[1] for b in box_parts)
                        x2 = max(b[0] + b[2] for b in box_parts)
                        y2 = max(b[1] + b[3] for b in box_parts)
                        detections.append({
                            'type': 'PATTERN',
                            'entity': 'PHONE',
                            'text': candidate,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': 0.84,
                            'risk': 'MEDIUM'
                        })
                        logger.info("  → PHONE (context)")
                        break

            if key in {"email", "mail", "emailid", "e-mail"}:
                cand_parts = []
                box_parts = []
                for j in range(i + 1, min(i + 6, len(cleaned))):
                    _, ntok, nraw, nbbox = cleaned[j]
                    if not ntok:
                        continue
                    cand_parts.append(str(nraw))
                    box_parts.append(nbbox)
                    candidate = "".join(cand_parts)
                    if email_re.search(candidate):
                        x1 = min(b[0] for b in box_parts)
                        y1 = min(b[1] for b in box_parts)
                        x2 = max(b[0] + b[2] for b in box_parts)
                        y2 = max(b[1] + b[3] for b in box_parts)
                        detections.append({
                            'type': 'PATTERN',
                            'entity': 'EMAIL',
                            'text': candidate,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': 0.88,
                            'risk': 'MEDIUM'
                        })
                        logger.info("  → EMAIL (context)")
                        break

        return self._dedupe_detections(detections)

    def _dedupe_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep first occurrence per entity+bbox to avoid duplicate redactions."""
        ordered = sorted(detections, key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for d in ordered:
            bbox = d.get('bbox', [])
            if len(bbox) != 4:
                continue
            entity = str(d.get('entity', 'PII'))
            raw_text = str(d.get('text', ''))
            norm_text = re.sub(r"[^a-z0-9]", "", raw_text.lower())
            if entity == 'PHONE':
                digits = re.sub(r"\D", "", raw_text)
                if len(digits) >= 7:
                    key = (entity, digits[-10:])
                else:
                    key = (entity, tuple(int(v) for v in bbox))
            elif entity == 'EMAIL':
                key = (entity, norm_text)
            elif len(norm_text) >= 6:
                key = (entity, norm_text)
            else:
                key = (entity, tuple(int(v) for v in bbox))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(d)
        return deduped

    def _normalize_words_with_boxes(self, words_with_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Split OCR-merged spans into token-level boxes to improve PII precision."""
        normalized: List[Tuple[str, Tuple[int, int, int, int]]] = []
        token_re = re.compile(r"[A-Za-z0-9@._+\-:/]+")

        for text, bbox in words_with_boxes:
            text = str(text or "").strip()
            if not text:
                continue

            x, y, w, h = bbox
            if ' ' not in text and len(text) <= 32:
                normalized.append((text, (int(x), int(y), int(w), int(h))))
                continue

            matches = list(token_re.finditer(text))
            if not matches:
                normalized.append((text, (int(x), int(y), int(w), int(h))))
                continue

            span_len = max(1, len(text))
            for m in matches:
                token = m.group(0).strip()
                if not token:
                    continue
                start_ratio = m.start() / span_len
                end_ratio = m.end() / span_len
                tx1 = int(x + w * start_ratio)
                tx2 = int(x + w * end_ratio)
                tw = max(1, tx2 - tx1)
                normalized.append((token, (tx1, int(y), tw, int(h))))

        return normalized

    def _detect_visual_pii_precise(self, image: Image.Image) -> List[Dict]:
        """
        Detect ONLY faces, QR codes, and signatures.
        NO generic regions.
        """
        detector = self._get_visual_detector()
        if not detector:
            return []
        
        detections = []
        
        try:
            # Import specific detectors
            try:
                from .detector import SignatureDetector, QRCodeDetector, StampDetector, FingerprintDetector
            except ImportError:
                from detector import SignatureDetector, QRCodeDetector, StampDetector, FingerprintDetector
            
            img_w, img_h = image.size
            page_area = max(1, img_w * img_h)

            # 1. Signatures
            sig_detector = SignatureDetector()
            signatures = sig_detector.detect(image)
            sig_count = 0
            for sig in signatures:
                if sig['confidence'] >= 0.92:  # Very high confidence only
                    x, y, w, h = sig['bbox']
                    area_ratio = (w * h) / page_area
                    if area_ratio < 0.0002 or area_ratio > 0.08:
                        continue
                    sig_count += 1
                    if sig_count > 10:
                        break
                    detections.append({
                        'type': 'VISUAL',
                        'entity': 'SIGNATURE',
                        'text': '',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': sig['confidence'],
                        'risk': 'HIGH'
                    })
                    logger.info(f"  → SIGNATURE at ({x}, {y})")

            # If signatures explode, it's likely a false-positive page-wide texture pattern.
            if sig_count > 8:
                detections = [d for d in detections if d.get('entity') != 'SIGNATURE']
                logger.warning("Signature detections exceeded safety limit; dropped as likely false positives")
            
            # 2. QR Codes
            qr_detector = QRCodeDetector()
            qr_codes = qr_detector.detect(image)
            for qr in qr_codes:
                if qr['confidence'] > 0.65:
                    x, y, w, h = qr['bbox']
                    detections.append({
                        'type': 'VISUAL',
                        'entity': 'QR_CODE',
                        'text': '',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': qr['confidence'],
                        'risk': 'HIGH'
                    })
                    logger.info(f"  → QR CODE at ({x}, {y})")

            # 3. Stamps
            stamp_detector = StampDetector()
            stamps = stamp_detector.detect(image)
            stamp_count = 0
            for st in stamps:
                if st['confidence'] >= 0.60:
                    x, y, w, h = st['bbox']
                    area_ratio = (w * h) / page_area
                    if area_ratio < 0.0004 or area_ratio > 0.25:
                        continue
                    stamp_count += 1
                    if stamp_count > 5:
                        break
                    detections.append({
                        'type': 'VISUAL',
                        'entity': 'STAMP',
                        'text': '',
                        'bbox': [x, y, x + w, y + h],
                        'confidence': st['confidence'],
                        'risk': 'HIGH'
                    })
                    logger.info(f"  → STAMP at ({x}, {y})")

            # 4. Fingerprints
            fp_detector = FingerprintDetector()
            prints = fp_detector.detect(image)
            fp_count = 0
            for fp in prints:
                if fp['confidence'] >= 0.60:
                    x, y, w, h = fp['bbox']
                    area_ratio = (w * h) / page_area
                    if area_ratio < 0.0002 or area_ratio > 0.06:
                        continue
                    fp_count += 1
                    if fp_count > 6:
                        break
                    detections.append({
                        'type': 'VISUAL',
                        'entity': 'FINGERPRINT',
                        'text': '',
                        'bbox': [x, y, x + w, y + h],
                        'confidence': fp['confidence'],
                        'risk': 'HIGH'
                    })
                    logger.info(f"  → FINGERPRINT at ({x}, {y})")
            
            # 5. Faces/Photos (YOLO if available)
            if detector.model:
                import numpy as np
                img_array = np.array(image)
                results = detector.model(img_array, conf=0.6, verbose=False, classes=[0])
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = box
                            detections.append({
                                'type': 'VISUAL',
                                'entity': 'PHOTO/FACE',
                                'text': '',
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': 0.85,
                                'risk': 'HIGH'
                            })
                            logger.info(f"  → PHOTO/FACE at ({int(x1)}, {int(y1)})")
        
        except Exception as e:
            logger.error(f"Visual detection error: {e}")

        return self._suppress_overlaps(detections)

    def _suppress_overlaps(self, detections: List[Dict], iou_threshold: float = 0.6) -> List[Dict]:
        """Simple NMS-style suppression to reduce duplicate overlapping visual boxes."""
        if not detections:
            return detections

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                return 0.0
            a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / float(a_area + b_area - inter)

        ordered = sorted(detections, key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
        kept: List[Dict] = []
        for d in ordered:
            box = d.get('bbox', [0, 0, 0, 0])
            if not kept:
                kept.append(d)
                continue
            if all(iou(box, k.get('bbox', [0, 0, 0, 0])) <= iou_threshold for k in kept):
                kept.append(d)

        return kept
    
    def _redact_single_item(self, draw: ImageDraw.Draw, detection: Dict, index: int):
        """Redact a single PII item."""
        try:
            bbox = detection['bbox']
            risk = detection.get('risk', 'MEDIUM')
            entity = detection.get('entity', 'PII')
            
            # Parse bbox
            if len(bbox) == 4:
                if all(isinstance(x, (int, float)) for x in bbox):
                    # Could be [x, y, w, h] or [x, y, x2, y2]
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1] and bbox[2] < 5000:
                        # Likely [x, y, x2, y2]
                        x1, y1, x2, y2 = map(int, bbox)
                    else:
                        # Likely [x, y, w, h]
                        x, y, w, h = map(int, bbox)
                        x1, y1, x2, y2 = x, y, x+w, y+h
                else:
                    return
            else:
                return
            
            # Validate coordinates
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 5 or (y2-y1) < 5:
                logger.debug(f"Skipped invalid box: {bbox}")
                return
            
            # Color by risk
            colors = {
                'HIGH': (0, 0, 0),
                'MEDIUM': (40, 40, 40),
                'LOW': (80, 80, 80)
            }
            color = colors.get(risk, (0, 0, 0))
            
            # Draw redaction
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
            
            logger.debug(f"[{index}] Redacted {entity}: ({x1},{y1}) to ({x2},{y2})")
        
        except Exception as e:
            logger.error(f"Redaction failed for item {index}: {e}")
    
    def _map_risk(self, entity: str) -> str:
        """Map entity to risk level."""
        high_risk = {'AADHAAR', 'SSN', 'PASSPORT', 'PAN', 'CREDIT_CARD', 'PERSON_NAME'}
        medium_risk = {'EMAIL', 'PHONE', 'ACCOUNT', 'IP_ADDRESS'}
        
        return 'HIGH' if entity in high_risk else ('MEDIUM' if entity in medium_risk else 'LOW')
    
    def _calculate_statistics(self, detections: List[Dict]) -> Dict:
        """Calculate statistics."""
        stats = {
            'total_detections': len(detections),
            'by_type': {},
            'by_risk': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'entities': {}
        }
        
        for det in detections:
            # By type
            det_type = det.get('type', 'UNKNOWN')
            stats['by_type'][det_type] = stats['by_type'].get(det_type, 0) + 1
            
            # By risk
            risk = det.get('risk', 'LOW')
            if risk in stats['by_risk']:
                stats['by_risk'][risk] += 1
            
            # By entity
            entity = det.get('entity', 'UNKNOWN')
            stats['entities'][entity] = stats['entities'].get(entity, 0) + 1
        
        return stats
    
    def _generate_explanation(self, detections: List[Dict]) -> str:
        """
        Generate DPDP compliant XAI audit explanation (fully local).
        """
        if not detections:
            return "No sensitive information detected in this document."

        counts = {}
        for d in detections:
            counts[d['entity']] = counts.get(d['entity'], 0) + 1

        summary = ", ".join([f"{k} ({v})" for k, v in counts.items()])
        return f"System redacted {len(detections)} sensitive items: {summary}. Processed fully on-device conforming to DPDP Act 2023."
    
    def _assess_risk(self, detections: List[Dict]) -> str:
        """Assess overall risk."""
        if not detections:
            return "MINIMAL - No PII detected"
        
        stats = self._calculate_statistics(detections)
        high = stats['by_risk']['HIGH']
        
        if high >= 3:
            return "CRITICAL - Multiple high-risk PII elements"
        elif high >= 1:
            return "HIGH - High-risk PII present"
        elif len(detections) >= 5:
            return "MODERATE - Multiple PII items"
        else:
            return "LOW - Minimal PII"
    
    def _save_audit(self, audit_data: Dict):
        """Save audit log."""
        try:
            audit_id = hashlib.sha256(
                f"{audit_data['filename']}{audit_data['timestamp']}".encode()
            ).hexdigest()[:12]
            
            audit_file = self.audit_dir / f"audit_{audit_id}.json"
            
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
            audit_data['audit_file'] = audit_file.name
            audit_data['audit_path'] = str(audit_file)
            
            logger.info(f"✓ Audit saved: {audit_file.name}")
        except Exception as e:
            logger.error(f"Audit save failed: {e}")

    def save_redacted_image(self, image: Image.Image, filename: str) -> Path:
        """Persist a redacted image to the output directory."""
        output_path = self.redacted_dir / filename
        image.save(output_path)
        return output_path



def get_enhanced_redactor() -> PIIRedactor:
    """Factory function."""
    return PIIRedactor()
