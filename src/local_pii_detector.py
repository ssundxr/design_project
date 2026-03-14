"""
Local PII Detector - Fully On-Device PII Detection Engine
Combines spaCy NER, contextual heuristics, and regex patterns.
No cloud APIs required. DPDP Act 2023 compliant.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# --- spaCy NER ---
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("✓ spaCy NER loaded (en_core_web_sm)")
except Exception:
    SPACY_AVAILABLE = False
    _nlp = None
    logger.warning("spaCy not available. Install: pip install spacy && python -m spacy download en_core_web_sm")


# ============================================
# HIGH-PRECISION REGEX PATTERNS
# ============================================

REGEX_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "AADHAAR_MASKED": re.compile(r"\*{4,}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "PASSPORT_NUM": re.compile(r"\b[A-Z]{1,2}\d{6,8}\b"),
    "DATE_OF_BIRTH": re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"),
    "PINCODE": re.compile(r"\b[1-9]\d{5}\b"),
}

# Risk mapping per entity type
ENTITY_RISK = {
    "PERSON_NAME": "HIGH",
    "FATHER_NAME": "HIGH",
    "MOTHER_NAME": "HIGH",
    "SPOUSE_NAME": "HIGH",
    "DATE_OF_BIRTH": "HIGH",
    "PLACE_OF_BIRTH": "HIGH",
    "AADHAAR": "HIGH",
    "AADHAAR_MASKED": "HIGH",
    "PAN": "HIGH",
    "PASSPORT_NUM": "HIGH",
    "SSN": "HIGH",
    "CREDIT_CARD": "HIGH",
    "EMAIL": "MEDIUM",
    "PHONE": "MEDIUM",
    "ADDRESS": "HIGH",
    "PINCODE": "MEDIUM",
}


# ============================================
# CONTEXTUAL FIELD LABELS (Indian Documents)
# ============================================

# Maps lowercase field label → entity type for the VALUE that follows it
_NAME_LABELS = {
    "name", "fullname", "full name", "candidate", "applicant",
    "given name", "surname", "first name", "last name",
    "applicant name", "holder name", "card holder",
}
_FATHER_LABELS = {
    "father", "father's name", "fathers name", "father name",
    "father's given name", "fathers given name",
    "s/o", "son of", "d/o", "daughter of",
}
_MOTHER_LABELS = {
    "mother", "mother's name", "mothers name", "mother name",
    "mother's given name", "mothers given name",
}
_SPOUSE_LABELS = {
    "spouse", "spouse name", "spouse's name", "husband", "wife",
    "husband's name", "wife's name", "w/o",
}
_DOB_LABELS = {
    "dob", "date of birth", "birth date", "birthdate",
    "born on", "born",
}
_POB_LABELS = {
    "place of birth", "birth place", "birthplace",
}
_ADDRESS_LABELS = {
    "address", "permanent address", "present address",
    "residential address", "residence", "communication address",
    "correspondence address", "current address",
}
_PHONE_LABELS = {
    "phone", "mobile", "contact", "tel", "telephone",
    "mobile number", "phone number", "contact number",
    "mobile number / telephone number",
}
_EMAIL_LABELS = {
    "email", "e-mail", "email id", "emailid", "email address",
    "e-mail id", "mail", "mail id",
}

# All label sets with their entity type
_LABEL_MAP: List[Tuple[set, str]] = [
    (_NAME_LABELS, "PERSON_NAME"),
    (_FATHER_LABELS, "FATHER_NAME"),
    (_MOTHER_LABELS, "MOTHER_NAME"),
    (_SPOUSE_LABELS, "SPOUSE_NAME"),
    (_DOB_LABELS, "DATE_OF_BIRTH"),
    (_POB_LABELS, "PLACE_OF_BIRTH"),
    (_ADDRESS_LABELS, "ADDRESS"),
    (_PHONE_LABELS, "PHONE"),
    (_EMAIL_LABELS, "EMAIL"),
]


class LocalPIIDetector:
    """
    Fully local PII detection engine.
    Uses spaCy NER + contextual heuristics + regex patterns.
    """

    def __init__(self):
        self.nlp = _nlp
        logger.info(f"LocalPIIDetector ready (spaCy={'yes' if SPACY_AVAILABLE else 'no'})")

    def detect_pii(
        self,
        words_with_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
    ) -> List[Dict[str, Any]]:
        """
        Main entry: detect all PII from OCR words with bounding boxes.

        Returns list of detection dicts:
            [{'type': 'LOCAL', 'entity': str, 'text': str,
              'bbox': [x1,y1,x2,y2], 'confidence': float, 'risk': str}, ...]
        """
        detections: List[Dict[str, Any]] = []

        # 1. Regex pattern detection (high precision)
        detections.extend(self._regex_detection(words_with_boxes))

        # 2. Contextual field-label detection (document structure)
        detections.extend(self._context_detection(words_with_boxes))

        # 3. spaCy NER detection (names, locations, dates)
        if self.nlp:
            detections.extend(self._spacy_detection(words_with_boxes))

        logger.info(
            f"LocalPIIDetector found {len(detections)} PII items "
            f"(regex + context + spaCy)"
        )
        return detections

    # --------------------------------------------------
    # 1. REGEX DETECTION
    # --------------------------------------------------
    def _regex_detection(
        self, words_with_boxes: List[Tuple[str, Tuple[int, int, int, int]]]
    ) -> List[Dict[str, Any]]:
        """Match regex patterns against individual tokens and sliding windows."""
        detections: List[Dict[str, Any]] = []

        for word, bbox in words_with_boxes:
            x, y, w, h = bbox
            for label, pattern in REGEX_PATTERNS.items():
                if pattern.search(word):
                    detections.append(self._det(
                        label, word, x, y, w, h, 0.92,
                    ))
                    logger.debug(f"  → {label} (regex): {word}")

        # Multi-word sliding window for Aadhaar (split across tokens)
        full_text_parts = []
        for word, bbox in words_with_boxes:
            full_text_parts.append((word, bbox))

        # Sliding window of 2-4 tokens for multi-token patterns
        for window_size in (2, 3, 4):
            for i in range(len(full_text_parts) - window_size + 1):
                window = full_text_parts[i:i + window_size]
                combined = " ".join(w for w, _ in window)
                for label in ("AADHAAR", "PHONE", "CREDIT_CARD"):
                    pattern = REGEX_PATTERNS[label]
                    if pattern.search(combined):
                        boxes = [b for _, b in window]
                        x1 = min(b[0] for b in boxes)
                        y1 = min(b[1] for b in boxes)
                        x2 = max(b[0] + b[2] for b in boxes)
                        y2 = max(b[1] + b[3] for b in boxes)
                        detections.append({
                            'type': 'LOCAL_REGEX',
                            'entity': label,
                            'text': combined,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': 0.90,
                            'risk': ENTITY_RISK.get(label, 'MEDIUM'),
                        })

        return detections

    # --------------------------------------------------
    # 2. CONTEXTUAL FIELD-LABEL DETECTION
    # --------------------------------------------------
    def _context_detection(
        self, words_with_boxes: List[Tuple[str, Tuple[int, int, int, int]]]
    ) -> List[Dict[str, Any]]:
        """
        Detect PII by finding field labels and redacting the values next to them.
        This is the most powerful method for structured government forms.
        """
        detections: List[Dict[str, Any]] = []

        # Build token list with cleaned text
        tokens = []
        for idx, (word, bbox) in enumerate(words_with_boxes):
            tokens.append({
                'idx': idx,
                'raw': word,
                'clean': re.sub(r'[^A-Za-z0-9/\']', ' ', str(word or '')).strip().lower(),
                'bbox': bbox,
            })

        # For each token position, check if it starts a known field label
        consumed = set()
        for i, tok in enumerate(tokens):
            if i in consumed:
                continue

            # Try matching multi-word labels first (up to 5 tokens)
            matched_entity = None
            label_end = i  # last index of the label

            for label_len in range(5, 0, -1):
                if i + label_len > len(tokens):
                    continue
                candidate = " ".join(
                    tokens[j]['clean'] for j in range(i, i + label_len)
                ).strip()
                # Remove trailing colons, question marks
                candidate = re.sub(r'[:\?\.]$', '', candidate).strip()

                for label_set, entity in _LABEL_MAP:
                    if candidate in label_set:
                        matched_entity = entity
                        label_end = i + label_len - 1
                        break
                if matched_entity:
                    break

            if not matched_entity:
                continue

            # Found a label — now collect the VALUE tokens that follow
            value_start = label_end + 1
            value_tokens = []

            # How many value tokens to grab depends on entity type
            if matched_entity in ("ADDRESS",):
                max_value_tokens = 20
            elif matched_entity in ("PERSON_NAME", "FATHER_NAME", "MOTHER_NAME", "SPOUSE_NAME"):
                max_value_tokens = 6
            elif matched_entity in ("PLACE_OF_BIRTH",):
                max_value_tokens = 4
            else:
                max_value_tokens = 5

            for j in range(value_start, min(value_start + max_value_tokens, len(tokens))):
                if j in consumed:
                    break
                t = tokens[j]
                t_clean = t['clean']

                # Stop if we hit another field label
                is_label = False
                for label_set, _ in _LABEL_MAP:
                    if t_clean in label_set:
                        is_label = True
                        break
                    # Also check 2-word combos
                    if j + 1 < len(tokens):
                        two_word = t_clean + " " + tokens[j + 1]['clean']
                        two_word = re.sub(r'[:\?\.]$', '', two_word).strip()
                        if two_word in label_set:
                            is_label = True
                            break
                if is_label:
                    break

                # Stop if token is a known non-value (form label words)
                stop_words = {
                    "service", "required", "type", "applying", "booklet",
                    "have", "you", "ever", "changed", "your", "is",
                    "are", "the", "of", "for", "with", "from",
                    "government", "india", "ministry", "passport",
                    "application", "form", "details", "section",
                    "family", "police", "station", "permanent",
                }
                if matched_entity == "PERSON_NAME" and t_clean in stop_words:
                    break

                # For names: only accept capitalized words or ALL-CAPS
                if matched_entity in ("PERSON_NAME", "FATHER_NAME", "MOTHER_NAME", "SPOUSE_NAME"):
                    raw = t['raw']
                    if not raw or len(raw) < 2:
                        break
                    # Accept if: all caps, title case, or common name pattern
                    if not (raw[0].isupper() or raw.isupper()):
                        break
                    # Stop at clearly non-name tokens
                    if re.fullmatch(r'[^A-Za-z]+', raw):
                        break

                # For address: stop at semicolons/pipes
                if matched_entity == "ADDRESS":
                    if any(ch in str(t['raw']) for ch in '|;'):
                        break

                value_tokens.append(t)

            # Create detections for value tokens
            if value_tokens:
                for vt in value_tokens:
                    x, y, w, h = vt['bbox']
                    consumed.add(vt['idx'])
                    detections.append(self._det(
                        matched_entity, vt['raw'],
                        x, y, w, h, 0.88,
                    ))

                value_text = " ".join(vt['raw'] for vt in value_tokens)
                logger.info(f"  → {matched_entity} (context): {value_text}")

            # Mark label tokens as consumed
            for k in range(i, label_end + 1):
                consumed.add(k)

        return detections

    # --------------------------------------------------
    # 3. SPACY NER DETECTION
    # --------------------------------------------------
    def _spacy_detection(
        self, words_with_boxes: List[Tuple[str, Tuple[int, int, int, int]]]
    ) -> List[Dict[str, Any]]:
        """
        Run spaCy NER on the full OCR text to find PERSON, GPE, DATE entities,
        then map detected spans back to OCR bounding boxes.
        """
        if not self.nlp:
            return []

        detections: List[Dict[str, Any]] = []

        # Build full text and a char→token index
        full_text = ""
        char_to_token: Dict[int, int] = {}
        for idx, (word, _) in enumerate(words_with_boxes):
            start = len(full_text)
            full_text += word + " "
            for c in range(start, start + len(word)):
                char_to_token[c] = idx

        doc = self.nlp(full_text)

        # Map spaCy entity labels to our PII labels
        entity_map = {
            "PERSON": "PERSON_NAME",
            "GPE": "PLACE_OF_BIRTH",  # Geo-Political Entity (cities, etc.)
            "LOC": "ADDRESS",
            "DATE": "DATE_OF_BIRTH",
        }

        for ent in doc.ents:
            pii_label = entity_map.get(ent.label_)
            if not pii_label:
                continue

            # Skip very short or generic entities
            if len(ent.text.strip()) < 3:
                continue

            # Skip known non-PII names (government bodies)
            skip_texts = {
                "india", "government", "ministry", "external affairs",
                "passport", "republic", "union",
            }
            if ent.text.strip().lower() in skip_texts:
                continue

            # Find which OCR tokens this entity maps to
            token_indices = set()
            for c in range(ent.start_char, min(ent.end_char, len(full_text))):
                if c in char_to_token:
                    token_indices.add(char_to_token[c])

            if not token_indices:
                continue

            # Create a detection for each matched token
            for tidx in sorted(token_indices):
                word, bbox = words_with_boxes[tidx]
                x, y, w, h = bbox
                detections.append(self._det(
                    pii_label, word, x, y, w, h, 0.82,
                ))

            logger.debug(f"  → {pii_label} (spaCy): {ent.text}")

        return detections

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    @staticmethod
    def _det(entity: str, text: str, x: int, y: int, w: int, h: int,
             confidence: float) -> Dict[str, Any]:
        return {
            'type': 'LOCAL',
            'entity': entity,
            'text': text,
            'bbox': [int(x), int(y), int(x + w), int(y + h)],
            'confidence': confidence,
            'risk': ENTITY_RISK.get(entity, 'MEDIUM'),
        }
