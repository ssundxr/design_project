"""
Privara Multilingual OCR Module - India Localized
Combines PaddleOCR (Fast, 22+ Languages), Microsoft TrOCR (Handwriting), and Tesseract (Fallback).
Version: 3.0 Enterprise (DPDP Compliant)
"""

from typing import List, Tuple, Dict, Any
from PIL import Image, ImageEnhance
import logging
import numpy as np
import time
import os

logger = logging.getLogger(__name__)

# --- EASYOCR (Pure-Python fallback) ---
try:
    import easyocr
    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False

EASY_OCR_READER = None

# --- PADDLE OCR (Primary High-Speed Text Engine) ---
try:
    from paddleocr import PaddleOCR
    # Default to English and Hindi for maximum utility in Indian context
    # Use angle cls (use_angle_cls=True) for skewed scans
    PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang='en')
    PADDLE_AVAILABLE = True
    logger.info("✓ PaddleOCR loaded (Primary Engine)")
except ImportError:
    PADDLE_AVAILABLE = False
    PADDLE_OCR = None
    logger.warning("PaddleOCR not available. Run: pip install paddlepaddle paddleocr")

# --- TESSERACT (Fallback Engine) ---
try:
    import pytesseract
    import shutil
    import os
    
    tesseract_paths = [
        shutil.which("tesseract"),
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\sdshy\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
    ]
    TESSERACT_AVAILABLE = False
    for path in tesseract_paths:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            TESSERACT_AVAILABLE = True
            logger.info(f"✓ Tesseract loaded (Fallback Engine)")
            break
except ImportError:
    TESSERACT_AVAILABLE = False

# --- TrOCR (Specialized Handwriting Engine) ---
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


TROCR_EXTRACTOR = None


def extract_words_with_boxes(image: Image.Image, lang: str = 'en') -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Extracts words and bounding boxes using PaddleOCR primarily.
    Falls back to Tesseract if PaddleOCR fails or is unavailable.
    """
    prefer_trocr = os.environ.get("USE_TROCR", "false").strip().lower() in {"1", "true", "yes", "on"}

    if prefer_trocr and TROCR_AVAILABLE:
        words = _trocr_extract_words(image)
        if words:
            return words

    global PADDLE_AVAILABLE

    if PADDLE_AVAILABLE:
        try:
            return _paddle_extract_words(image)
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}. Falling back to Tesseract.")
            # Avoid repeatedly hitting known Paddle runtime incompatibilities.
            PADDLE_AVAILABLE = False
            
    tesseract_words: List[Tuple[str, Tuple[int, int, int, int]]] = []
    easy_words: List[Tuple[str, Tuple[int, int, int, int]]] = []

    if TESSERACT_AVAILABLE:
        tesseract_words = _tesseract_extract_words(image)

    if EASY_OCR_AVAILABLE:
        easy_words = _easyocr_extract_words(image)

    if tesseract_words and easy_words:
        return _merge_word_boxes(tesseract_words, easy_words)

    if tesseract_words:
        return tesseract_words

    if easy_words:
        return easy_words
    
    logger.error("No OCR engines available!")
    return []


def _merge_word_boxes(
    primary: List[Tuple[str, Tuple[int, int, int, int]]],
    secondary: List[Tuple[str, Tuple[int, int, int, int]]]
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Merge OCR outputs while avoiding near-duplicate entries."""
    merged: List[Tuple[str, Tuple[int, int, int, int]]] = list(primary)
    seen = set()

    def _sig(text: str, bbox: Tuple[int, int, int, int]) -> Tuple[str, int, int]:
        norm = ''.join(ch.lower() for ch in text if ch.isalnum())
        x, y, _, _ = bbox
        return norm, int(x / 8), int(y / 8)

    for text, bbox in primary:
        seen.add(_sig(text, bbox))

    for text, bbox in secondary:
        key = _sig(text, bbox)
        if key in seen:
            continue
        merged.append((text, bbox))
        seen.add(key)

    return merged


def _trocr_extract_words(image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Optional TrOCR extraction path controlled by USE_TROCR env var."""
    global TROCR_EXTRACTOR

    try:
        if TROCR_EXTRACTOR is None:
            TROCR_EXTRACTOR = TrOCRExtractor()

        if TROCR_EXTRACTOR is None:
            return []

        words = TROCR_EXTRACTOR.extract_text_lines(image)
        logger.info(f"TrOCR extracted {len(words)} elements")
        return words
    except Exception as e:
        logger.error(f"TrOCR extraction failed: {e}")
        return []


def _easyocr_extract_words(image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Fallback OCR using EasyOCR when Paddle/Tesseract are unavailable."""
    global EASY_OCR_READER

    try:
        if EASY_OCR_READER is None:
            EASY_OCR_READER = easyocr.Reader(['en'], gpu=False)

        results = EASY_OCR_READER.readtext(np.array(image.convert('RGB')), detail=1)
        words: List[Tuple[str, Tuple[int, int, int, int]]] = []

        for box, text, conf in results:
            if not text or len(text.strip()) <= 1:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            w = x_max - x_min
            h = y_max - y_min
            if w > 0 and h > 0 and conf >= 0.4:
                words.append((text.strip(), (x_min, y_min, w, h)))

        logger.info(f"EasyOCR extracted {len(words)} elements")
        return words
    except Exception as e:
        logger.error(f"EasyOCR fallback failed: {e}")
        return []


def extract_text(image: Image.Image) -> str:
    """
    Extract all raw text from image.
    """
    words_bboxes = extract_words_with_boxes(image)
    return " ".join([word for word, bbox in words_bboxes])


def _paddle_extract_words(image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Run PaddleOCR and convert results to standard Privara format.
    Paddle returns: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence)], ...]
    """
    img_array = np.array(image.convert('RGB'))
    
    # Run inference with compatibility across PaddleOCR versions.
    start = time.time()
    try:
        result = PADDLE_OCR.ocr(img_array, cls=True)
    except TypeError:
        try:
            result = PADDLE_OCR.ocr(img_array)
        except Exception:
            result = PADDLE_OCR.predict(img_array)
    
    words = []
    if not result:
        return words

    # Legacy format: [[[[x,y]...], ('text', conf)], ...]
    if isinstance(result, list) and result and isinstance(result[0], list):
        for line in result[0]:
            try:
                box = line[0]
                text, conf = line[1]
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                w = x_max - x_min
                h = y_max - y_min
                if len(str(text).strip()) > 1 and float(conf) > 0.6:
                    words.append((str(text).strip(), (x_min, y_min, w, h)))
            except Exception:
                continue

    # Newer format: list of dict with rec_texts/rec_scores/rec_polys
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        for page in result:
            rec_texts = page.get('rec_texts', []) or []
            rec_scores = page.get('rec_scores', []) or []
            rec_polys = page.get('rec_polys', []) or []
            n = min(len(rec_texts), len(rec_polys))
            for i in range(n):
                text = str(rec_texts[i]).strip()
                conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                poly = rec_polys[i]
                if not text or len(text) <= 1 or conf <= 0.6:
                    continue
                try:
                    arr = np.array(poly)
                    xs = arr[:, 0]
                    ys = arr[:, 1]
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    w = max(1, x_max - x_min)
                    h = max(1, y_max - y_min)
                    words.append((text, (x_min, y_min, w, h)))
                except Exception:
                    continue
            
    logger.info(f"PaddleOCR extracted {len(words)} elements in {time.time()-start:.2f}s")
    return words


def _tesseract_extract_words(image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Fallback using standard Tesseract OCR."""
    try:
        img = image.convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
        words = []
        n = len(data.get("text", []))
        
        for i in range(n):
            text = str(data["text"][i]).strip()
            if text and len(text) > 1:
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                if w > 0 and h > 0:
                    words.append((text, (x, y, w, h)))
                    
        return words
    except Exception as e:
        logger.error(f"Tesseract fallback failed: {e}")
        return []


class TrOCRExtractor:
    """
    Specialized OCR using Microsoft TrOCR.
    Only active for specific zones requested by LayoutLM where handwritten text is suspected.
    """
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.processor = None
        self.device = torch.device("cpu")
        
        if not TROCR_AVAILABLE:
            self.logger.warning("TrOCR unavailable")
            return
            
        try:
            self.logger.info(f"Lazy loading TrOCR model: {model_name}")
            self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("✓ TrOCR loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load TrOCR: {e}")
            self.model = None

    def extract_text_lines(self, image: Image.Image, beams: int = 1) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Extract text lines using TrOCR; returns approximate word boxes."""
        if not self.model:
            return []
            
        try:
            rgb = image.convert('RGB')
            pixel_values = self.processor(images=rgb, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values, num_beams=max(1, int(beams)), max_new_tokens=256)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if not text:
                return []

            tokens = [t for t in text.split() if t.strip()]
            if not tokens:
                return []

            w, h = rgb.size
            token_w = max(1, int(w / max(1, len(tokens))))
            y = int(h * 0.1)
            bh = max(18, int(h * 0.06))
            words: List[Tuple[str, Tuple[int, int, int, int]]] = []
            for idx, token in enumerate(tokens):
                x = int(idx * token_w)
                tw = token_w
                words.append((token, (x, y, tw, bh)))

            return words
        except Exception as e:
            self.logger.error(f"TrOCR extraction failed: {e}")
            return []
