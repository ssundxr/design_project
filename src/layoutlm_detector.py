"""
Privara LayoutLM Detector - Document Structure Understanding
Detects PII in structured forms using Microsoft LayoutLMv3
Version: 2.0 Enterprise
"""

from typing import List, Dict, Any, Tuple
import logging
from PIL import Image
import numpy as np

try:
    import pytesseract
    TESSERACT_BINDING_AVAILABLE = True
except ImportError:
    TESSERACT_BINDING_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Try importing LayoutLM
try:
    from transformers import LayoutLMv3ImageProcessor, LayoutLMv3ForTokenClassification
    from datasets import Features, Sequence, ClassLabel, Value, Array2D
    LAYOUTLM_AVAILABLE = TORCH_AVAILABLE and TESSERACT_BINDING_AVAILABLE
except ImportError:
    LAYOUTLM_AVAILABLE = False
    logger.warning("LayoutLM not available. Install: pip install transformers datasets")


class LayoutLMDetector:
    """
    Structured document PII detection using LayoutLMv3.
    
    Best for:
    - Banking forms
    - Insurance applications
    - Government documents
    - Tax forms
    - Employment records
    """
    
    # PII entity labels for LayoutLM
    PII_ENTITIES = {
        "PERSON", "ORG", "LOC", "DATE", "TIME",
        "MONEY", "PERCENT", "ID_NUMBER", "ADDRESS",
        "EMAIL", "PHONE", "SSN", "ACCOUNT", "LICENSE"
    }
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        """
        Initialize LayoutLM detector.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "cpu"
        
        if not LAYOUTLM_AVAILABLE:
            logger.error("LayoutLM dependencies not available")
            return
        
        try:
            logger.info(f"Loading LayoutLMv3: {model_name}")
            self.processor = LayoutLMv3ImageProcessor.from_pretrained(model_name)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"✓ LayoutLMv3 loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load LayoutLM: {e}")
    
    def detect_structured_pii(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Detect PII in structured documents.
        
        Args:
            image: PIL Image of form/document
            confidence_threshold: Minimum confidence (0-1)
            
        Returns:
            List of detection dicts with entity, text, bbox, confidence
        """
        if not self.model or not self.processor:
            logger.warning("LayoutLM not initialized, using fallback")
            return self._fallback_detection(image)
        
        try:
            # Extract words and bounding boxes with Tesseract
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )
            
            words, norm_boxes, pixel_boxes = self._parse_tesseract_output(data)
            
            if not words:
                return []
            
            # Prepare input for LayoutLM
            encoding = self.processor(
                image,
                text=words,
                boxes=norm_boxes,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            
            # Move to device
            for key in encoding:
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
                confidences = torch.softmax(outputs.logits, dim=-1).max(-1).values.squeeze().tolist()
            
            # Parse predictions
            detections = []
            for i, (word, pbox, pred, conf) in enumerate(zip(words, pixel_boxes, predictions, confidences)):
                # Skip padding and low confidence
                if word == "[PAD]" or conf < confidence_threshold:
                    continue
                
                # Map prediction to entity label (simplified)
                entity_label = self._map_prediction_to_entity(pred)
                
                if entity_label in self.PII_ENTITIES:
                    x_min, y_min, x_max, y_max = pbox
                    detections.append({
                        'entity': entity_label,
                        'text': word,
                        'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                        'confidence': float(conf),
                        'risk': self._assess_risk(entity_label)
                    })
            
            logger.info(f"Detected {len(detections)} structured PII entities")
            return detections
            
        except Exception as e:
            logger.error(f"Structured detection failed: {e}")
            return []
    
    def _parse_tesseract_output(self, data: dict) -> Tuple[List[str], List[List[int]], List[List[int]]]:
        """Parse Tesseract output into words with normalized and pixel boxes."""
        words = []
        norm_boxes = []
        pixel_boxes = []
        
        if not data.get('text'):
            return words, norm_boxes, pixel_boxes

        img_width = max(1, max(data['left']) + max(data['width']))
        img_height = max(1, max(data['top']) + max(data['height']))
        
        n = len(data['text'])
        for i in range(n):
            text = data['text'][i]
            if text.strip():
                # Normalize coordinates to 0-1000 range (LayoutLM standard)
                x = int((data['left'][i] / img_width) * 1000)
                y = int((data['top'][i] / img_height) * 1000)
                x_max = int(((data['left'][i] + data['width'][i]) / img_width) * 1000)
                y_max = int(((data['top'][i] + data['height'][i]) / img_height) * 1000)
                px1 = int(data['left'][i])
                py1 = int(data['top'][i])
                px2 = int(data['left'][i] + data['width'][i])
                py2 = int(data['top'][i] + data['height'][i])
                
                words.append(text)
                norm_boxes.append([x, y, x_max, y_max])
                pixel_boxes.append([px1, py1, px2, py2])
        
            return words, norm_boxes, pixel_boxes
    
    def _map_prediction_to_entity(self, pred: int) -> str:
        """Map model prediction ID to entity label."""
        # Simplified mapping (actual model has specific label mapping)
        entity_map = {
            0: "O",  # Outside
            1: "PERSON",
            2: "ORG",
            3: "ID_NUMBER",
            4: "ADDRESS",
            5: "PHONE",
            6: "EMAIL"
        }
        return entity_map.get(pred, "O")
    
    def _assess_risk(self, entity: str) -> str:
        """Assess PII risk level."""
        high_risk = {"PERSON", "SSN", "ID_NUMBER", "ACCOUNT"}
        medium_risk = {"ADDRESS", "PHONE", "EMAIL", "DATE"}
        
        if entity in high_risk:
            return "HIGH"
        elif entity in medium_risk:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _fallback_detection(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Fallback to pattern-based detection if LayoutLM unavailable."""
        try:
            from .detector import detect_pii_patterns
        except ImportError:
            from detector import detect_pii_patterns
        
        try:
            text = pytesseract.image_to_string(image)
            patterns = detect_pii_patterns(text)
            
            detections = []
            for label, (start, end) in patterns:
                detections.append({
                    'entity': label,
                    'text': text[start:end],
                    'bbox': [0, 0, 100, 20],  # Placeholder
                    'confidence': 0.8,
                    'risk': 'MEDIUM'
                })
            
            return detections
        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            return []
