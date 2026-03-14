"""
Privara Visual PII Detector - YOLO + QR + Signature Detection
Enterprise-grade signature, face, QR code detection
Version: 2.2 Enterprise - Signature Edition
"""

from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw
import logging
import numpy as np
import re

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# OpenCV for advanced detection
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False  # ← Signature/QR detection DISABLED


# ============================================
# TEXT-BASED PII PATTERN DETECTION
# ============================================

PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "PASSPORT": re.compile(r"\b[A-Z]{1,2}\d{6,8}\b"),
    "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "DATE": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    "ACCOUNT": re.compile(r"\b\d{9,18}\b"),
}


def detect_pii_patterns(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Detect PII patterns in text using regex."""
    findings = []
    for label, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((label, (match.start(), match.end())))
    return findings


# ============================================
# SIGNATURE DETECTOR
# ============================================

class SignatureDetector:
    """
    Advanced signature detection using multiple methods.
    
    Signature characteristics:
    - Curved, complex strokes (high entropy)
    - Elongated horizontal structure (width > height * 1.5)
    - Dark ink on light background
    - Isolated from text (edges detection)
    - Low text density (few recognizable characters)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_width = 30
        self.min_height = 20
        self.min_area = 600
    
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect signatures in image using multiple methods.
        
        Returns:
            List of signature detection dicts
        """
        if not OPENCV_AVAILABLE:
            return []
        
        detections = []
        
        try:
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Method 1: Stroke pattern detection (curved lines)
            stroke_sigs = self._detect_signature_strokes(img_array)
            detections.extend(stroke_sigs)
            
            # Method 2: Isolated region detection
            isolated_sigs = self._detect_isolated_regions(img_array)
            detections.extend(isolated_sigs)
            
            # Method 3: Handwriting pattern detection
            handwrite_sigs = self._detect_handwriting_patterns(img_array)
            detections.extend(handwrite_sigs)
            
            # Deduplicate
            unique = self._deduplicate_signatures(detections)
            
            self.logger.info(f"Detected {len(unique)} signatures")
            return unique
            
        except Exception as e:
            self.logger.error(f"Signature detection failed: {e}")
            return []
    
    def _detect_signature_strokes(self, gray_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect signature by analyzing curved strokes.
        Signatures have smooth, continuous curves unlike printed text.
        """
        try:
            detections = []
            height, width = gray_img.shape
            
            # Threshold for binary
            _, binary = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Dilate to connect strokes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(binary, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Signature size range
                if self.min_area < area < (width * height * 0.3):
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Signatures are wider than tall (landscape orientation)
                    if w > h * 1.2 and w > self.min_width and h > self.min_height:
                        
                        # Analyze stroke pattern (curvature)
                        curvature = self._calculate_curvature(contour)
                        
                        # Signatures have high curvature (smooth curves)
                        if curvature > 0.15:
                            detections.append({
                                'type': 'SIGNATURE',
                                'bbox': (x, y, w, h),
                                'confidence': 0.80,
                                'method': 'stroke_analysis',
                                'curvature': curvature
                            })
                            self.logger.debug(f"Signature strokes at ({x}, {y}): curvature={curvature:.2f}")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Stroke detection failed: {e}")
            return []
    
    def _detect_isolated_regions(self, gray_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect isolated dark regions (signatures are often isolated from text).
        """
        try:
            detections = []
            height, width = gray_img.shape
            
            # Binary threshold
            _, binary = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            for i in range(1, num_labels):  # Skip background (0)
                area = stats[i, cv2.CC_STAT_AREA]
                left = stats[i, cv2.CC_STAT_LEFT]
                top = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Signature characteristics
                if self.min_area < area < (width * height * 0.25):
                    if w > h * 1.0 and w > self.min_width and h > self.min_height:
                        
                        # Check isolation (not touching image edges heavily)
                        if left > width * 0.05 and top > height * 0.05:
                            
                            # Analyze ink distribution
                            ink_density = area / (w * h)
                            
                            # Signatures have moderate ink density
                            if 0.2 < ink_density < 0.7:
                                detections.append({
                                    'type': 'SIGNATURE',
                                    'bbox': (left, top, w, h),
                                    'confidence': 0.75,
                                    'method': 'isolated_region',
                                    'ink_density': ink_density
                                })
                                self.logger.debug(f"Isolated signature at ({left}, {top}): density={ink_density:.2f}")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Isolated region detection failed: {e}")
            return []
    
    def _detect_handwriting_patterns(self, gray_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect handwriting-like patterns (signatures are hand-written).
        Uses texture analysis and line connectivity.
        """
        try:
            detections = []
            height, width = gray_img.shape
            
            # Edge detection
            edges = cv2.Canny(gray_img, 50, 150)
            
            # Hough line detection (signatures have multiple connected lines)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=10, maxLineGap=5)
            
            if lines is not None:
                line_regions = {}
                
                # Group lines into regions
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Find region this line belongs to
                    region_y = (y1 + y2) // 2
                    region_key = region_y // 20  # Group by vertical position
                    
                    if region_key not in line_regions:
                        line_regions[region_key] = []
                    line_regions[region_key].append((x1, y1, x2, y2))
                
                # Analyze line clusters
                for region_key, lines_in_region in line_regions.items():
                    if len(lines_in_region) > 3:  # Signatures have multiple strokes
                        
                        # Find bounding box of all lines in region
                        xs = []
                        ys = []
                        for x1, y1, x2, y2 in lines_in_region:
                            xs.extend([x1, x2])
                            ys.extend([y1, y2])
                        
                        if xs and ys:
                            x, y = min(xs), min(ys)
                            w, h = max(xs) - x, max(ys) - y
                            
                            if w > self.min_width and h > self.min_height:
                                # Count line crossings (signature complexity)
                                complexity = len(lines_in_region)
                                
                                if complexity > 4:  # Signatures are complex
                                    detections.append({
                                        'type': 'SIGNATURE',
                                        'bbox': (x, y, w, h),
                                        'confidence': 0.70,
                                        'method': 'handwriting_pattern',
                                        'complexity': complexity
                                    })
                                    self.logger.debug(f"Handwriting signature at ({x}, {y}): complexity={complexity}")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Handwriting detection failed: {e}")
            return []
    
    def _calculate_curvature(self, contour: np.ndarray) -> float:
        """
        Calculate curvature of contour.
        High curvature = signature (smooth curves)
        Low curvature = printed text (straight lines)
        """
        try:
            if len(contour) < 5:
                return 0.0
            
            # Approximate curve with specific precision
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Curvature = 1 - (approximated_points / original_points)
            # Higher curvature for smooth curves, lower for polygonal shapes
            curvature = 1.0 - (len(approx) / len(contour))
            
            return max(0.0, min(1.0, curvature))
            
        except:
            return 0.0
    
    def _deduplicate_signatures(self, detections: List[Dict]) -> List[Dict]:
        """Remove overlapping signature detections."""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        unique = []
        for det in sorted_dets:
            bbox = det['bbox']
            
            is_duplicate = False
            for existing in unique:
                if self._boxes_overlap(bbox, existing['bbox']):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(det)
        
        return unique
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple, threshold: float = 0.5) -> bool:
        """Check if boxes overlap."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        iou = intersection / min(area1, area2) if min(area1, area2) > 0 else 0
        return iou > threshold


# ============================================
# QR CODE DETECTOR
# ============================================

class QRCodeDetector:
    """QR code detection using pattern matching."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect QR codes."""
        if not OPENCV_AVAILABLE:
            return []
        
        try:
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            detections = []
            
            blobs = self._detect_blob_patterns(img_array)
            grids = self._detect_grid_patterns(img_array)
            
            all_detections = blobs + grids
            unique_detections = self._deduplicate_detections(all_detections)
            
            self.logger.info(f"QR patterns: {len(unique_detections)}")
            return unique_detections
            
        except Exception as e:
            self.logger.error(f"QR detection failed: {e}")
            return []
    
    def _detect_blob_patterns(self, gray_img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect QR blob patterns."""
        try:
            _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            height, width = gray_img.shape
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 500 < area < (width * height * 0.15):
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.6 < aspect_ratio < 1.4 and w > 30:
                        detections.append({
                            'type': 'QR_CODE',
                            'bbox': (x, y, w, h),
                            'confidence': 0.70,
                            'method': 'blob_pattern'
                        })
            
            return detections
        except:
            return []
    
    def _detect_grid_patterns(self, gray_img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect QR grid patterns."""
        try:
            detections = []
            height, width = gray_img.shape
            
            for y in range(0, max(height - 50, 1), 25):
                for x in range(0, max(width - 50, 1), 25):
                    window = gray_img[y:y+50, x:x+50]
                    variance = np.var(window)
                    
                    if variance > 3000:
                        detections.append({
                            'type': 'QR_CODE',
                            'bbox': (x, y, 50, 50),
                            'confidence': 0.65,
                            'method': 'grid_pattern'
                        })
            
            return detections
        except:
            return []
    
    def _deduplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove overlapping detections."""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        unique = []
        
        for det in sorted_dets:
            bbox = det['bbox']
            is_duplicate = any(self._boxes_overlap(bbox, e['bbox']) for e in unique)
            if not is_duplicate:
                unique.append(det)
        
        return unique
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple, threshold: float = 0.5) -> bool:
        """Check box overlap."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area = min(w1 * h1, w2 * h2)
        
        return (intersection / area if area > 0 else 0) > threshold


# ============================================
# MAIN VISUAL PII DETECTOR
# ============================================

class VisualPIIDetector:
    """
    Comprehensive visual PII detector.
    - Signatures (NEW!)
    - YOLO for faces/people
    - QR codes
    - Photos/seals
    """
    
    def __init__(self, use_gpu: bool = True, model_path: str = None):
        """Initialize all detectors."""
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.model = None
        self.signature_detector = SignatureDetector()
        self.qr_detector = QRCodeDetector()
        
        logger.info(f"VisualPIIDetector on {self.device}")
        
        if YOLO_AVAILABLE:
            self._load_model(model_path)
    
    def _load_model(self, custom_path: str = None):
        """Load YOLO model."""
        models_to_try = [("YOLOv8n", "yolov8n.pt"), ("YOLOv8s", "yolov8s.pt")]
        
        for name, path in models_to_try:
            try:
                self.model = YOLO(path)
                if self.use_gpu:
                    try:
                        self.model.to('cuda')
                    except:
                        pass
                logger.info(f"✓ {name} loaded")
                return
            except Exception as e:
                logger.warning(f"Failed {name}: {e}")
    
    def detect(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect all visual PII."""
        boxes = []
        
        # Priority 1: Signatures
        sig_detections = self.signature_detector.detect(image)
        for sig in sig_detections:
            x, y, w, h = sig['bbox']
            boxes.append((x, y, w, h))
        
        # Priority 2: QR Codes
        qr_detections = self.qr_detector.detect(image)
        for qr in qr_detections:
            x, y, w, h = qr['bbox']
            boxes.append((x, y, w, h))
        
        # Priority 3: YOLO
        if self.model:
            boxes.extend(self._detect_yolo(image))
        
        # Priority 4: High contrast regions
        boxes.extend(self._detect_high_contrast_regions(image))
        
        logger.info(f"Total visual detections: {len(boxes)}")
        return boxes
    
    def _detect_yolo(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """YOLO detection (Optimized for speed)."""
        try:
            img_array = np.array(image)
            # Optimize inference speed: downscale to 320 and use FP16 on GPU
            results = self.model(
                img_array, 
                conf=0.25, 
                verbose=False, 
                classes=[0], 
                imgsz=320, 
                half=True if getattr(self, 'use_gpu', False) else False
            )
            
            boxes = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = box
                        boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            
            return boxes
        except:
            return []
    
    def _detect_high_contrast_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect high-contrast regions."""
        try:
            img_array = np.array(image.convert('L'))
            boxes = []
            
            for y in range(0, max(img_array.shape[0] - 50, 1), 25):
                for x in range(0, max(img_array.shape[1] - 50, 1), 25):
                    window = img_array[y:y+50, x:x+50]
                    
                    if np.std(window) > 30:
                        boxes.append((x, y, 50, 50))
            
            return boxes
        except:
            return []
    
    def detect_visual_pii(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect all visual PII with metadata."""
        detections = []
        
        # Signatures
        for sig in self.signature_detector.detect(image):
            x, y, w, h = sig['bbox']
            detections.append({
                'class_name': 'signature',
                'bbox': [x, y, x + w, y + h],
                'confidence': sig['confidence'],
                'risk': 'HIGH'
            })
        
        # QR codes
        for qr in self.qr_detector.detect(image):
            x, y, w, h = qr['bbox']
            detections.append({
                'class_name': 'qr_code',
                'bbox': [x, y, x + w, y + h],
                'confidence': qr['confidence'],
                'risk': 'HIGH'
            })
        
        # Others
        for box in self.detect(image):
            x, y, w, h = box
            detections.append({
                'class_name': 'visual_pii',
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.75,
                'risk': 'MEDIUM'
            })
        
        return detections


class StampDetector:
    """Heuristic stamp detector (seal-like circular/elliptic high-saturation regions)."""

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not OPENCV_AVAILABLE:
            return []

        detections: List[Dict[str, Any]] = []
        try:
            bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # Typical ink stamp hues: red/blue ranges with moderate saturation.
            red1 = cv2.inRange(hsv, (0, 50, 40), (12, 255, 255))
            red2 = cv2.inRange(hsv, (165, 50, 40), (180, 255, 255))
            blue = cv2.inRange(hsv, (90, 40, 40), (135, 255, 255))
            mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), blue)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 1200:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w < 35 or h < 35:
                    continue
                aspect = w / max(h, 1)
                if aspect < 0.5 or aspect > 2.0:
                    continue
                detections.append({'bbox': (x, y, w, h), 'confidence': 0.7, 'type': 'STAMP'})
        except Exception as e:
            logger.error(f"Stamp detection failed: {e}")

        return detections


class FingerprintDetector:
    """Heuristic fingerprint detector using ridge-like texture density."""

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not OPENCV_AVAILABLE:
            return []

        detections: List[Dict[str, Any]] = []
        try:
            gray = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 60, 140)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 800 or area > 25000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w < 25 or h < 25:
                    continue
                aspect = w / max(h, 1)
                if aspect < 0.6 or aspect > 1.7:
                    continue
                roi = edges[y:y + h, x:x + w]
                if roi.size == 0:
                    continue
                density = float(np.count_nonzero(roi)) / float(roi.size)
                if 0.10 <= density <= 0.45:
                    detections.append({'bbox': (x, y, w, h), 'confidence': 0.65, 'type': 'FINGERPRINT'})
        except Exception as e:
            logger.error(f"Fingerprint detection failed: {e}")

        return detections


# ============================================
# HELPER FUNCTIONS
# ============================================

def detect_pii_patterns(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Detect PII patterns."""
    findings = []
    for label, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((label, (match.start(), match.end())))
    return findings


def is_valid_email(text: str) -> bool:
    """Check email."""
    return bool(PII_PATTERNS["EMAIL"].fullmatch(text))


def is_valid_phone(text: str) -> bool:
    """Check phone."""
    return bool(PII_PATTERNS["PHONE"].search(text))


def redact_text_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """Redact PII in text."""
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(replacement, redacted)
    return redacted
