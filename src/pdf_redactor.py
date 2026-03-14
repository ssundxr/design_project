"""
PDF Redactor - Multi-page PDF PII Redaction
Converts PDF to images, redacts PII, converts back to PDF
Version: 1.0 Enterprise
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Try importing PDF libraries
try:
    import fitz  # PyMuPDF
    PDF_ENGINE_AVAILABLE = True
except ImportError:
    PDF_ENGINE_AVAILABLE = False
    logger.warning("PyMuPDF not available - install: pip install pymupdf")

try:
    import img2pdf
    IMG2PDF_AVAILABLE = True
except ImportError:
    IMG2PDF_AVAILABLE = False
    logger.warning("img2pdf not available - install: pip install img2pdf")

# Import redactor
from .redactor import PIIRedactor


class PDFRedactor:
    """
    Multi-page PDF PII Redactor.
    
    Features:
    - Converts PDF to images (page by page)
    - Redacts PII on each page
    - Converts back to PDF
    - Supports up to 100 pages
    - Generates comprehensive audit log
    """
    
    MAX_PAGES = 100
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize PDF redactor.
        
        Args:
            output_dir: Output directory for redacted PDFs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.pdf_output_dir = self.output_dir / "pdf_redacted"
        self.pdf_output_dir.mkdir(exist_ok=True)
        
        self.pdf_audit_dir = self.output_dir / "pdf_audit_logs"
        self.pdf_audit_dir.mkdir(exist_ok=True)
        
        # Initialize image redactor
        self.redactor = PIIRedactor(output_dir=str(output_dir))
        
        logger.info("✓ PDF Redactor initialized")
    
    def redact_pdf(
        self,
        pdf_path: str,
        output_filename: str = None,
        dpi: int = 200,
        progress_callback = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Redact PII from multi-page PDF.
        
        Args:
            pdf_path: Path to input PDF
            output_filename: Name for output PDF (auto-generated if None)
            dpi: DPI for PDF to image conversion (200 recommended)
            progress_callback: Optional callback(page, total_pages, status)
            
        Returns:
            Tuple of (output_pdf_path, audit_data)
        """
        if not PDF_ENGINE_AVAILABLE:
            raise Exception("pymupdf not installed. Run: pip install pymupdf")
        
        if not IMG2PDF_AVAILABLE:
            raise Exception("img2pdf not installed. Run: pip install img2pdf")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING PDF REDACTION: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Initialize audit data
        audit_data = {
            'filename': pdf_path.name,
            'timestamp': start_time.isoformat(),
            'dpi': dpi,
            'pages': [],
            'total_pages': 0,
            'total_detections': 0,
            'processing_time': 0,
            'statistics': {}
        }
        
        try:
            # Step 1: Extract PDF pages (image + words)
            if progress_callback:
                progress_callback(0, 0, "Extracting PDF pages...")
            
            logger.info("\n[1/3] Extracting PDF pages (PyMuPDF)...")
            page_payloads = self._extract_pdf_pages(pdf_path, dpi)
            
            total_pages = len(page_payloads)
            audit_data['total_pages'] = total_pages
            
            if total_pages == 0:
                raise Exception("No pages found in PDF")
            
            if total_pages > self.MAX_PAGES:
                raise Exception(f"PDF has {total_pages} pages. Maximum allowed: {self.MAX_PAGES}")
            
            logger.info(f"✓ Converted {total_pages} pages")
            
            # Step 2: Redact each page (PARALLELIZED)
            logger.info(f"\n[2/3] Redacting {total_pages} pages in parallel...")
            redacted_images = [None] * total_pages
            
            import concurrent.futures
            
            def process_page(page_idx_and_payload):
                idx, img, words_with_boxes = page_idx_and_payload
                page_n = idx + 1
                logger.info(f"--- Started Page {page_n}/{total_pages} ---")
                try:
                    r_img, p_audit = self.redactor.redact_image(
                        img,
                        filename=f"{pdf_path.stem}_page_{page_n}.jpg",
                        generate_audit=False,
                        words_with_boxes=words_with_boxes
                    )
                    return idx, r_img, p_audit
                except Exception as e:
                    logger.error(f"Error on page {page_n}: {e}")
                    return idx, img, {'detections': [], 'statistics': {}}
            
            completed = 0
            # Use max 4 workers to balance memory usage vs speed
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(process_page, (i, payload[0], payload[1])): i
                    for i, payload in enumerate(page_payloads)
                }
                
                for future in concurrent.futures.as_completed(futures):
                    idx, r_img, p_audit = future.result()
                    redacted_images[idx] = r_img
                    
                    page_n = idx + 1
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total_pages, f"Redacting page {completed}/{total_pages}")
                    
                    # Store page audit
                    page_summary = {
                        'page_number': page_n,
                        'detections': len(p_audit.get('detections', [])),
                        'statistics': p_audit.get('statistics', {}),
                        'risk_assessment': p_audit.get('risk_assessment', 'N/A')
                    }
                    audit_data['pages'].append(page_summary)
                    audit_data['total_detections'] += len(p_audit.get('detections', []))
                    
                    logger.info(f"✓ Page {page_n}: {len(p_audit.get('detections', []))} detections")
            
            # Step 3: Convert back to PDF
            if progress_callback:
                progress_callback(total_pages, total_pages, "Converting to PDF...")
            
            logger.info(f"\n[3/3] Converting to PDF...")
            
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{pdf_path.stem}_redacted_{timestamp}.pdf"
            
            output_path = self.pdf_output_dir / output_filename
            
            self._convert_images_to_pdf(redacted_images, output_path)
            
            logger.info(f"✓ Redacted PDF saved: {output_path.name}")
            
            # Finalize audit
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            audit_data['processing_time'] = round(processing_time, 2)
            audit_data['output_file'] = output_filename
            audit_data['statistics'] = self._calculate_pdf_statistics(audit_data['pages'])
            
            # Save audit
            self._save_audit(audit_data)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ PDF REDACTION COMPLETE")
            logger.info(f"  Total Pages: {total_pages}")
            logger.info(f"  Total Detections: {audit_data['total_detections']}")
            logger.info(f"  Processing Time: {processing_time:.2f}s")
            logger.info(f"  Output: {output_path}")
            logger.info(f"{'='*60}\n")
            
            return str(output_path), audit_data
        
        except Exception as e:
            logger.error(f"PDF redaction failed: {e}")
            raise
    
    def _extract_pdf_pages(self, pdf_path: Path, dpi: int) -> List[Tuple[Image.Image, List[Tuple[str, Tuple[int, int, int, int]]]]]:
        """
        Extract each PDF page as an image and word-level bounding boxes using PyMuPDF.
        """
        doc = None
        payloads: List[Tuple[Image.Image, List[Tuple[str, Tuple[int, int, int, int]]]]] = []
        try:
            doc = fitz.open(str(pdf_path))
            scale = max(dpi / 72.0, 0.1)
            matrix = fitz.Matrix(scale, scale)

            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                words_with_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
                for item in page.get_text("words"):
                    x0, y0, x1, y1, text = item[:5]
                    text = str(text).strip()
                    if not text:
                        continue
                    sx0, sy0, sx1, sy1 = int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)
                    w = max(1, sx1 - sx0)
                    h = max(1, sy1 - sy0)
                    words_with_boxes.append((text, (sx0, sy0, w, h)))

                payloads.append((image, words_with_boxes))

            return payloads
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise Exception(f"Failed to extract PDF pages: {e}")
        finally:
            if doc is not None:
                doc.close()
    
    def _convert_images_to_pdf(self, images: List[Image.Image], output_path: Path):
        """
        Convert list of images to PDF.
        
        Args:
            images: List of PIL Images
            output_path: Output PDF path
        """
        try:
            # Convert images to bytes
            image_bytes = []
            for img in images:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                from io import BytesIO
                img_buffer = BytesIO()
                img.save(img_buffer, format='JPEG', quality=95)
                image_bytes.append(img_buffer.getvalue())
            
            # Create PDF
            with open(output_path, 'wb') as f:
                f.write(img2pdf.convert(image_bytes))
            
            logger.info(f"✓ PDF created: {output_path.name} ({len(images)} pages)")
        
        except Exception as e:
            logger.error(f"PDF creation failed: {e}")
            raise Exception(f"Failed to create PDF: {e}")
    
    def _calculate_pdf_statistics(self, pages: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics for PDF."""
        stats = {
            'total_pages': len(pages),
            'total_detections': sum(p['detections'] for p in pages),
            'pages_with_pii': sum(1 for p in pages if p['detections'] > 0),
            'pages_without_pii': sum(1 for p in pages if p['detections'] == 0),
            'avg_detections_per_page': 0,
            'by_risk': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'by_type': {},
            'high_risk_pages': []
        }
        
        if len(pages) > 0:
            stats['avg_detections_per_page'] = round(
                stats['total_detections'] / len(pages), 2
            )
        
        # Aggregate statistics
        for page in pages:
            page_stats = page.get('statistics', {})
            
            # By risk
            for risk, count in page_stats.get('by_risk', {}).items():
                if risk in stats['by_risk']:
                    stats['by_risk'][risk] += count
            
            # By type
            for type_, count in page_stats.get('by_type', {}).items():
                stats['by_type'][type_] = stats['by_type'].get(type_, 0) + count
            
            # Track high-risk pages
            if page_stats.get('by_risk', {}).get('HIGH', 0) >= 3:
                stats['high_risk_pages'].append(page['page_number'])
        
        return stats
    
    def _save_audit(self, audit_data: Dict):
        """Save PDF audit log."""
        try:
            audit_id = hashlib.sha256(
                f"{audit_data['filename']}{audit_data['timestamp']}".encode()
            ).hexdigest()[:12]
            
            audit_file = self.pdf_audit_dir / f"pdf_audit_{audit_id}.json"
            
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
            audit_data['audit_file'] = audit_file.name
            audit_data['audit_path'] = str(audit_file)
            
            logger.info(f"✓ PDF audit saved: {audit_file.name}")
        except Exception as e:
            logger.error(f"Audit save failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get PDF redactor information."""
        return {
            'max_pages': self.MAX_PAGES,
            'pdf_engine_available': PDF_ENGINE_AVAILABLE,
            'img2pdf_available': IMG2PDF_AVAILABLE,
            'output_directory': str(self.pdf_output_dir),
            'audit_directory': str(self.pdf_audit_dir)
        }


def redact_pdf_simple(pdf_path: str, output_filename: str = None) -> str:
    """
    Simple function to redact a PDF.
    
    Args:
        pdf_path: Path to input PDF
        output_filename: Optional output filename
        
    Returns:
        Path to redacted PDF
    """
    redactor = PDFRedactor()
    output_path, _ = redactor.redact_pdf(pdf_path, output_filename)
    return output_path
