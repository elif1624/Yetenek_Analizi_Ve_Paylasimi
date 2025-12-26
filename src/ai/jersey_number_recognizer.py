"""Forma numarası tanıma modülü - OCR tabanlı"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# PaddleOCR import
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PaddleOCR bulunamadı. Alternatif olarak EasyOCR kullanılabilir.")

# EasyOCR alternatif
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class JerseyNumberRecognizer:
    """Forma numarası tanıma sistemi"""
    
    def __init__(
        self,
        use_paddle: bool = True,
        lang: str = 'en',
        use_gpu: bool = False
    ):
        """
        Args:
            use_paddle: PaddleOCR kullan (True) veya EasyOCR (False)
            lang: OCR dili
            use_gpu: GPU kullan (eğer mevcutsa)
        """
        self.use_paddle = use_paddle and PADDLEOCR_AVAILABLE
        self.lang = lang
        self.use_gpu = use_gpu
        
        if self.use_paddle:
            logger.info("PaddleOCR yükleniyor...")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info("PaddleOCR hazır")
        elif EASYOCR_AVAILABLE:
            logger.info("EasyOCR yükleniyor...")
            self.ocr = easyocr.Reader([lang], gpu=use_gpu)
            logger.info("EasyOCR hazır")
        else:
            raise ImportError(
                "OCR kütüphanesi bulunamadı. "
                "pip install paddleocr veya pip install easyocr yapın."
            )
    
    def extract_jersey_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Bounding box'tan forma bölgesini çıkar (üst kısmı - numara genellikle orada)
        
        Args:
            frame: Video frame
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Forma bölgesi (cropped image)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Bounding box'ı crop et
        height = y2 - y1
        width = x2 - x1
        
        # Forma numarası genellikle üst 1/3 kısımda
        jersey_top = y1
        jersey_bottom = y1 + int(height * 0.4)  # Üst %40
        jersey_left = x1 + int(width * 0.3)  # Ortaya yakın
        jersey_right = x2 - int(width * 0.3)
        
        # Sınırları kontrol et
        jersey_top = max(0, jersey_top)
        jersey_bottom = min(frame.shape[0], jersey_bottom)
        jersey_left = max(0, jersey_left)
        jersey_right = min(frame.shape[1], jersey_right)
        
        jersey_region = frame[jersey_top:jersey_bottom, jersey_left:jersey_right]
        
        return jersey_region
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        OCR için görüntüyü ön işle (contrast, binarization vb.)
        
        Args:
            image: Forma bölgesi
            
        Returns:
            Ön işlenmiş görüntü
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Binarization (threshold)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Resize (OCR daha iyi çalışır)
        scale_factor = 3
        resized = cv2.resize(
            denoised,
            (denoised.shape[1] * scale_factor, denoised.shape[0] * scale_factor),
            interpolation=cv2.INTER_CUBIC
        )
        
        return resized
    
    def recognize_number(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[int]:
        """
        Forma numarasını tanı
        
        Args:
            frame: Video frame
            bbox: Oyuncu bounding box [x1, y1, x2, y2]
            
        Returns:
            Tanınan numara (1-99 arası) veya None
        """
        # Forma bölgesini çıkar
        jersey_region = self.extract_jersey_region(frame, bbox)
        
        if jersey_region.size == 0:
            return None
        
        # Ön işle
        processed = self.preprocess_for_ocr(jersey_region)
        
        # OCR
        try:
            if self.use_paddle:
                results = self.ocr.ocr(processed, cls=True)
                # PaddleOCR format: [[(bbox, (text, confidence)), ...], ...]
                if results and results[0]:
                    for line in results[0]:
                        text = line[1][0].strip()
                        confidence = line[1][1]
                        
                        # Numara olup olmadığını kontrol et
                        number = self._extract_number(text)
                        if number and confidence > 0.5:
                            logger.debug(f"Numara tanındı: {number} (confidence: {confidence:.2f})")
                            return number
            else:
                # EasyOCR format
                results = self.ocr.readtext(processed)
                for (bbox_text, text, confidence) in results:
                    number = self._extract_number(text)
                    if number and confidence > 0.5:
                        logger.debug(f"Numara tanındı: {number} (confidence: {confidence:.2f})")
                        return number
        
        except Exception as e:
            logger.warning(f"OCR hatası: {e}")
            return None
        
        return None
    
    def _extract_number(self, text: str) -> Optional[int]:
        """
        Metinden numara çıkar (1-99 arası)
        
        Args:
            text: OCR'dan gelen metin
            
        Returns:
            Numara veya None
        """
        # Rakamları çıkar
        digits = ''.join(filter(str.isdigit, text))
        
        if not digits:
            return None
        
        # İlk 2 rakamı al (0-99 arası numaralar için)
        number = int(digits[:2]) if len(digits) >= 2 else int(digits[0])
        
        # Geçerli numara aralığı (basketbol için genellikle 0-99)
        if 0 <= number <= 99:
            return number
        
        return None
    
    def recognize_batch(
        self,
        frame: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Optional[int]]:
        """
        Birden fazla bbox için numara tanı
        
        Args:
            frame: Video frame
            bboxes: Bounding box listesi
            
        Returns:
            Her bbox için tanınan numara listesi
        """
        numbers = []
        for bbox in bboxes:
            number = self.recognize_number(frame, bbox)
            numbers.append(number)
        
        return numbers


