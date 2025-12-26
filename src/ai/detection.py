"""Oyuncu tespiti modülü - SAM3 ile detection"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Union
import logging

SAM3_LOCAL_AVAILABLE = False
SAM3_API_AVAILABLE = False

# SAM3Local import kontrolü
try:
    from .sam3_local import SAM3Local
    SAM3_LOCAL_AVAILABLE = True
except (ImportError, Exception) as e:
    pass

# SAM3Client import kontrolü  
try:
    from .sam3_client import SAM3Client
    SAM3_API_AVAILABLE = True
except (ImportError, Exception) as e:
    pass

logger = logging.getLogger(__name__)


class PlayerDetector:
    """SAM3 tabanlı oyuncu tespiti"""
    
    def __init__(
        self,
        sam3_model: Optional[Union['SAM3Local', 'SAM3Client']] = None,
        text_prompt: str = "basketball player",
        conf_threshold: float = 0.5,
        use_local: bool = True
    ):
        """
        Args:
            sam3_model: SAM3 model instance (SAM3Local veya SAM3Client)
            text_prompt: Text prompt for detection (e.g., "basketball player", "person")
            conf_threshold: Confidence threshold for detections
            use_local: Yerel model kullan (True) veya API (False)
        """
        if sam3_model is None:
            if use_local and SAM3_LOCAL_AVAILABLE:
                from .sam3_local import SAM3Local
                sam3_model = SAM3Local()
            elif SAM3_API_AVAILABLE:
                from .sam3_client import SAM3Client
                sam3_model = SAM3Client()
            else:
                raise ImportError("SAM3 model yüklenemedi. transformers yüklü olmalı.")
        
        self.sam3_model = sam3_model
        self.text_prompt = text_prompt
        self.conf_threshold = conf_threshold
        logger.info(f"SAM3 PlayerDetector başlatıldı (prompt: '{text_prompt}', local: {use_local})")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Frame'deki oyuncuları SAM3 ile tespit et
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Detection listesi, her detection:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str,
                'mask': np.ndarray (optional)
            }
        """
        try:
            # SAM3 ile text prompt ile segmentasyon
            result = self.sam3_model.segment_with_text(frame, self.text_prompt)
            
            # SAM3 sonuçlarını parse et
            if isinstance(result, dict) and 'detections' in result:
                # SAM3Local formatı
                detections = result['detections']
            else:
                # SAM3Client formatı (eski)
                detections = self._parse_sam3_result(result, frame.shape)
            
            # Confidence threshold ile filtrele
            detections = [
                det for det in detections 
                if det['confidence'] >= self.conf_threshold
            ]
            
            logger.debug(f"SAM3 ile {len(detections)} oyuncu tespit edildi")
            return detections
            
        except Exception as e:
            logger.error(f"SAM3 detection hatası: {e}")
            return []
    
    def _parse_sam3_result(
        self,
        result: Dict,
        image_shape: tuple
    ) -> List[Dict]:
        """
        SAM3 API sonucunu parse et ve detection formatına çevir
        
        Args:
            result: SAM3 API response
            image_shape: (height, width, channels)
            
        Returns:
            Detection listesi
        """
        detections = []
        
        # SAM3 response formatına göre parse
        # Not: Gerçek API response formatı test edilerek güncellenebilir
        if isinstance(result, dict):
            # Mask'lar ve bounding box'lar
            if 'masks' in result:
                masks = result['masks']
                scores = result.get('scores', [1.0] * len(masks))
                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    # Mask'tan bounding box hesapla
                    bbox = self._mask_to_bbox(mask, image_shape)
                    
                    if bbox:
                        detections.append({
                            'bbox': bbox,
                            'confidence': float(score),
                            'class_id': 0,  # person class
                            'class_name': 'person',
                            'mask': mask
                        })
            
            # Alternatif format: bounding boxes direkt
            elif 'boxes' in result:
                boxes = result['boxes']
                scores = result.get('scores', [1.0] * len(boxes))
                
                for box, score in zip(boxes, scores):
                    detections.append({
                        'bbox': list(box),
                        'confidence': float(score),
                        'class_id': 0,
                        'class_name': 'person'
                    })
            
            # Alternatif format: segments
            elif 'segments' in result:
                segments = result['segments']
                scores = result.get('scores', [1.0] * len(segments))
                
                for segment, score in zip(segments, scores):
                    # Segment'ten bbox hesapla
                    if isinstance(segment, list) and len(segment) > 0:
                        # Polygon formatından bbox
                        points = np.array(segment)
                        x1, y1 = points.min(axis=0)
                        x2, y2 = points.max(axis=0)
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        
                        detections.append({
                            'bbox': bbox,
                            'confidence': float(score),
                            'class_id': 0,
                            'class_name': 'person'
                        })
            
            # Eğer direkt detection formatı varsa
            elif 'detections' in result:
                detections = result['detections']
        
        elif isinstance(result, list):
            # Liste formatı
            for item in result:
                if isinstance(item, dict):
                    if 'bbox' in item:
                        detections.append(item)
        
        # Bbox formatını normalize et (x1, y1, x2, y2)
        for det in detections:
            if 'bbox' in det:
                bbox = det['bbox']
                if len(bbox) == 4:
                    # Zaten doğru formatta
                    det['bbox'] = [float(x) for x in bbox]
                else:
                    logger.warning(f"Beklenmeyen bbox formatı: {bbox}")
        
        return detections
    
    def _mask_to_bbox(
        self,
        mask: np.ndarray,
        image_shape: tuple
    ) -> Optional[List[float]]:
        """
        Mask'tan bounding box hesapla
        
        Args:
            mask: Binary mask array
            image_shape: (height, width, channels)
            
        Returns:
            [x1, y1, x2, y2] veya None
        """
        try:
            if isinstance(mask, list):
                mask = np.array(mask)
            
            if mask.size == 0:
                return None
            
            # Mask'ı 2D yap
            if len(mask.shape) > 2:
                mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.max(axis=2)
            
            # Binary mask
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Koordinatları bul
            coords = np.where(mask_binary > 0)
            
            if len(coords[0]) == 0:
                return None
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            return [float(x_min), float(y_min), float(x_max), float(y_max)]
            
        except Exception as e:
            logger.warning(f"Mask'tan bbox hesaplama hatası: {e}")
            return None
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Birden fazla frame'i batch olarak işle
        
        Args:
            frames: Frame listesi
            
        Returns:
            Her frame için detection listesi
        """
        all_results = []
        for frame in frames:
            detections = self.detect(frame)
            all_results.append(detections)
        
        return all_results


def filter_detections_by_size(
    detections: List[Dict],
    min_area: int = 1000,
    max_area: int = 50000,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 0.8
) -> List[Dict]:
    """
    Tespitleri boyut ve aspect ratio'ya göre filtrele
    (Basketbol oyuncuları genellikle belirli bir boyutta ve uzun formda olur)
    
    Args:
        detections: Detection listesi
        min_area: Minimum bbox alanı
        max_area: Maksimum bbox alanı
        min_aspect_ratio: Minimum aspect ratio (height/width)
        max_aspect_ratio: Maksimum aspect ratio
        
    Returns:
        Filtrelenmiş detection listesi
    """
    filtered = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = height / width if width > 0 else 0
        
        if (min_area <= area <= max_area and 
            min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            filtered.append(det)
    
    return filtered
