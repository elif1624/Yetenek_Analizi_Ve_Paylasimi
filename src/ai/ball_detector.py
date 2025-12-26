"""
Top tespiti için SAM3 kullanımı
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from .detection import PlayerDetector

logger = logging.getLogger(__name__)


class BallDetector:
    """SAM3 ile top tespiti"""
    
    def __init__(self, sam3_model=None, conf_threshold: float = 0.4):
        """
        Args:
            sam3_model: SAM3 model instance (None ise otomatik oluşturulur)
            conf_threshold: Confidence threshold
        """
        # PlayerDetector'ı "basketball" prompt ile kullan
        self.detector = PlayerDetector(
            sam3_model=sam3_model,
            text_prompt="basketball",  # Top için text prompt
            conf_threshold=conf_threshold,
            use_local=True
        )
        logger.info(f"BallDetector başlatıldı (prompt: 'basketball', threshold: {conf_threshold})")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Frame'deki topu tespit et
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Detection listesi (genellikle 1 top olmalı)
        """
        detections = self.detector.detect(frame)
        
        # Top genellikle tek olmalı, en yüksek confidence'ı al
        if len(detections) > 0:
            # En yüksek confidence'lı detection
            best_detection = max(detections, key=lambda d: d['confidence'])
            logger.debug(f"Top tespit edildi: confidence={best_detection['confidence']:.2f}")
            return [best_detection]
        
        return []
    
    def get_ball_position(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Topun pozisyonunu al (center point)
        
        Returns:
            (center_x, center_y) veya None
        """
        detections = self.detect(frame)
        
        if not detections:
            return None
        
        bbox = detections[0]['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        return (center_x, center_y)


