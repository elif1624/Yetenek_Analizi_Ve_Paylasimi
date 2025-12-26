"""Video yükleme ve doğrulama"""

import cv2
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoLoader:
    """Video dosyalarını yükleyen ve doğrulayan sınıf"""
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    
    def __init__(self, video_path: Path):
        """
        Args:
            video_path: Video dosyasının yolu
        """
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self._validate()
    
    def _validate(self) -> None:
        """Video dosyasını doğrula"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video dosyası bulunamadı: {self.video_path}")
        
        if self.video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Desteklenmeyen video formatı: {self.video_path.suffix}. "
                f"Desteklenen formatlar: {', '.join(self.SUPPORTED_FORMATS)}"
            )
    
    def open(self) -> cv2.VideoCapture:
        """Video dosyasını aç"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise IOError(f"Video dosyası açılamadı: {self.video_path}")
            logger.info(f"Video açıldı: {self.video_path}")
        return self.cap
    
    def get_metadata(self) -> dict:
        """Video metadata bilgilerini al"""
        cap = self.open()
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        
        logger.info(f"Video metadata: {metadata}")
        return metadata
    
    def close(self) -> None:
        """Video dosyasını kapat"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video kapatıldı")
    
    def __enter__(self):
        """Context manager giriş"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkış"""
        self.close()
    
    def __del__(self):
        """Destructor"""
        self.close()


