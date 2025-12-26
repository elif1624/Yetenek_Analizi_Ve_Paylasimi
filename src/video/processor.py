"""Video frame çıkarma ve preprocessing"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional
import logging
from .loader import VideoLoader

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video işleme ve frame extraction"""
    
    def __init__(self, video_path: Path, fps: Optional[float] = None):
        """
        Args:
            video_path: Video dosyasının yolu
            fps: Frame extraction için FPS (None ise video FPS'i kullanılır)
        """
        self.video_loader = VideoLoader(video_path)
        self.video_path = video_path
        self.fps = fps
        self.metadata = None
    
    def get_frames(self, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Video'dan frame'leri çıkar
        
        Args:
            max_frames: Maksimum frame sayısı (None ise tüm frame'ler)
            
        Yields:
            (frame_number, frame_array) tuple'ları
        """
        cap = self.video_loader.open()
        self.metadata = self.video_loader.get_metadata()
        
        video_fps = self.metadata['fps']
        extraction_fps = self.fps if self.fps else video_fps
        
        # Frame atlama oranı
        frame_skip = max(1, int(video_fps / extraction_fps))
        
        frame_number = 0
        extracted_count = 0
        
        logger.info(f"Frame extraction başlıyor: FPS={extraction_fps}, Skip={frame_skip}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Belirtilen FPS'e göre frame atla
            if frame_number % frame_skip == 0:
                yield extracted_count, frame
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_number += 1
    
    def extract_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Belirli bir zamandaki frame'i çıkar
        
        Args:
            time_seconds: Zaman (saniye)
            
        Returns:
            Frame array veya None
        """
        cap = self.video_loader.open()
        self.metadata = self.video_loader.get_metadata()
        
        fps = self.metadata['fps']
        frame_number = int(time_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            return frame
        return None
    
    def extract_sample_frames(self, num_samples: int = 10) -> list:
        """
        Video'dan eşit aralıklarla örnek frame'ler çıkar
        
        Args:
            num_samples: Örnek frame sayısı
            
        Returns:
            Frame listesi
        """
        self.metadata = self.video_loader.get_metadata()
        duration = self.metadata['duration']
        
        frames = []
        time_intervals = np.linspace(0, duration, num_samples, endpoint=False)
        
        for time in time_intervals:
            frame = self.extract_frame_at_time(time)
            if frame is not None:
                frames.append((time, frame))
        
        return frames
    
    def resize_frame(self, frame: np.ndarray, max_size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
        """
        Frame'i yeniden boyutlandır
        
        Args:
            frame: Frame array
            max_size: Maksimum (width, height)
            
        Returns:
            Yeniden boyutlandırılmış frame
        """
        height, width = frame.shape[:2]
        max_width, max_height = max_size
        
        # Aspect ratio'yu koru
        scale = min(max_width / width, max_height / height)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame


