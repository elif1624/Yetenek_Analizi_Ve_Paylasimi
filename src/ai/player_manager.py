"""Oyuncu yönetim sistemi - Tracking, numara tanıma ve ID eşleştirme"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    from .sam3_local import SAM3Local
    SAM3_LOCAL_AVAILABLE = True
except ImportError:
    SAM3_LOCAL_AVAILABLE = False

from .detection import PlayerDetector
from .tracking import PlayerTracker, TrackedPlayer
from .jersey_number_recognizer import JerseyNumberRecognizer

logger = logging.getLogger(__name__)


class PlayerManager:
    """Oyuncu tespiti, takibi ve numara tanıma yönetimi"""
    
    def __init__(
        self,
        sam3_model=None,
        detection_text_prompt: str = "basketball player",
        detection_conf: float = 0.5,
        track_thresh: float = 0.25,
        frame_rate: float = 30.0,
        jersey_recognition_enabled: bool = True,
        jersey_recognition_interval: int = 30,  # Her N frame'de bir numara tanı
        use_local_sam3: bool = True
    ):
        """
        Args:
            sam3_model: SAM3 model instance (None ise yeni oluşturulur)
            detection_text_prompt: SAM3 text prompt (e.g., "basketball player", "person")
            detection_conf: Detection confidence threshold
            track_thresh: Tracking threshold
            frame_rate: Video frame rate
            jersey_recognition_enabled: Numara tanıma aktif mi
            jersey_recognition_interval: Kaç frame'de bir numara tanıma yapılacak
            use_local_sam3: Yerel SAM3 model kullan (True) veya API (False)
        """
        # SAM3 model (yerel veya API)
        if sam3_model is None:
            if use_local_sam3 and SAM3_LOCAL_AVAILABLE:
                sam3_model = SAM3Local()
            else:
                from .sam3_client import SAM3Client
                sam3_model = SAM3Client()
        
        self.sam3_model = sam3_model
        
        # Modüller - SAM3 tabanlı detection
        self.detector = PlayerDetector(
            sam3_model=sam3_model,
            text_prompt=detection_text_prompt,
            conf_threshold=detection_conf,
            use_local=use_local_sam3
        )
        self.tracker = PlayerTracker(
            track_thresh=track_thresh,
            frame_rate=frame_rate
        )
        
        self.jersey_recognizer: Optional[JerseyNumberRecognizer] = None
        if jersey_recognition_enabled:
            try:
                self.jersey_recognizer = JerseyNumberRecognizer()
                logger.info("Jersey number recognizer başlatıldı")
            except Exception as e:
                logger.warning(f"Jersey recognizer başlatılamadı: {e}")
                self.jersey_recognizer = None
        
        self.jersey_recognition_enabled = jersey_recognition_enabled and self.jersey_recognizer is not None
        self.jersey_recognition_interval = jersey_recognition_interval
        
        # Numara - Track ID eşleştirmesi
        self.jersey_to_track_id: Dict[int, int] = {}  # jersey_number -> track_id
        self.track_id_to_jersey: Dict[int, int] = {}  # track_id -> jersey_number
        
        # Frame sayacı (numara tanıma için)
        self.frame_count = 0
        
        logger.info("PlayerManager başlatıldı")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> List[TrackedPlayer]:
        """
        Bir frame'i işle: tespit, takip ve numara tanıma
        
        Args:
            frame: Video frame
            frame_number: Frame numarası
            
        Returns:
            TrackedPlayer listesi
        """
        self.frame_count = frame_number
        
        # 1. Oyuncu tespiti
        detections = self.detector.detect(frame)
        
        if not detections:
            return []
        
        # 2. Tracking update
        tracked_players = self.tracker.update(detections, frame, frame_number)
        
        # 3. Numara tanıma (belirli aralıklarla)
        if (self.jersey_recognition_enabled and 
            frame_number % self.jersey_recognition_interval == 0):
            self._recognize_jersey_numbers(frame, tracked_players)
        
        return tracked_players
    
    def _recognize_jersey_numbers(
        self,
        frame: np.ndarray,
        tracked_players: List[TrackedPlayer]
    ) -> None:
        """
        Takip edilen oyuncuların forma numaralarını tanı
        
        Args:
            frame: Video frame
            tracked_players: Takip edilen oyuncular
        """
        if not self.jersey_recognizer:
            return
        
        for player in tracked_players:
            # Eğer zaten numarası biliniyorsa, tekrar tanıma
            if player.track_id in self.track_id_to_jersey:
                continue
            
            # Numara tanı
            jersey_number = self.jersey_recognizer.recognize_number(
                frame,
                player.bbox
            )
            
            if jersey_number:
                # Eşleştirme
                self.assign_jersey_number(player.track_id, jersey_number)
    
    def assign_jersey_number(self, track_id: int, jersey_number: int) -> None:
        """
        Bir track_id'ye jersey numarası ata
        
        Args:
            track_id: Takip ID
            jersey_number: Forma numarası
        """
        # Çakışma kontrolü
        if jersey_number in self.jersey_to_track_id:
            old_track_id = self.jersey_to_track_id[jersey_number]
            if old_track_id != track_id:
                logger.warning(
                    f"Numara #{jersey_number} zaten track_id {old_track_id} ile eşleşmiş. "
                    f"Yeni eşleştirme: track_id {track_id}"
                )
                # Eski eşleştirmeyi kaldır
                if old_track_id in self.track_id_to_jersey:
                    del self.track_id_to_jersey[old_track_id]
        
        # Yeni eşleştirme
        self.jersey_to_track_id[jersey_number] = track_id
        self.track_id_to_jersey[track_id] = jersey_number
        
        # Tracker'a da bildir
        self.tracker.assign_jersey_number(track_id, jersey_number)
        
        logger.info(f"Eşleştirme: Track ID {track_id} <-> Jersey #{jersey_number}")
    
    def get_player_by_jersey(self, jersey_number: int) -> Optional[TrackedPlayer]:
        """
        Jersey numarasına göre oyuncuyu bul
        
        Args:
            jersey_number: Forma numarası
            
        Returns:
            En son TrackedPlayer veya None
        """
        track_id = self.jersey_to_track_id.get(jersey_number)
        if track_id is None:
            return None
        
        trajectory = self.tracker.get_player_trajectory(track_id)
        if trajectory:
            return trajectory[-1]  # En son pozisyon
        
        return None
    
    def get_player_trajectory(self, identifier: int, by_jersey: bool = False) -> List[TrackedPlayer]:
        """
        Bir oyuncunun trajectory'sini al
        
        Args:
            identifier: Track ID veya Jersey numarası
            by_jersey: True ise identifier jersey numarası, False ise track_id
            
        Returns:
            TrackedPlayer listesi
        """
        if by_jersey:
            track_id = self.jersey_to_track_id.get(identifier)
            if track_id is None:
                return []
            return self.tracker.get_player_trajectory(track_id)
        else:
            return self.tracker.get_player_trajectory(identifier)
    
    def get_all_tracked_players(self) -> List[TrackedPlayer]:
        """Tüm aktif takip edilen oyuncuları al"""
        tracked_players = []
        for track_id in self.tracker.get_active_tracks():
            trajectory = self.tracker.get_player_trajectory(track_id)
            if trajectory:
                tracked_players.append(trajectory[-1])
        
        return tracked_players
    
    def get_player_statistics(self) -> Dict:
        """
        Oyuncu istatistiklerini al
        
        Returns:
            İstatistikler dict'i
        """
        stats = {
            'total_tracks': len(self.tracker.get_active_tracks()),
            'jersey_mappings': len(self.jersey_to_track_id),
            'players_with_jersey': {}
        }
        
        for jersey_number, track_id in self.jersey_to_track_id.items():
            trajectory = self.tracker.get_player_trajectory(track_id)
            stats['players_with_jersey'][jersey_number] = {
                'track_id': track_id,
                'total_frames': len(trajectory),
                'first_frame': trajectory[0].frame_number if trajectory else None,
                'last_frame': trajectory[-1].frame_number if trajectory else None
            }
        
        return stats
    
    def reset(self) -> None:
        """Tüm verileri sıfırla"""
        self.tracker.reset()
        self.jersey_to_track_id.clear()
        self.track_id_to_jersey.clear()
        self.frame_count = 0
        logger.info("PlayerManager sıfırlandı")

