"""Geliştirilmiş tracking - Kalman filter ve daha iyi matching"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrackedPlayer:
    """Takip edilen oyuncu bilgileri"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    frame_number: int
    jersey_number: Optional[int] = None
    position: Optional[Tuple[float, float]] = None


class KalmanTracker:
    """Basit Kalman filter tabanlı tracker (hareket tahmini)"""
    
    def __init__(self, bbox: List[float]):
        """
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # State: [cx, cy, vx, vy, w, h]
        # cx, cy: center, vx, vy: velocity, w, h: width, height
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.state = np.array([cx, cy, 0, 0, w, h], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1000  # Covariance
        self.prev_cx = cx  # Önceki pozisyon (velocity için)
        self.prev_cy = cy
        
        # Process noise
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 10
    
    def predict(self) -> List[float]:
        """Bir sonraki pozisyonu tahmin et"""
        # Basit hareket modeli: x = x + v*dt (dt=1 frame)
        F = np.array([
            [1, 0, 1, 0, 0, 0],  # cx = cx + vx
            [0, 1, 0, 1, 0, 0],  # cy = cy + vy
            [0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 1, 0],  # w = w
            [0, 0, 0, 0, 0, 1]   # h = h
        ], dtype=np.float32)
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        # Predicted bbox
        cx, cy, _, _, w, h = self.state
        return [
            cx - w/2,
            cy - h/2,
            cx + w/2,
            cy + h/2
        ]
    
    def update(self, bbox: List[float]):
        """Ölçümle state'i güncelle"""
        # Measurement: [cx, cy, w, h]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float32)
        
        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Predicted measurement
        z_pred = H @ self.state
        
        # Innovation
        y = z - z_pred
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        
        # Velocity update (position difference'den hesapla)
        self.state[2] = cx - self.prev_cx  # Velocity = position difference
        self.state[3] = cy - self.prev_cy
        
        # Önceki pozisyonu güncelle
        self.prev_cx = cx
        self.prev_cy = cy


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU hesapla"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_distance_cost(box1: np.ndarray, box2: np.ndarray) -> float:
    """Center distance + size similarity cost"""
    # Center points
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    
    # Distance
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # Size similarity
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    size_diff = abs(w1 - w2) + abs(h1 - h2)
    
    # Normalize (basit normalizasyon)
    # Distance: 0-1000 pixels -> 0-1, Size diff: 0-500 -> 0-1
    normalized_dist = min(dist / 500.0, 1.0)
    normalized_size = min(size_diff / 200.0, 1.0)
    
    # Combined cost (lower is better)
    cost = normalized_dist * 0.7 + normalized_size * 0.3
    
    return cost


class ImprovedTracker:
    """Geliştirilmiş tracker - Kalman filter + daha iyi matching"""
    
    def __init__(
        self,
        iou_threshold: float = 0.4,  # Daha yüksek threshold (daha az yeni track)
        max_disappeared: int = 20,  # Daha uzun süre beklet
        distance_threshold: float = 300.0,  # Maksimum hareket mesafesi (pixel)
        min_track_length: int = 3  # Minimum track uzunluğu (filtreleme için)
    ):
        """
        Args:
            iou_threshold: IoU threshold
            max_disappeared: Maksimum kaybolma frame sayısı
            distance_threshold: Maksimum hareket mesafesi (pixel)
        """
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.min_track_length = min_track_length
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}  # track_id -> {bbox, kalman, ...}
        self.disappeared: Dict[int, int] = {}
        self.track_lengths: Dict[int, int] = {}  # Track uzunluklarını takip et
        
        logger.info(f"ImprovedTracker başlatıldı (iou_threshold={iou_threshold}, max_disappeared={max_disappeared})")
    
    def update(
        self,
        detections: List[Dict],
        frame_number: int
    ) -> List[TrackedPlayer]:
        """Detection'lar ile tracker'ı güncelle"""
        if not detections:
            # Tüm track'leri kaybolmuş olarak işaretle
            for track_id in list(self.tracks.keys()):
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                # Kalman ile tahmin yap
                if track_id in self.tracks:
                    kalman = self.tracks[track_id].get('kalman')
                    if kalman:
                        predicted_bbox = kalman.predict()
                        self.tracks[track_id]['bbox'] = predicted_bbox
                
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []
        
        # Tüm track'ler için tahmin yap (Kalman filter)
        predicted_boxes = {}
        for track_id, track_info in self.tracks.items():
            kalman = track_info.get('kalman')
            if kalman:
                predicted_boxes[track_id] = kalman.predict()
            else:
                predicted_boxes[track_id] = track_info['bbox']
        
        # Matching için cost matrix oluştur
        track_ids = list(self.tracks.keys())
        det_boxes = np.array([det['bbox'] for det in detections])
        
        # Cost matrix: IoU (yüksek = iyi) ve distance (düşük = iyi)
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            track_box = np.array(predicted_boxes[track_id])
            for j, det_box in enumerate(det_boxes):
                # IoU score (0-1, yüksek = iyi)
                iou = calculate_iou(track_box, det_box)
                
                # Distance cost (0-1, düşük = iyi)
                dist_cost = calculate_distance_cost(track_box, det_box)
                
                # Combined score (yüksek = iyi matching)
                # IoU ağırlıklı, distance sadece filtreleme için
                if iou > 0.15:  # Minimum IoU (biraz artırıldı)
                    # Distance threshold kontrolü
                    cx1 = (track_box[0] + track_box[2]) / 2
                    cy1 = (track_box[1] + track_box[3]) / 2
                    cx2 = (det_box[0] + det_box[2]) / 2
                    cy2 = (det_box[1] + det_box[3]) / 2
                    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                    
                    if dist > self.distance_threshold:
                        score = 0  # Çok uzak, eşleşme yok
                    else:
                        score = iou * (1.0 - dist_cost * 0.2)  # IoU dominant
                else:
                    score = 0
                
                cost_matrix[i, j] = score
        
        # Hungarian algorithm yerine greedy matching (daha hızlı)
        # Ama önce yüksek score'lardan başla
        matched_tracks = set()
        matched_dets = set()
        matches = []
        
        # Score'a göre sırala
        match_candidates = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if cost_matrix[i, j] >= self.iou_threshold:
                    match_candidates.append((cost_matrix[i, j], i, j))
        
        match_candidates.sort(reverse=True)  # Yüksek score'dan düşüğe
        
        # En iyi eşleşmeleri seç
        for score, i, j in match_candidates:
            if i not in matched_tracks and j not in matched_dets:
                matches.append((track_ids[i], j))
                matched_tracks.add(i)
                matched_dets.add(j)
        
        # Eşleşen track'leri güncelle
        tracked_players = []
        for track_id, det_idx in matches:
            det = detections[det_idx]
            bbox = det['bbox']
            
            # Kalman filter update
            if track_id in self.tracks:
                kalman = self.tracks[track_id].get('kalman')
                if kalman:
                    kalman.update(bbox)
                else:
                    # İlk kez görüldü, Kalman oluştur
                    kalman = KalmanTracker(bbox)
                    self.tracks[track_id]['kalman'] = kalman
            else:
                kalman = KalmanTracker(bbox)
            
            self.tracks[track_id] = {
                'bbox': bbox,
                'confidence': det.get('confidence', 1.0),
                'last_frame': frame_number,
                'kalman': kalman
            }
            self.disappeared[track_id] = 0
            self.track_lengths[track_id] = self.track_lengths.get(track_id, 0) + 1
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            tracked_players.append(TrackedPlayer(
                track_id=track_id,
                bbox=tuple(bbox),
                confidence=det.get('confidence', 1.0),
                frame_number=frame_number,
                position=(center_x, center_y)
            ))
        
        # Yeni track'ler oluştur (eşleşmeyen detection'lar)
        for j, det in enumerate(detections):
            if j not in matched_dets:
                track_id = self.next_id
                self.next_id += 1
                bbox = det['bbox']
                kalman = KalmanTracker(bbox)
                
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'confidence': det.get('confidence', 1.0),
                    'last_frame': frame_number,
                    'kalman': kalman
                }
                self.disappeared[track_id] = 0
                self.track_lengths[track_id] = 1
                
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                tracked_players.append(TrackedPlayer(
                    track_id=track_id,
                    bbox=tuple(bbox),
                    confidence=det.get('confidence', 1.0),
                    frame_number=frame_number,
                    position=(center_x, center_y)
                ))
        
        # Eşleşmeyen track'leri işaretle
        for i, track_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    # Sadece minimum uzunluğa sahip track'leri sil
                    if self.track_lengths.get(track_id, 0) >= self.min_track_length:
                        # Uzun track'leri silme, sadece kısa olanları sil
                        pass
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
                    if track_id in self.track_lengths:
                        del self.track_lengths[track_id]
        
        # Tüm track'leri döndür (filtreleme sadece silme işleminde yapılıyor)
        return tracked_players


class PlayerTracker:
    """Geliştirilmiş player tracker"""
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_disappeared: int = 15,
        frame_rate: float = 30.0
    ):
        """
        Args:
            iou_threshold: IoU threshold (daha düşük = daha esnek)
            max_disappeared: Maksimum kaybolma frame sayısı
            frame_rate: Video frame rate
        """
        self.tracker = ImprovedTracker(
            iou_threshold=iou_threshold,
            max_disappeared=max_disappeared
        )
        self.track_history: Dict[int, List[TrackedPlayer]] = defaultdict(list)
        self.player_info: Dict[int, Dict] = {}
        logger.info("PlayerTracker başlatıldı (ImprovedTracker)")
    
    def update(
        self,
        detections: List[Dict],
        frame: np.ndarray,
        frame_number: int
    ) -> List[TrackedPlayer]:
        """Detection'lar ile tracker'ı güncelle"""
        tracked_players = self.tracker.update(detections, frame_number)
        
        # Geçmişe ekle
        for player in tracked_players:
            self.track_history[player.track_id].append(player)
        
        logger.debug(f"Frame {frame_number}: {len(tracked_players)} oyuncu takip ediliyor")
        return tracked_players
    
    def assign_jersey_number(self, track_id: int, jersey_number: int) -> None:
        """Jersey numarası ata"""
        if track_id not in self.player_info:
            self.player_info[track_id] = {}
        self.player_info[track_id]['jersey_number'] = jersey_number
        
        for player in self.track_history[track_id]:
            player.jersey_number = jersey_number
        
        logger.info(f"Track ID {track_id} -> Jersey #{jersey_number}")
    
    def get_player_trajectory(self, track_id: int) -> List[TrackedPlayer]:
        """Oyuncu trajectory'sini al"""
        return self.track_history.get(track_id, [])
    
    def get_active_tracks(self) -> List[int]:
        """Aktif track ID'lerini al"""
        return list(self.track_history.keys())
    
    def reset(self) -> None:
        """Tracker'ı sıfırla"""
        self.track_history.clear()
        self.player_info.clear()
        self.tracker = ImprovedTracker(
            iou_threshold=self.tracker.iou_threshold,
            max_disappeared=self.tracker.max_disappeared
        )
        logger.info("Tracker sıfırlandı")


