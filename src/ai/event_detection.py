"""
Basketbol olay tespiti - ML Model için veri yapıları
Manuel etiketleme ile veri toplama, sonra ML model eğitimi
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BasketballEvent:
    """Basketbol olayı - ML model için veri yapısı"""
    event_type: str  # "basket", "pas"
    track_id: int  # İlgili oyuncu track_id
    frame_start: int
    frame_end: int
    confidence: float
    position: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict] = None
