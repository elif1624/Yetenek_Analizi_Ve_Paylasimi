"""Yardımcı fonksiyonlar"""

import logging
from pathlib import Path
from typing import Optional

# Logging yapılandırması
def setup_logging(log_level: str = "INFO") -> None:
    """Logging yapılandırması"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def ensure_dir(directory: Path) -> Path:
    """Dizini oluştur (yoksa)"""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_video_files(directory: Path, extensions: Optional[list] = None) -> list:
    """
    Dizindeki video dosyalarını listele
    
    Args:
        directory: Dizin yolu
        extensions: Desteklenen uzantılar (varsayılan: ['.mp4', '.avi', '.mov'])
        
    Returns:
        Video dosya yolları listesi
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    
    video_files = []
    for ext in extensions:
        video_files.extend(directory.glob(f'*{ext}'))
        video_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(video_files)


