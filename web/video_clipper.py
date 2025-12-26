"""
Video kırpma (clipping) modülü - OpenCV kullanarak
"""

import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def clip_video(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    buffer: float = 0.5
) -> Path:
    """
    Video'dan belirli bir zaman aralığını kırp (OpenCV kullanarak)
    
    Args:
        video_path: Kaynak video dosyası
        start_time: Başlangıç zamanı (saniye)
        end_time: Bitiş zamanı (saniye)
        output_path: Çıktı dosyası yolu
        buffer: Öncesi/sonrası buffer süresi (saniye)
    
    Returns:
        Kırpılmış video dosyası yolu
    """
    try:
        logger.info(f"Video kırpılıyor: {video_path}")
        logger.info(f"Zaman aralığı: {start_time}s - {end_time}s")
        
        # Video aç
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Video özellikleri
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video özellikleri: {width}x{height}, {fps:.2f} FPS, {video_duration:.2f} saniye")
        
        # Buffer ekle
        start_with_buffer = max(0, start_time - buffer)
        end_with_buffer = min(video_duration, end_time + buffer)
        
        # Frame numaralarını hesapla
        start_frame = int(start_with_buffer * fps)
        end_frame = int(end_with_buffer * fps)
        
        logger.info(f"Kırpılacak frame aralığı: {start_frame} - {end_frame} (toplam {end_frame - start_frame} frame)")
        
        # Klasörü oluştur
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer ayarla
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Video writer açılamadı: {output_path}. Codec: mp4v")
        
        # İlgili frame'lere git
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        written_frames = 0
        
        logger.info(f"Video yazılıyor...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            current_frame = start_frame + frame_count
            
            # İstenen aralığın dışındaysa dur
            if current_frame >= end_frame:
                break
            
            # Frame'i yaz
            out.write(frame)
            written_frames += 1
            frame_count += 1
            
            # İlerleme göster (her 30 frame'de bir)
            if written_frames % 30 == 0:
                logger.debug(f"Yazılan frame: {written_frames}/{end_frame - start_frame}")
        
        # Temizle
        cap.release()
        out.release()
        
        logger.info(f"Video başarıyla kırpıldı: {output_path} ({written_frames} frame)")
        return output_path
        
    except Exception as e:
        logger.error(f"Video kırpma hatası: {e}", exc_info=True)
        raise


def create_clip_filename(
    base_filename: str,
    event_type: str,
    start_time: float,
    end_time: float
) -> str:
    """Kırpılmış video dosya adı oluştur"""
    timestamp_start = int(start_time)
    timestamp_end = int(end_time)
    extension = Path(base_filename).suffix
    return f"{Path(base_filename).stem}_{event_type}_{timestamp_start}_{timestamp_end}{extension}"



