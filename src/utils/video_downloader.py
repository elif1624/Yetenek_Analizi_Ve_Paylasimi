"""YouTube video indirme modülü"""

import logging
from pathlib import Path
from typing import Optional
import subprocess

logger = logging.getLogger(__name__)


def download_youtube_video(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None
) -> Path:
    """
    YouTube videosunu indir
    
    Args:
        url: YouTube video URL'si
        output_dir: İndirme dizini
        filename: Dosya adı (None ise video başlığı kullanılır)
        
    Returns:
        İndirilen video dosyasının yolu
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp yüklü değil! pip install yt-dlp yapın."
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output template
    if filename:
        output_template = str(output_dir / f"{filename}.%(ext)s")
    else:
        output_template = str(output_dir / "%(title)s.%(ext)s")
    
    # yt-dlp options
    # Sadece tek format indir (merge gerektirmesin)
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # MP4 formatını tercih et, yoksa en iyi formatı al
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': False,
    }
    
    logger.info(f"YouTube videosu indiriliyor: {url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Video bilgilerini al
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'video')
            logger.info(f"Video başlığı: {video_title}")
            
            # İndir
            ydl.download([url])
            
            # İndirilen dosyayı bul
            # yt-dlp genellikle .mp4 uzantısı kullanır
            ext = info.get('ext', 'mp4')
            if filename:
                downloaded_file = output_dir / f"{filename}.{ext}"
            else:
                # Dosya adını güvenli hale getir
                safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                downloaded_file = output_dir / f"{safe_title}.{ext}"
            
            # Dosya yoksa, tüm .mp4 dosyalarını kontrol et
            if not downloaded_file.exists():
                mp4_files = list(output_dir.glob("*.mp4"))
                if mp4_files:
                    downloaded_file = mp4_files[0]
                    logger.info(f"Dosya bulundu: {downloaded_file}")
                else:
                    raise FileNotFoundError("İndirilen video dosyası bulunamadı")
            
            logger.info(f"Video başarıyla indirildi: {downloaded_file}")
            return downloaded_file
            
    except Exception as e:
        logger.error(f"YouTube video indirme hatası: {e}")
        raise


def get_video_info(url: str) -> dict:
    """
    YouTube video bilgilerini al (indirmeden)
    
    Args:
        url: YouTube video URL'si
        
    Returns:
        Video bilgileri dict'i
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp yüklü değil!")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'view_count': info.get('view_count'),
                'uploader': info.get('uploader'),
            }
    except Exception as e:
        logger.error(f"Video bilgisi alma hatası: {e}")
        raise

