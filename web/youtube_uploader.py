"""
YouTube Video Uploader
YouTube Data API v3 kullanarak video yükleme
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict
import json

logger = logging.getLogger(__name__)

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    logger.warning("Google API kütüphaneleri yüklü değil. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")


# YouTube API scope'ları
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# Video kategorileri (sports = 17)
YOUTUBE_CATEGORY_ID = 17  # Sports


def get_authenticated_service(client_secrets_file: Path, credentials_file: Path) -> Optional[object]:
    """
    YouTube API için authenticated service döndür
    
    Args:
        client_secrets_file: Google Cloud Console'dan indirilen client_secrets.json dosyası
        credentials_file: OAuth token'ların kaydedildiği dosya
    
    Returns:
        YouTube API service object veya None
    """
    if not GOOGLE_APIS_AVAILABLE:
        logger.error("Google API kütüphaneleri yüklü değil")
        return None
    
    credentials = None
    
    # Kaydedilmiş credentials varsa yükle
    if credentials_file.exists():
        try:
            credentials = Credentials.from_authorized_user_file(
                str(credentials_file), SCOPES
            )
        except Exception as e:
            logger.warning(f"Credentials yüklenemedi: {e}")
    
    # Credentials yoksa veya geçersizse OAuth flow başlat
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception as e:
                logger.error(f"Token refresh hatası: {e}")
                credentials = None
        
        if not credentials:
            if not client_secrets_file.exists():
                logger.error(f"Client secrets dosyası bulunamadı: {client_secrets_file}")
                logger.error("Google Cloud Console'dan OAuth 2.0 credentials indirmeniz gerekiyor")
                return None
            
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(client_secrets_file), SCOPES
                )
                credentials = flow.run_local_server(port=0)
                
                # Credentials'ı kaydet
                credentials_file.parent.mkdir(parents=True, exist_ok=True)
                with open(credentials_file, 'w') as token:
                    token.write(credentials.to_json())
            except Exception as e:
                logger.error(f"OAuth flow hatası: {e}")
                return None
    
    try:
        service = build('youtube', 'v3', credentials=credentials)
        return service
    except Exception as e:
        logger.error(f"YouTube service oluşturulamadı: {e}")
        return None


def upload_video_to_youtube(
    video_path: Path,
    title: str,
    description: str = "",
    tags: Optional[list] = None,
    privacy_status: str = "unlisted",  # "private", "public", "unlisted"
    client_secrets_file: Optional[Path] = None,
    credentials_file: Optional[Path] = None
) -> Dict[str, any]:
    """
    Video'yu YouTube'a yükle
    
    Args:
        video_path: Yüklenecek video dosyası yolu
        title: Video başlığı
        description: Video açıklaması
        tags: Video etiketleri
        privacy_status: Video gizlilik ayarı ("private", "public", "unlisted")
        client_secrets_file: Google Cloud Console'dan indirilen client_secrets.json
        credentials_file: OAuth token'ların kaydedildiği dosya
    
    Returns:
        {
            'success': bool,
            'video_id': str (eğer başarılıysa),
            'video_url': str (eğer başarılıysa),
            'error': str (eğer hata varsa)
        }
    """
    if not GOOGLE_APIS_AVAILABLE:
        return {
            'success': False,
            'error': 'Google API kütüphaneleri yüklü değil. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib'
        }
    
    if not video_path.exists():
        return {
            'success': False,
            'error': f'Video dosyası bulunamadı: {video_path}'
        }
    
    # Default dosya yolları
    if client_secrets_file is None:
        client_secrets_file = Path(__file__).parent.parent / 'config' / 'client_secrets.json'
    
    if credentials_file is None:
        credentials_file = Path(__file__).parent.parent / 'config' / 'youtube_credentials.json'
    
    try:
        # Authenticated service al
        service = get_authenticated_service(client_secrets_file, credentials_file)
        
        if not service:
            return {
                'success': False,
                'error': 'YouTube authentication başarısız. Client secrets dosyasını kontrol edin.'
            }
        
        # Video metadata
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags or [],
                'categoryId': YOUTUBE_CATEGORY_ID
            },
            'status': {
                'privacyStatus': privacy_status
            }
        }
        
        # Media upload
        media = MediaFileUpload(
            str(video_path),
            chunksize=-1,
            resumable=True,
            mimetype='video/mp4'
        )
        
        logger.info(f"YouTube'a video yükleniyor: {video_path}")
        logger.info(f"Başlık: {title}")
        
        # Upload request
        insert_request = service.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )
        
        # Resumable upload
        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                logger.info(f"Yükleme ilerlemesi: {progress}%")
        
        video_id = response['id']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        logger.info(f"Video başarıyla yüklendi: {video_url}")
        
        return {
            'success': True,
            'video_id': video_id,
            'video_url': video_url
        }
        
    except HttpError as e:
        error_details = e.error_details[0] if e.error_details else {}
        reason = error_details.get('reason', 'unknown')
        
        if reason == 'youtubeSignupRequired':
            error_msg = (
                "YouTube hesabı bulunamadı. "
                "Lütfen Google hesabınızda YouTube hesabı oluşturun: "
                "https://www.youtube.com/signup"
            )
        elif reason == 'accessNotConfigured':
            error_msg = (
                "YouTube Data API v3 etkinleştirilmemiş. "
                "Google Cloud Console'da YouTube Data API v3'ü etkinleştirin: "
                "https://console.cloud.google.com/apis/library/youtube.googleapis.com"
            )
        else:
            error_msg = f"YouTube API hatası: {e}"
            if error_details:
                error_msg += f"\nDetaylar: {error_details.get('message', 'Bilinmeyen hata')}"
        
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    except Exception as e:
        error_msg = f"Video yükleme hatası: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            'success': False,
            'error': error_msg
        }

