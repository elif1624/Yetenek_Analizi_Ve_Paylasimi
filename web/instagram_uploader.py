"""
Instagram Video Uploader
Instagram Graph API kullanarak video paylaşımı
"""

import logging
import os
import requests
from pathlib import Path
from typing import Optional, Dict
import json

logger = logging.getLogger(__name__)


def upload_video_to_instagram(
    video_path: Path,
    caption: str = "",
    access_token: Optional[str] = None,
    page_id: Optional[str] = None
) -> Dict[str, any]:
    """
    Video'yu Instagram'a paylaş (Instagram Graph API)
    
    NOT: Instagram Graph API için Business veya Creator hesabı gerekiyor
    ve Facebook Page'e bağlı olmalı.
    
    Args:
        video_path: Yüklenecek video dosyası yolu
        caption: Video açıklaması
        access_token: Instagram Graph API access token
        page_id: Facebook Page ID (Instagram Business hesabı için)
    
    Returns:
        {
            'success': bool,
            'media_id': str (eğer başarılıysa),
            'error': str (eğer hata varsa)
        }
    """
    if not video_path.exists():
        return {
            'success': False,
            'error': f'Video dosyası bulunamadı: {video_path}'
        }
    
    # Access token kontrolü
    if not access_token:
        access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    
    if not access_token:
        return {
            'success': False,
            'error': 'Instagram access token bulunamadı. Environment variable INSTAGRAM_ACCESS_TOKEN ayarlayın veya access_token parametresini verin.'
        }
    
    # Page ID kontrolü (Business hesabı için)
    if not page_id:
        page_id = os.getenv('INSTAGRAM_PAGE_ID')
    
    try:
        # Video dosya boyutu kontrolü (Instagram limiti: 100MB)
        file_size = video_path.stat().st_size / (1024 * 1024)  # MB
        if file_size > 100:
            return {
                'success': False,
                'error': f'Video dosyası çok büyük ({file_size:.1f}MB). Instagram limiti 100MB.'
            }
        
        # Instagram Graph API - 2 aşamalı upload
        # 1. Aşama: Video container oluştur
        
        if page_id:
            # Business hesabı için
            container_url = f"https://graph.facebook.com/v18.0/{page_id}/media"
        else:
            # Creator hesabı için (user ID gerekir)
            return {
                'success': False,
                'error': 'Instagram upload için page_id (Facebook Page ID) gerekli. Business veya Creator hesabı kullanın.'
            }
        
        # Video container oluştur
        with open(video_path, 'rb') as video_file:
            container_data = {
                'media_type': 'REELS',  # veya 'VIDEO'
                'caption': caption,
                'access_token': access_token
            }
            
            container_response = requests.post(
                container_url,
                data=container_data,
                files={'video': video_file}
            )
        
        if container_response.status_code != 200:
            error_data = container_response.json()
            return {
                'success': False,
                'error': f"Instagram container oluşturma hatası: {error_data.get('error', {}).get('message', 'Bilinmeyen hata')}"
            }
        
        container_data = container_response.json()
        creation_id = container_data.get('id')
        
        if not creation_id:
            return {
                'success': False,
                'error': 'Instagram container ID alınamadı'
            }
        
        logger.info(f"Instagram container oluşturuldu: {creation_id}")
        
        # 2. Aşama: Container'ı publish et
        publish_url = f"https://graph.facebook.com/v18.0/{page_id}/media_publish"
        publish_data = {
            'creation_id': creation_id,
            'access_token': access_token
        }
        
        publish_response = requests.post(publish_url, data=publish_data)
        
        if publish_response.status_code != 200:
            error_data = publish_response.json()
            return {
                'success': False,
                'error': f"Instagram publish hatası: {error_data.get('error', {}).get('message', 'Bilinmeyen hata')}"
            }
        
        publish_data = publish_response.json()
        media_id = publish_data.get('id')
        
        logger.info(f"Video başarıyla Instagram'a yüklendi: {media_id}")
        
        return {
            'success': True,
            'media_id': media_id
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Instagram API istek hatası: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    except Exception as e:
        error_msg = f"Instagram video yükleme hatası: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            'success': False,
            'error': error_msg
        }


def get_instagram_auth_url(client_id: str, redirect_uri: str) -> str:
    """
    Instagram OAuth authorization URL oluştur
    
    Args:
        client_id: Instagram App ID
        redirect_uri: OAuth redirect URI
    
    Returns:
        Authorization URL
    """
    auth_url = (
        f"https://api.instagram.com/oauth/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope=user_profile,user_media"
        f"&response_type=code"
    )
    return auth_url

