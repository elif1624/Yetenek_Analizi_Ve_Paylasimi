"""
Facebook Video Uploader
Facebook Graph API kullanarak video paylaşımı
"""

import logging
import os
import requests
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Facebook Page Access Token (önerilen - daha kolay izin alınır)
# Page Access Token kullanmak için bu değeri doldurun
PAGE_ACCESS_TOKEN = "EAARdULcyRssBQfNoJbuSlrQZA0bYNgdEAjBByHN9FNkZABdtgQJbVwtZATVzUILQO7kBEY6lTN5n1EFc3D0Go0tPwwYIUi1VDbwXA249dhUVtLJfDNMRsAlrf2m9eiDznxWz2ApH0IfozQZAGQyjLWokJfe5MxY2zHGHaWl4RXZCM3LrMQH9XV7kz4yp4o7hOWHm5gBVElYQdPlI7ZBi0Ln8MacmNj39xnD3mCmD4ZD"
# Page ID - Page Access Token kullanıldığında 'me' olarak ayarlanır (token sayfaya ait olduğu için 'me' o sayfayı temsil eder)
PAGE_ID = 'me'


def check_token_permissions(access_token: str) -> Dict[str, any]:
    """
    Access token'ın izinlerini kontrol et
    
    Returns:
        {
            'success': bool,
            'permissions': list,
            'error': str (eğer hata varsa)
        }
    """
    try:
        url = "https://graph.facebook.com/v18.0/me/permissions"
        params = {'access_token': access_token}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            permissions = [p['permission'] for p in data.get('data', []) if p.get('status') == 'granted']
            return {
                'success': True,
                'permissions': permissions
            }
        else:
            return {
                'success': False,
                'error': f"Token izinleri alınamadı: {response.status_code}"
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def upload_video_to_facebook(
    video_path: Path,
    description: str = "",
    access_token: Optional[str] = None,
    page_id: Optional[str] = None
) -> Dict[str, any]:
    """
    Video'yu Facebook'a yükle (Facebook Graph API)
    
    Args:
        video_path: Yüklenecek video dosyası yolu
        description: Video açıklaması
        access_token: Facebook Graph API access token (opsiyonel, varsayılan token kullanılır)
        page_id: Facebook Page ID (opsiyonel, belirtilmezse kullanıcının kendi profilinde paylaşılır)
    
    Returns:
        {
            'success': bool,
            'video_id': str (eğer başarılıysa),
            'video_url': str (eğer başarılıysa),
            'error': str (eğer hata varsa)
        }
    """
    if not video_path.exists():
        return {
            'success': False,
            'error': f'Video dosyası bulunamadı: {video_path}'
        }
    
    # Access token kontrolü - Önce Page Access Token, sonra User Token
    if not access_token:
        # Page Access Token varsa onu kullan (daha kolay izin alınır)
        if PAGE_ACCESS_TOKEN:
            access_token = PAGE_ACCESS_TOKEN
            # Page Access Token kullanıldığında page_id'yi 'me' yap
            if not page_id:
                page_id = PAGE_ID if PAGE_ID != 'me' else 'me'
            logger.info("Page Access Token kullanılıyor")
        else:
            # User Token kullan (sadece environment variable'dan)
            access_token = os.getenv('FACEBOOK_ACCESS_TOKEN', None)
            if not access_token:
                return {
                    'success': False,
                    'error': 'Facebook access token bulunamadı. PAGE_ACCESS_TOKEN veya FACEBOOK_ACCESS_TOKEN environment variable ayarlayın.'
                }
            logger.info("User Access Token kullanılıyor (environment variable'dan)")
    
    if not access_token:
        return {
            'success': False,
            'error': 'Facebook access token bulunamadı. PAGE_ACCESS_TOKEN veya FACEBOOK_ACCESS_TOKEN ayarlayın.'
        }
    
    # Token izinlerini kontrol et (debug için)
    try:
        perm_check = check_token_permissions(access_token)
        if perm_check.get('success'):
            permissions = perm_check.get('permissions', [])
            logger.info(f"Token permissions: {permissions}")
            if 'publish_video' not in permissions and 'pages_manage_posts' not in permissions:
                logger.warning("Token'da 'publish_video' veya 'pages_manage_posts' permission'ı yok!")
    except Exception as e:
        logger.warning(f"Token permission kontrolü yapılamadı: {e}")
    
    try:
        # Video dosya boyutu kontrolü (Facebook limiti genellikle 4GB, ama pratikte 100MB yeterli)
        file_size = video_path.stat().st_size / (1024 * 1024)  # MB
        if file_size > 100:
            logger.warning(f"Video dosyası büyük ({file_size:.1f}MB). Yükleme uzun sürebilir.")
        
        # Facebook Graph API endpoint'i
        # Page Access Token kullanıldığında page_id 'me' olur ve /me/videos endpoint'i kullanılır
        # User Token kullanıldığında da /me/videos kullanılır ama 'publish_video' permission gerekiyor
        if page_id and page_id != 'me':
            # Belirli bir Page ID ile yükle
            upload_url = f"https://graph.facebook.com/v18.0/{page_id}/videos"
        else:
            # /me/videos endpoint'i kullan (Page Access Token veya User Token ile)
            upload_url = "https://graph.facebook.com/v18.0/me/videos"
        
        # Video dosyasını yükle
        with open(video_path, 'rb') as video_file:
            files = {
                'source': video_file
            }
            data = {
                'access_token': access_token,
                'description': description
            }
            
            logger.info(f"Facebook'a video yükleniyor: {video_path}")
            response = requests.post(upload_url, files=files, data=data, timeout=300)  # 5 dakika timeout
        
        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            error_message = error_data.get('error', {}).get('message', 'Bilinmeyen hata')
            error_code = error_data.get('error', {}).get('code', '')
            error_subcode = error_data.get('error', {}).get('error_subcode', '')
            
            logger.error(f"Facebook video yükleme hatası ({response.status_code}): {error_message}")
            
            # Hata 190 ise (expired token), daha açıklayıcı mesaj
            if error_code == 190 or 'expired' in error_message.lower() or 'Session has expired' in error_message:
                detailed_error = (
                    "Facebook access token'ının süresi dolmuş. Yeni bir token almanız gerekiyor.\n\n"
                    "Çözüm:\n"
                    "1. Facebook Graph API Explorer'da (https://developers.facebook.com/tools/explorer/) "
                    "uygulamanızı seçin\n"
                    "2. Page Access Token almak için: https://developers.facebook.com/tools/accesstoken/\n"
                    "3. Sayfanızı seçin ve yeni token'ı kopyalayın\n"
                    "4. Token'ı web/facebook_uploader.py dosyasındaki PAGE_ACCESS_TOKEN değişkenine ekleyin\n\n"
                    "Not: Page Access Token'lar genellikle daha uzun süre geçerlidir."
                )
                return {
                    'success': False,
                    'error': detailed_error
                }
            
            # Hata 100 ise (permission hatası), daha açıklayıcı mesaj
            if error_code == 100:
                detailed_error = (
                    "Video yayınlama izni yok. Access token'ınızın 'publish_video' veya 'pages_manage_posts' "
                    "izinlerine sahip olması gerekiyor.\n\n"
                    "Çözüm:\n"
                    "1. Facebook Graph API Explorer'da (https://developers.facebook.com/tools/explorer/) "
                    "uygulamanızı seçin\n"
                    "2. 'publish_video' veya 'pages_manage_posts' permission'larını ekleyin\n"
                    "3. Yeni bir access token oluşturun\n"
                    "4. Token'ı web/facebook_uploader.py dosyasındaki PAGE_ACCESS_TOKEN değişkenine ekleyin"
                )
                return {
                    'success': False,
                    'error': detailed_error
                }
            
            return {
                'success': False,
                'error': f"Facebook API hatası: {error_message} (Code: {error_code})"
            }
        
        result_data = response.json()
        video_id = result_data.get('id')
        
        if not video_id:
            return {
                'success': False,
                'error': 'Facebook video ID alınamadı'
            }
        
        # Video URL'ini oluştur
        if page_id:
            video_url = f"https://www.facebook.com/{page_id}/videos/{video_id}/"
        else:
            video_url = f"https://www.facebook.com/{video_id}"
        
        logger.info(f"Video başarıyla Facebook'a yüklendi: {video_id}")
        
        return {
            'success': True,
            'video_id': video_id,
            'video_url': video_url
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Facebook API istek hatası: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    except Exception as e:
        error_msg = f"Facebook video yükleme hatası: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            'success': False,
            'error': error_msg
        }

