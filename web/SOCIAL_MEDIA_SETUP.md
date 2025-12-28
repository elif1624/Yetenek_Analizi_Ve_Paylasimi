# 🎬 Sosyal Medya API Kurulum Rehberi

YouTube ve Facebook API entegrasyonları için kurulum adımları.

## 📺 YouTube API Kurulumu

### 1. Google Cloud Console Setup

1. [Google Cloud Console](https://console.cloud.google.com/) üzerinden yeni bir proje oluşturun
2. **APIs & Services** > **Library** bölümüne gidin
3. **YouTube Data API v3**'ü arayın ve **ETKİNLEŞTİRİN** (ENABLE)
   
   **ÖNEMLİ:** API'nin etkinleştirildiğinden emin olun. Etkinleştirilmemişse "Unauthorized" hatası alırsınız.

### 2. OAuth 2.0 Credentials Oluşturma

1. **APIs & Services** > **Credentials** bölümüne gidin
2. **Create Credentials** > **OAuth client ID** seçin
3. Application type: **Desktop app** seçin
4. Name: "Basketbol Video Uploader" (veya istediğiniz bir isim)
5. **Create** butonuna tıklayın
6. İndirilen `client_secrets.json` dosyasını `config/client_secrets.json` konumuna koyun

### 3. Test Kullanıcısı Ekleme (ÖNEMLİ!)

**OAuth uygulamanız "Testing" modunda olduğu için test kullanıcısı eklemeniz gerekir:**

1. Google Cloud Console > **APIs & Services** > **OAuth consent screen**
2. **Test users** bölümünde **+ ADD USERS** butonuna tıklayın
3. Kendi email adresinizi ekleyin (örn: `your-email@gmail.com`)
4. **ADD** butonuna tıklayın

**NOT:** Test kullanıcısı eklemeden YouTube'a yüklemeye çalışırsanız "Erişim engellendi" hatası alırsınız.

**ÖNEMLİ:** Test kullanıcısı olarak eklediğiniz email'in **YouTube kanalı olmalıdır**. Kanalı olmayan bir email ile "Unauthorized" hatası alırsınız.

### 4. YouTube Hesabı Oluşturma veya Test Kullanıcısını Değiştirme

**Eğer mevcut test kullanıcısının YouTube kanalı yoksa:**

**Seçenek A: Test kullanıcısını değiştirin (Önerilen)**
1. OAuth consent screen > Test users bölümünden mevcut email'i kaldırın
2. YouTube kanalı olan bir email ekleyin
3. `config/youtube_credentials.json` dosyasını silin (eski hesap token'larını temizlemek için)
4. OAuth flow'da yeni email ile giriş yapın

**Seçenek B: Mevcut hesapta YouTube kanalı oluşturun**
1. [YouTube](https://www.youtube.com) sitesine gidin
2. Google hesabınızla giriş yapın (test kullanıcısı olarak eklediğiniz email)
3. Kanal oluşturun (ücretsiz)

### 5. İlk OAuth Authorization

İlk çalıştırmada tarayıcı açılacak ve Google hesabınızla giriş yapmanız istenecek.
Authorization sonrası `config/youtube_credentials.json` dosyası otomatik oluşturulacak.

### 6. Gerekli Python Kütüphaneleri

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## 📘 Facebook API Kurulumu

### 1. Access Token Kullanımı

Facebook Graph API için Page Access Token kullanılır. Token `web/facebook_uploader.py` dosyasında tanımlıdır.

**Önemli:** Page Access Token kullanılıyor (User Access Token yerine). Bu, videoların bir Facebook Page'e yüklenmesini sağlar ve daha uzun süreli geçerliliğe sahiptir.

**Not:** Token süresi dolduğunda yeni bir Page Access Token almanız gerekebilir.

### 2. Facebook Developer Console (ÖNEMLİ - Token İzni İçin)

**EĞER "#100 No permission to publish the video" HATASI ALIYORSANIZ:**

Bu hata, access token'ınızın video yayınlama izni (`publish_video`) olmadığını gösterir. Çözüm:

1. [Facebook Graph API Explorer](https://developers.facebook.com/tools/explorer/) sayfasına gidin
2. Sağ üst köşede uygulamanızı seçin
3. **Get Token** > **Get User Access Token** butonuna tıklayın
4. **Permissions** bölümünde şu izinleri seçin:
   - ✅ `publish_video` (Video yayınlama - ZORUNLU)
   - ✅ `pages_manage_posts` (Page'e video yükleme için - Page kullanıyorsanız)
   - ✅ `user_videos` (Kullanıcı videolarına erişim)
5. **Generate Access Token** butonuna tıklayın
6. Facebook'tan izin verin
7. Oluşturulan token'ı kopyalayın
8. Token'ı `web/facebook_uploader.py` dosyasındaki `PAGE_ACCESS_TOKEN` değişkenine ekleyin
9. Page ID'yi `PAGE_ID` değişkenine ekleyin (veya 'me' olarak bırakın)

**NOT:** `publish_video` permission'ı genellikle Facebook tarafından manuel olarak onaylanması gerekebilir. Eğer token oluştururken bu permission görünmüyorsa, Facebook Developer Console'da uygulamanızın ayarlarından bu permission'ı talep etmeniz gerekebilir.

**Önerilen Yöntem - Page Access Token:** 
Uygulama varsayılan olarak Page Access Token kullanır (User Access Token yerine):
1. [Page Access Token Tool](https://developers.facebook.com/tools/accesstoken/) sayfasına gidin
2. Page'inizi seçin
3. Token'ı kopyalayın ve `web/facebook_uploader.py` dosyasındaki `PAGE_ACCESS_TOKEN` değişkenine ekleyin
4. Page ID'yi `PAGE_ID` değişkenine ekleyin (veya 'me' olarak bırakın)

Page Access Token'ın avantajları:
- Daha uzun süreli geçerlilik
- Video yayınlama izni genellikle otomatik olarak dahildir
- Videolar direkt olarak Page'e yüklenir

### 3. Page ID (Opsiyonel)

Eğer videoları bir Facebook Page'e yüklemek istiyorsanız:

1. Facebook sayfanızın ID'sini alın
2. `upload_video_to_facebook` fonksiyonunu çağırırken `page_id` parametresini verin
3. Belirtilmezse videolar kullanıcının kendi profilinde paylaşılır

### 4. Video Formatı Gereksinimleri

- Maksimum dosya boyutu: **4GB** (pratikte 100MB'a kadar önerilir)
- Format: MP4, MOV, AVI, WMV, FLV
- Süre: En az 1 saniye
- Çözünürlük: Minimum 720p önerilir

## 🔧 Klasör Yapısı

```
project_root/
├── config/
│   ├── client_secrets.json      # Google OAuth credentials (YouTube için)
│   └── youtube_credentials.json  # OAuth tokens (otomatik oluşturulur)
└── web/
    ├── youtube_uploader.py
    └── facebook_uploader.py      # Facebook Graph API entegrasyonu
```

## ⚠️ Önemli Notlar

### YouTube
- İlk çalıştırmada tarayıcı açılacak ve authorization gerekli
- Video **unlisted** olarak yüklenir (değiştirilebilir)
- Video kategorisi: **Sports (17)**

### Facebook
- Access token gerekiyor (kodda varsayılan olarak tanımlı)
- Page ID belirtilirse videolar Page'e, belirtilmezse kullanıcı profilinde paylaşılır
- Video boyutu limiti: 4GB (pratikte 100MB'a kadar önerilir)
- Video formatları: MP4, MOV, AVI, WMV, FLV

## 🚀 Kullanım

### YouTube

```python
from web.youtube_uploader import upload_video_to_youtube

result = upload_video_to_youtube(
    video_path=Path("clips/video.mp4"),
    title="Basket Anı - Basketbol Highlights",
    description="Basketbol video analizi ile tespit edilen basket anı.",
    tags=['basketbol', 'basketball', 'highlights'],
    privacy_status='unlisted'
)

if result['success']:
    print(f"Video yüklendi: {result['video_url']}")
```

### Facebook

```python
from web.facebook_uploader import upload_video_to_facebook

result = upload_video_to_facebook(
    video_path=Path("clips/video.mp4"),
    description="🏀 Basket anı - Basketbol highlights #basketbol",
    page_id=None  # None ise kullanıcı profilinde paylaşılır
)

if result['success']:
    print(f"Video yüklendi: {result['video_url']}")
    print(f"Video ID: {result['video_id']}")
```

