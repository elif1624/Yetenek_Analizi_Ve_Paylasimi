# 🎬 Sosyal Medya API Kurulum Rehberi

YouTube ve Instagram API entegrasyonları için kurulum adımları.

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

## 📸 Instagram API Kurulumu

### 1. Facebook Developer Console Setup

**NOT:** Instagram Graph API için **Business** veya **Creator** hesabı gerekiyor.

1. [Facebook Developers](https://developers.facebook.com/) üzerinden yeni bir uygulama oluşturun
2. **Instagram Graph API** ürününü ekleyin
3. **Instagram Basic Display** veya **Instagram Graph API** seçin

### 2. Business Hesabı için (Önerilen)

1. Bir **Facebook Page** oluşturun
2. Instagram hesabınızı bu Page'e bağlayın (Instagram > Settings > Account > Linked Accounts)
3. **Instagram Graph API** > **User Token Generator** ile access token oluşturun

### 3. Environment Variables

`.env` dosyasına ekleyin:

```env
INSTAGRAM_ACCESS_TOKEN=your_instagram_access_token
INSTAGRAM_PAGE_ID=your_facebook_page_id
```

veya `app.py` içinde doğrudan ayarlayabilirsiniz.

### 4. Video Formatı Gereksinimleri

- Maksimum dosya boyutu: **100MB**
- Format: MP4, MOV
- Süre: 3 saniye - 60 saniye (Reels için)
- En boy oranı: 9:16 (Reels için) veya 1:1 (normal video)

## 🔧 Klasör Yapısı

```
project_root/
├── config/
│   ├── client_secrets.json      # Google OAuth credentials (YouTube için)
│   └── youtube_credentials.json  # OAuth tokens (otomatik oluşturulur)
└── web/
    ├── youtube_uploader.py
    └── instagram_uploader.py
```

## ⚠️ Önemli Notlar

### YouTube
- İlk çalıştırmada tarayıcı açılacak ve authorization gerekli
- Video **unlisted** olarak yüklenir (değiştirilebilir)
- Video kategorisi: **Sports (17)**

### Instagram
- Business veya Creator hesabı gerekiyor
- Facebook Page'e bağlı olmalı
- Video boyutu limiti: 100MB
- Reels için 9:16 en-boy oranı önerilir

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

### Instagram

```python
from web.instagram_uploader import upload_video_to_instagram

result = upload_video_to_instagram(
    video_path=Path("clips/video.mp4"),
    caption="🏀 Basket anı - Basketbol highlights #basketbol"
)

if result['success']:
    print(f"Video paylaşıldı: {result['media_id']}")
```

