# 🎬 Video Kırpma ve Paylaşım Arayüzü - Plan

## 📋 Proje Özeti

Basketbol videosu yükleyip, ML model ile basket ve pas anlarını tespit eden, bu anları kırparak ayrı videolar oluşturan ve YouTube/Instagram'a paylaşan web arayüzü.

---

## 🎯 Özellikler

### 1. Video Yükleme ve Analiz
- ✅ Video yükleme (drag & drop veya file picker)
- ✅ Video önizleme
- ✅ Model ile otomatik olay tespiti (basket, pas)
- ✅ Analiz ilerlemesi gösterimi
- ✅ Tespit edilen olayların listesi

### 2. Video Kırpma
- ✅ Basket anlarını otomatik kırpma
- ✅ Pas anlarını otomatik kırpma
- ✅ Ön izleme (thumbnail) oluşturma
- ✅ Kırpılmış videoları indirme

### 3. Paylaşım Özellikleri
- ✅ YouTube'a video yükleme
- ✅ Instagram'a video paylaşımı
- ✅ Başlık ve açıklama ekleme
- ✅ Thumbnail seçimi

---

## 🏗️ Mimari

### Backend
- **Framework:** Flask (hafif ve hızlı)
- **Video İşleme:** OpenCV, moviepy
- **ML Model:** Eğitilmiş EventClassifier
- **API:** REST API endpoints

### Frontend
- **HTML/CSS/JavaScript:** Modern, responsive
- **Video Player:** HTML5 video player
- **File Upload:** Drag & drop
- **UI Framework:** Bootstrap veya Tailwind CSS

### Dosya Yapısı
```
web/
├── app.py                 # Flask uygulaması
├── templates/
│   ├── index.html        # Ana sayfa
│   ├── results.html      # Sonuçlar sayfası
│   └── upload.html       # Yükleme sayfası
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/          # Yüklenen videolar
└── clips/                # Kırpılmış videolar
```

---

## 📊 İş Akışı

```
1. Kullanıcı video yükler
   ↓
2. Video backend'e kaydedilir
   ↓
3. Video analizi başlatılır (background job)
   ↓
4. Model ile olay tespiti yapılır
   ↓
5. Tespit edilen olaylar listelenir
   ↓
6. Kullanıcı hangi olayları kırpmak istediğini seçer
   ↓
7. Video kırpma işlemi başlatılır
   ↓
8. Kırpılmış videolar oluşturulur
   ↓
9. Kullanıcı videoları önizler
   ↓
10. YouTube veya Instagram'a paylaşır
```

---

## 🔧 Teknik Detaylar

### 1. Video Analizi
```python
# analyze_video_with_model.py
def analyze_video_with_model(video_path):
    # Video analizi yap (SAM3 + tracking)
    analysis = analyze_video_final(video_path)
    
    # Feature extraction
    features = extract_features_for_all_frames(analysis)
    
    # Model ile tahmin
    events = []
    for feature in features:
        event_type, confidence = model.predict(feature)
        if confidence > 0.7:  # Threshold
            events.append({
                'type': event_type,
                'start_time': feature['start_time'],
                'end_time': feature['end_time'],
                'confidence': confidence
            })
    
    return events
```

### 2. Video Kırpma
```python
# video_clipper.py
from moviepy.editor import VideoFileClip

def clip_event(video_path, start_time, end_time, output_path):
    clip = VideoFileClip(video_path)
    event_clip = clip.subclip(start_time, end_time)
    
    # Kısa bir buffer ekle (0.5 saniye öncesi/sonrası)
    event_clip = clip.subclip(
        max(0, start_time - 0.5),
        min(clip.duration, end_time + 0.5)
    )
    
    event_clip.write_videofile(output_path)
    return output_path
```

### 3. YouTube API
```python
# youtube_uploader.py
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_youtube(video_path, title, description, credentials):
    youtube = build('youtube', 'v3', credentials=credentials)
    
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'categoryId': '17'  # Sports
        },
        'status': {
            'privacyStatus': 'public'
        }
    }
    
    media = MediaFileUpload(video_path)
    response = youtube.videos().insert(
        part='snippet,status',
        body=body,
        media_body=media
    ).execute()
    
    return response['id']
```

### 4. Instagram API
```python
# instagram_uploader.py
from instagrapi import Client

def upload_to_instagram(video_path, caption, credentials):
    cl = Client()
    cl.login(credentials['username'], credentials['password'])
    
    # Instagram Reels için video yükle
    cl.clip_upload(
        video_path,
        caption=caption
    )
```

---

## 📁 Gerekli Kütüphaneler

```txt
flask==3.0.0
flask-cors==4.0.0
moviepy==1.0.3
opencv-python==4.8.1
google-api-python-client==2.100.0
google-auth-httplib2==0.1.1
google-auth-oauthlib==1.1.0
instagrapi==2.0.0
```

---

## 🚀 Geliştirme Adımları

### Faz 1: Temel Arayüz (1-2 gün)
- [ ] Flask uygulaması kurulumu
- [ ] Video yükleme sayfası
- [ ] Video önizleme

### Faz 2: Model Entegrasyonu (1-2 gün)
- [ ] Model yükleme
- [ ] Video analizi endpoint'i
- [ ] Olay tespiti sonuçları gösterimi

### Faz 3: Video Kırpma (1 gün)
- [ ] Video kırpma fonksiyonu
- [ ] Kırpılmış videoları listeleme
- [ ] Video önizleme ve indirme

### Faz 4: Paylaşım (2-3 gün)
- [ ] YouTube API entegrasyonu
- [ ] Instagram API entegrasyonu
- [ ] Paylaşım arayüzü

---

## 🔐 Güvenlik ve Ayarlar

### Environment Variables
```env
FLASK_ENV=development
SECRET_KEY=your-secret-key
YOUTUBE_CLIENT_ID=your-client-id
YOUTUBE_CLIENT_SECRET=your-client-secret
INSTAGRAM_USERNAME=your-username
INSTAGRAM_PASSWORD=your-password
MODEL_PATH=data/models/event_classifier.pkl
```

### Dosya Yönetimi
- Yüklenen videolar: `web/static/uploads/` (temizlenebilir)
- Kırpılmış videolar: `web/clips/` (temizlenebilir)
- Geçici dosyalar: Otomatik temizleme

---

## 🎨 UI Tasarım Önerileri

1. **Ana Sayfa**
   - Büyük "Video Yükle" butonu
   - Drag & drop alanı
   - Son işlenen videolar listesi

2. **Analiz Sayfası**
   - İlerleme çubuğu
   - Tespit edilen olaylar listesi (thumbnail + bilgi)
   - "Kırp" butonu

3. **Sonuçlar Sayfası**
   - Kırpılmış videolar grid görünümü
   - Her video için:
     - Önizleme
     - Olay tipi (basket/pas)
     - Süre
     - İndir/YouTube/Instagram butonları

---

## 📝 Notlar

- Video işleme zaman alabilir → Background jobs kullan (Celery veya threading)
- Büyük videolar için → Chunked upload
- YouTube API → OAuth 2.0 gerekli
- Instagram API → Instagram Business API veya instagrapi (unofficial)

---

## 🎯 Başlangıç

İlk adım: Flask uygulaması ve temel video yükleme arayüzü oluşturalım!



