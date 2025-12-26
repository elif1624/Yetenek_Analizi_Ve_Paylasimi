# 🌐 Web Arayüzü - Basketbol Video Analizi

Web tabanlı basketbol video analizi, kırpma ve paylaşım sistemi.

## 🚀 Kurulum

### 1. Gereksinimleri Yükleyin
```bash
pip install flask flask-cors opencv-python
```

### 2. Klasör Yapısı
```
web/
├── app.py                    # Flask uygulaması
├── video_analyzer.py         # Video analizi modülü
├── video_clipper.py          # Video kırpma modülü
├── templates/                # HTML şablonları
│   ├── index.html           # Ana sayfa
│   ├── analyze.html         # Analiz sayfası
│   └── results.html         # Sonuçlar sayfası
├── static/                   # Statik dosyalar
│   ├── uploads/             # Yüklenen videolar
│   ├── css/
│   └── js/
└── clips/                    # Kırpılmış videolar
```

### 3. Uygulamayı Çalıştırın
```bash
cd web
python app.py
```

Tarayıcıda açın: http://localhost:5000

## 📋 Özellikler

- ✅ Video yükleme (drag & drop)
- ✅ Video önizleme
- ✅ Model ile otomatik olay tespiti
- ✅ Video kırpma
- ✅ Kırpılmış videoları görüntüleme
- ✅ Video indirme
- 🔄 YouTube paylaşımı (yakında)
- 🔄 Instagram paylaşımı (yakında)

## 🔧 Yapılandırma

Environment variables (`.env` dosyası):
```env
SECRET_KEY=your-secret-key
MODEL_PATH=data/models/event_classifier.pkl
```

## 📝 Kullanım

1. Ana sayfada video yükleyin
2. Video analizi otomatik başlar
3. Tespit edilen olayları görüntüleyin
4. İstediğiniz olayları kırpın
5. Sonuçlar sayfasında kırpılmış videoları görüntüleyin
6. Videoları indirin veya paylaşın

## 🐛 Bilinen Sorunlar

- Model entegrasyonu henüz tamamlanmadı (mock sonuçlar kullanılıyor)
- YouTube/Instagram API entegrasyonu yakında eklenecek
- Video kırpma işlemi zaman alabilir (büyük videolar için)

## 🚧 Geliştirme Durumu

- [x] Temel Flask uygulaması
- [x] Video yükleme
- [x] Video önizleme
- [x] Analiz sayfası
- [x] Video kırpma modülü
- [ ] Model entegrasyonu (tam)
- [ ] YouTube API
- [ ] Instagram API
- [ ] Background job processing
- [ ] İlerleme takibi (WebSocket)



