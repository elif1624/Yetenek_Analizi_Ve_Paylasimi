# 🏀 Basketbol Video Analizi - Yetenek Analizi

SAM3 tabanlı basketbol video analizi sistemi. Oyuncu tespiti, tracking, olay tespiti (basket, pas) ve manuel etiketleme araçları.

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Video Analizi
```bash
python analyze_video_final.py --video data/input/video.mp4 --fps 5.0
```

### 3. Manuel Etiketleme
```bash
python tools/labeling_tool_improved.py data/input/video.mp4
```

## 📁 Proje Yapısı

```
├── analyze_video_final.py      # Ana analiz scripti
├── src/
│   ├── ai/
│   │   ├── sam3_local.py       # SAM3 model (nesne tespiti)
│   │   ├── detection.py        # Oyuncu tespiti
│   │   ├── tracking_improved.py # Oyuncu tracking
│   │   ├── event_detection.py  # Olay tespiti (basket, pas)
│   │   └── ball_detector.py    # Top tespiti
│   ├── analysis/
│   │   └── metrics.py          # Doğruluk metrikleri
│   └── video/
│       └── processor.py        # Video işleme
└── tools/
    ├── labeling_tool_improved.py # İyileştirilmiş etiketleme aracı
    ├── test_model.py            # Model test scripti
    ├── analyze_labels.py       # Etiket analizi
    └── visualize_events.py      # Olay görselleştirme
```

## 🎯 Özellikler

### Video Analizi
- ✅ SAM3 ile oyuncu ve top tespiti
- ✅ Gelişmiş tracking (aynı oyuncu = aynı ID)
- ✅ Kalman filter tabanlı takip sistemi
- ✅ Top pozisyonu tespiti

### ML Model
- ✅ Gradient Boosting Classifier ile olay sınıflandırması (basket, pas)
- ✅ Feature extraction (30+ özellik)
- ✅ Model eğitimi ve değerlendirme araçları
- ✅ Overfitting önleme (regularized model)

### Web Uygulaması
- ✅ Web arayüzü ile video yükleme
- ✅ Otomatik olay tespiti (ML model)
- ✅ Video kırpma (clipping)
- ✅ YouTube/Facebook paylaşımı
- ✅ Analiz sonuçlarını cache'leme

### Araçlar
- ✅ Manuel etiketleme aracı
- ✅ Model doğruluk metrikleri ve grafikler
- ✅ Olay görselleştirme
- ✅ Dataset yönetimi

## 📊 Kullanım

### Video Analizi
```bash
python analyze_video_final.py --video data/input/video.mp4
```

### Manuel Etiketleme
```bash
python tools/labeling_tool_improved.py data/input/video.mp4
```

### Model Testi
```bash
python tools/test_model.py --video data/input/video.mp4 --labels data/labels/video_labels.json
```

### Model Eğitimi
```bash
# Normal model
python tools/train_model.py

# Regularized model (overfitting önleme - önerilen)
python tools/train_model_regularized.py
```

## 📝 Etiketleme Rehberi

Detaylı kullanım için: `ETIKETLEME_KILAVUZU.txt`

## 🔧 Yapılandırma

### Environment Variables (Opsiyonel)

`.env` dosyası oluştur:
```env
HUGGINGFACE_API_TOKEN=your_token
HUGGINGFACE_MODEL_NAME=facebook/sam3
FRAME_EXTRACTION_FPS=3.0
OUTPUT_DIR=data/output
RESULTS_DIR=data/results
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key  # Web uygulaması için
```

**Not:** Çoğu ayar varsayılan değerlerle çalışır. Sadece SAM3 modeli için Hugging Face token'ı gerekebilir.

## 🌐 Web Uygulaması

Web arayüzü ile video analizi, kırpma ve sosyal medya paylaşımı:

```bash
cd web
python app.py
```

Tarayıcıda açın: http://localhost:5000

Detaylı bilgi için: `web/README.md`

## 🤖 ML Model

Model eğitimi ve kullanımı:

```bash
# Model eğitimi
python tools/train_model.py

# Regularized model (overfitting önleme)
python tools/train_model_regularized.py
```

Eğitilmiş model: `data/models/event_classifier_regularized.pkl`
