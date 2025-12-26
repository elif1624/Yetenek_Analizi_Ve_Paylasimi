# 🏀 Basketbol Video Analizi - Yetenek Analizi

SAM3 tabanlı basketbol video analizi sistemi. Oyuncu tespiti, tracking, olay tespiti (basket, pas) ve manuel etiketleme araçları.

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Video Analizi
```bash
python analyze_video_final.py
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

- ✅ SAM3 ile oyuncu ve top tespiti
- ✅ Gelişmiş tracking (aynı oyuncu = aynı ID)
- ✅ Olay tespiti (basket, pas)
- ✅ Manuel etiketleme aracı
- ✅ Model doğruluk metrikleri
- ✅ Olay görselleştirme

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

## 📝 Etiketleme Rehberi

Detaylı kullanım için: `ETIKETLEME_REHBERI.md`

## 🔧 Yapılandırma

`.env` dosyası oluştur:
```
HUGGINGFACE_API_TOKEN=your_token
HUGGINGFACE_MODEL_NAME=facebook/sam3
FRAME_EXTRACTION_FPS=3.0
OUTPUT_DIR=data/output
RESULTS_DIR=data/results
LOG_LEVEL=INFO
```

## 📈 Sonraki Adımlar

1. Manuel etiketleme ile veri toplama (50-100 video)
2. Otomatik model entegrasyonu (VideoMAE/EITNet)
3. Custom model eğitimi (100+ veri ile)
