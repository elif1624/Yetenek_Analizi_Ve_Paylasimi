# 🎯 Sistem Geliştirme Planı - Manuel Etiketleme + ML Model

## 📊 Mevcut Durum

- ✅ **23 manuel etiket** (ground truth)
  - 11 basket
  - 6 pas
  - 6 blok
- ✅ **SAM3 tespitleri** (oyuncu, top)
- ✅ **Tracking verileri** (trajectory)
- ✅ **Manuel etiketleme araçları** (hazır)

---

## 🎯 HEDEF: ML Model ile Otomatik Tespit

### Genel Yaklaşım
```
Manuel Etiketleme (Ground Truth)
    ↓
Feature Extraction (SAM3 + Tracking) ← SAM3 BURADA GEREKLİ!
    ↓
ML Model Eğitimi
    ↓
Otomatik Tespit
```

**Önemli**: 
- Rule-based yaklaşım kullanılmayacak, sadece ML model.
- SAM3 neden gerekli? → ML model'e öğretecek feature'ları sağlıyor!

### SAM3'ün Rolü

**Manuel Etiketleme:**
- "Frame 172-187 = BASKET" (sadece zaman bilgisi)

**SAM3:**
- "Frame 172'de: Oyuncu (800,250), Top (810,245), Trajectory: potaya doğru..."
- "Frame 173'te: Oyuncu (805,248), Top (815,240), Hız: yavaşlıyor..."
- (Feature'lar - ML model bunları öğrenecek!)

**ML Model:**
- Input: SAM3 feature'ları (oyuncu, top, trajectory)
- Output: Olay tipi (basket, pas, blok)
- Training: Manuel etiketler (ground truth)

**Sonuç:**
- Manuel etiket: "Ne zaman, ne oldu?" (zaman + olay)
- SAM3: "Nasıl oldu?" (feature'lar)
- ML Model: "Öğreniyor!" (pattern matching)

---

## 📋 ADIM ADIM PLAN

### **ADIM 1: Veri Hazırlama** (Şimdi)

#### 1.1 Mevcut Veriyi Analiz Et
```bash
# Etiketleri analiz et
python tools/analyze_labels.py

# Pattern analizi (feature çıkarma için)
python tools/analyze_event_patterns.py
```

**Hedef**: 23 etiketten feature'ları çıkar

#### 1.2 Feature Extraction Sistemi Oluştur

**Her etiket için feature çıkar:**
- SAM3 tespitleri (oyuncu pozisyonları, bbox'lar)
- Tracking verileri (trajectory, hız, yön)
- Top pozisyonları (ball tracking)
- Zaman bilgileri (frame, süre)

**Çıktı**: Feature vector'lar (ML model için hazır)

---

### **ADIM 2: ML Model Seçimi ve Hazırlık** (1 hafta)

#### 2.1 Model Seçimi

**Seçenekler:**

**A. VideoMAE (Video Masked Autoencoder)**
- Pre-trained: Kinetics-400
- Fine-tuning: 23 etiketle başla
- Avantaj: Video action recognition
- Model: `MCG-NJU/videomae-base-finetuned-kinetics`

**B. X-CLIP (Cross-modal CLIP)**
- Pre-trained: Video-text matching
- Fine-tuning: 23 etiketle başla
- Avantaj: Text prompt desteği
- Model: `microsoft/xclip-base-patch32`

**C. TimeSformer**
- Pre-trained: Video understanding
- Fine-tuning: 23 etiketle başla
- Avantaj: Temporal modeling
- Model: `facebook/timesformer-base-finetuned-k400`

#### 2.2 Veri Formatı Hazırlama

**23 etiketten dataset oluştur:**
```python
# Her etiket için:
{
    'video_path': '...',
    'event_type': 'basket',
    'start_frame': 172,
    'end_frame': 187,
    'features': {
        'player_positions': [...],
        'ball_positions': [...],
        'trajectories': [...],
        ...
    }
}
```

#### 2.3 Model Entegrasyonu

**Pre-trained model yükle:**
- Hugging Face'den model indir
- Feature extraction layer ekle
- Fine-tuning için hazırla

---

### **ADIM 3: İlk Model Eğitimi** (1-2 hafta)

#### 3.1 23 Etiketle Fine-Tuning

**Transfer Learning:**
1. Pre-trained model yükle
2. Son katmanları değiştir (3 sınıf: basket, pas, blok)
3. 23 etiketle fine-tune et
4. Validation split (80/20)

**Beklenen Sonuç:**
- %40-60 doğruluk (ilk deneme)
- Model öğrenmeye başlıyor

#### 3.2 Test ve Değerlendirme

**23 etiketle test:**
```bash
python tools/test_model.py --model videomae
```

**Metrikler:**
- Precision, Recall, F1-Score
- Per-event accuracy
- Confusion matrix

---

### **ADIM 4: Veri Çoğaltma** (2-3 hafta)

#### 4.1 Ne Zaman?
- Model %40-60 doğrulukta takıldığında
- Daha fazla öğrenmek için veri gerekir

#### 4.2 Nasıl?
- İyileştirilmiş etiketleme aracı ile
- 50-100 video daha etiketle
- Her video 5-10 olay içermeli

**Hedef**: 50-100 etiketli video (250-500 olay)

#### 4.3 Model Yeniden Eğitimi

**Daha fazla veri ile:**
- 50-100 etiketle fine-tuning
- Daha iyi genelleme
- Daha yüksek doğruluk

**Beklenen Sonuç:**
- %60-80 doğruluk

---

### **ADIM 5: Model İyileştirme** (1-2 hafta)

#### 5.1 Hyperparameter Tuning

**Optimize edilecekler:**
- Learning rate
- Batch size
- Epoch sayısı
- Data augmentation

#### 5.2 Model Mimarisi İyileştirme

**Denenebilecekler:**
- Farklı pre-trained modeller
- Ensemble modeller
- Custom architecture

**Beklenen Sonuç:**
- %80-90 doğruluk

---

## 🔄 İTERATİF GELİŞİM DÖNGÜSÜ

```
1. Manuel Etiketleme
   ↓
2. Feature Extraction
   ↓
3. Model Eğitimi
   ↓
4. Test ve Değerlendirme
   ↓
5. Hata Analizi
   ↓
6. Daha Fazla Veri Toplama (gerekirse)
   ↓
7. Model İyileştirme
   ↓
8. Tekrarla (1'e dön)
```

---

## 📊 BEKLENEN GELİŞİM

| Adım | Veri | Doğruluk | Yöntem | Süre |
|------|------|----------|--------|------|
| Mevcut | 23 etiket | - | Manuel etiketleme | - |
| Adım 1 | 23 etiket | - | Feature extraction | 3-5 gün |
| Adım 2 | 23 etiket | - | Model hazırlık | 1 hafta |
| Adım 3 | 23 etiket | %40-60 | İlk fine-tuning | 1-2 hafta |
| Adım 4 | 50-100 etiket | %60-80 | Veri çoğaltma + eğitim | 2-3 hafta |
| Adım 5 | 50-100 etiket | %80-90 | Model iyileştirme | 1-2 hafta |

---

## 🎯 ŞİMDİ NE YAPALIM?

### Seçenek 1: Feature Extraction (Önerilen)
1. ✅ 23 etiketten feature çıkar
2. ✅ ML model için veri hazırla
3. ✅ Dataset oluştur

### Seçenek 2: Model Seçimi ve Entegrasyonu
1. ✅ VideoMAE/X-CLIP seç
2. ✅ Model entegre et
3. ✅ 23 etiketle test et

---

## 💡 ÖNERİM

**Şimdi yapılacaklar (sırayla):**

1. **Feature Extraction Sistemi** (3-5 gün)
   - 23 etiketten feature çıkar
   - ML model için veri formatı oluştur
   - Dataset hazırla

2. **Model Seçimi ve Entegrasyonu** (1 hafta)
   - VideoMAE veya X-CLIP seç
   - Model entegre et
   - Fine-tuning için hazırla

3. **İlk Model Eğitimi** (1-2 hafta)
   - 23 etiketle fine-tuning
   - Test ve değerlendirme
   - %40-60 doğruluk hedefi

4. **Veri Çoğaltma** (2-3 hafta)
   - 50-100 video daha etiketle
   - Model yeniden eğit
   - %60-80 doğruluk hedefi

5. **Model İyileştirme** (1-2 hafta)
   - Hyperparameter tuning
   - Model mimarisi iyileştirme
   - %80-90 doğruluk hedefi

---

## 📝 ÖNEMLİ NOTLAR

1. **Rule-based yaklaşım kullanılmayacak**
   - Sadece ML model
   - Manuel etiketleme → ML model

2. **23 etiket yeterli mi?**
   - İlk fine-tuning için: Evet (başlangıç)
   - Yüksek doğruluk için: Hayır (50-100 gerekli)

3. **Ne zaman veri çoğaltmalı?**
   - Model %40-60 doğrulukta takıldığında
   - Daha fazla öğrenmek için veri gerekir

---

## 🚀 BAŞLAYALIM MI?

Hangi adımdan başlamak istersin?

1. **Feature Extraction** (23 etiketten feature çıkarma)
2. **Model Seçimi** (VideoMAE/X-CLIP/TimeSformer)
3. **Model Entegrasyonu** (Pre-trained model yükleme)
