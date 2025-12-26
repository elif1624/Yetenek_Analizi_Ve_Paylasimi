# 🎯 Model Eğitim Stratejisi: Regularization + Data Augmentation

## 📚 Açıklamalar

### 1. Regularization (Düzenleme) Nedir?

**Sorun:** Model veriyi ezberleyebilir (overfitting)

**Çözüm:** Regularization modelin kompleksliğini sınırlar

**Nasıl Çalışır:**
- Daha az ağaç kullanır (n_estimators: 50 → 30)
- Daha sığ ağaçlar (max_depth: 5 → 3)
- Daha fazla örnek gerektirir (min_samples_split: 5 → 10)
- Her ağaç sadece %80 örnek kullanır (subsample=0.8)
- Sadece feature'ların bir kısmını kullanır (max_features='sqrt')

**Sonuç:**
- ✅ Model daha genel kurallar öğrenir
- ✅ Ezberleme azalır
- ⚠️ Performans biraz düşebilir (ama daha güvenilir)

---

### 2. Data Augmentation (Veri Çoğaltma) Nedir?

**Sorun:** Az veri var (104 örnek)

**Çözüm:** Mevcut veriyi çoğaltarak veri setini genişlet

**Nasıl Çalışır:**
- Her feature'a küçük rastgele noise eklenir (%5-10)
- Yeni örnekler oluşturulur ama "gerçekçi" kalır
- Orijinal veri korunur, sadece yeni örnekler eklenir

**Örnek:**
```
Orijinal: player_avg_x = 940.5
Augmented: player_avg_x = 940.5 + (rastgele ±5%) = 945.2
```

**Sonuç:**
- ✅ Daha fazla örnek = daha iyi öğrenme
- ✅ Model farklı varyasyonları görür
- ✅ Daha robust model

---

## 🔬 İkisini Birlikte Kullanmak

### Neden Mantıklı?

1. **Data Augmentation:** Daha fazla veri sağlar
2. **Regularization:** Fazla komplekslikten korur
3. **Birlikte:** Hem daha fazla veri, hem daha güvenilir model

### Strateji

```
1. Orijinal veri (104 örnek)
   ↓
2. Data Augmentation (104 → ~208 örnek)
   ↓
3. Regularized Model Eğitimi
   ↓
4. Daha iyi genelleştirme!
```

---

## 📊 Beklenen Sonuçlar

### Senaryo 1: Sadece Regularization
- ✅ Ezberleme azalır
- ⚠️ Performans biraz düşer (az veri yeterli değil)

### Senaryo 2: Sadece Data Augmentation  
- ✅ Daha fazla veri
- ⚠️ Model hala ezberleyebilir (kompleks model)

### Senaryo 3: İkisi Birlikte (ÖNERİLEN) ⭐
- ✅ Daha fazla veri (augmentation)
- ✅ Ezberleme azalır (regularization)
- ✅ Daha güvenilir ve robust model
- ✅ Test accuracy'de iyileşme beklenir

---

## 🚀 Kullanım

### Adım 1: Data Augmentation
```bash
python tools/augment_features.py --factor 1.0 --noise 0.05
```

### Adım 2: Regularized Model Eğitimi
```bash
python tools/train_model_regularized.py --features data/dataset/features_augmented.json
```

### Adım 3: Karşılaştırma
```bash
python tools/check_overfitting.py --model data/models/event_classifier_regularized.pkl
```

---

## ⚙️ Parametreler

### Augmentation Parametreleri

- `--factor 1.0`: Her örnek için 1 yeni örnek (2x veri)
- `--factor 2.0`: Her örnek için 2 yeni örnek (3x veri)
- `--noise 0.05`: %5 değişiklik (hafif)
- `--noise 0.10`: %10 değişiklik (daha fazla çeşitlilik)

**Öneri:** `--factor 1.0 --noise 0.05` ile başla, sonuçlara göre ayarla

### Regularization Parametreleri

- `n_estimators=30`: Daha az ağaç
- `max_depth=3`: Daha sığ ağaçlar
- `min_samples_split=10`: Daha fazla örnek gerektirir
- `subsample=0.8`: Her ağaç %80 örnek kullanır

---

## 📈 Sonuç Analizi

Karşılaştırılacak metrikler:

1. **Test Accuracy:** Hedef: %90+ (korunmalı veya artmalı)
2. **Train-Test Farkı:** Hedef: <%5 (ezberleme azalmalı)
3. **Cross-Validation:** Hedef: Test'e yakın (tutarlılık)

Başarı kriterleri:
- ✅ Test accuracy korunuyor veya artıyor
- ✅ Train-test farkı azalıyor (<%5)
- ✅ Model daha güvenilir hale geliyor



