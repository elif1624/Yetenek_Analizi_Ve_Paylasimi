# Model Optimizasyon Raporu - Basketbol Olay Sınıflandırma

## 📊 Model Karşılaştırması

### Test Sonuçları

| Model | CV Score | Test Accuracy | Test F1-Score |
|-------|----------|---------------|---------------|
| **Gradient Boosting** | **0.750** | **0.600** | **0.533** |
| SVM | 0.500 | 0.400 | 0.229 |
| Random Forest | 0.567 | 0.200 | 0.160 |

**Sonuç:** Gradient Boosting en iyi performansı gösterdi!

---

## 🏆 Seçilen Model: Gradient Boosting

### Optimize Edilmiş Parametreler
```python
GradientBoostingClassifier(
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=5,
    random_state=42
)
```

### Performans Metrikleri
- **Test Accuracy:** 60.0% (Random Forest'ın 3 katı)
- **Test F1-Score:** 53.3% (Random Forest'ın 3.3 katı)
- **CV Accuracy:** 75.0% (5-fold cross-validation)
- **CV Std:** ±45.2% (küçük dataset nedeniyle yüksek varyans)

---

## 📈 Feature Importance Analizi

Gradient Boosting modelinde en önemli feature'lar:

1. **player_max_y** (24.25%) - Oyuncunun maksimum Y pozisyonu
2. **player_ball_min_distance** (20.25%) - Oyuncu-top minimum mesafesi
3. **duration** (17.85%) - Olay süresi
4. **player_min_y** (12.69%) - Oyuncunun minimum Y pozisyonu
5. **player_total_movement** (10.88%) - Toplam oyuncu hareketi

**Önemli Bulgular:**
- **Spatial features** (Y pozisyonu) en önemli
- **Ball-player interactions** (mesafe) kritik
- **Temporal features** (duration, movement) değerli

---

## 🎯 Model Özellikleri

### Gradient Boosting Neden Basketbol Verisi İçin Uygun?

1. **Non-linear Patterns:** Basketbol olayları non-linear ilişkiler içerir (pozisyon, hareket, mesafe kombinasyonları)
2. **Feature Interactions:** Model feature'lar arası etkileşimleri otomatik öğrenir
3. **Küçük Dataset:** 22 örnekle bile iyi performans gösterir
4. **Overfitting Kontrolü:** `max_depth=5` ve `min_samples_split=5` ile overfitting önlenir

---

## 📝 Test Set Detayları

### Confusion Matrix (5 örnek)
```
            Predicted
              basket  pas  blok
Actual basket   2     0    0
Actual   pas    1     1    0
Actual  blok    1     0    0
```

### Per-Class Performance
- **Basket:** Precision: 0.50, Recall: 1.00, F1: 0.67 (2 örnek)
- **Pas:** Precision: 1.00, Recall: 0.50, F1: 0.67 (2 örnek)
- **Blok:** Precision: 0.00, Recall: 0.00, F1: 0.00 (1 örnek)

**Not:** Test set çok küçük (5 örnek), bu yüzden metrikler güvenilir değil. CV score daha güvenilir.

---

## 🔄 Model Kullanımı

### Eğitim
```bash
python tools/train_model.py --model-type gradient_boosting
```

### Test
```bash
python tools/test_trained_model.py
```

### Optimizasyon (yeniden çalıştırma)
```bash
python tools/optimize_model.py
```

---

## 📊 Veri Durumu

- **Toplam örnek:** 22
- **Train set:** 17 (80%)
- **Test set:** 5 (20%)
- **Feature sayısı:** 26
- **Event dağılımı:**
  - Basket: 10 (45.5%)
  - Pas: 6 (27.3%)
  - Blok: 6 (27.3%)

---

## 🚀 Sonraki Adımlar

1. **Daha Fazla Veri:** 22 örnek çok az, en az 50-100 örnek hedeflenmeli
2. **Feature Engineering:** 
   - Potaya yakınlık (basket tespiti için)
   - Oyuncu hızlanma/yavaşlama (pas/blok tespiti için)
3. **Model İyileştirme:**
   - Ensemble methods (Random Forest + Gradient Boosting)
   - Hyperparameter fine-tuning (daha fazla veri ile)

---

## 💡 Öneriler

1. **Şimdilik Gradient Boosting kullanın** - En iyi performans
2. **Veri artırınca tekrar optimize edin** - Daha fazla veri ile daha iyi parametreler bulunabilir
3. **Feature importance'ı kullanın** - Yeni feature'lar eklerken en önemli feature'lara odaklanın
4. **Cross-validation sonuçlarına güvenin** - Test set çok küçük, CV daha güvenilir

---

**Tarih:** 2024
**Model Tipi:** Gradient Boosting Classifier
**Veri Seti:** 22 örnek, 26 feature
**En İyi Test Accuracy:** 60.0%




