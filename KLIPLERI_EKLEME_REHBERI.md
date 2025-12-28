# 📁 Klipleri Veri Setine Ekleme Rehberi

Bu rehber, `pass_clips` ve `basket_clips` klasörlerindeki klipleri model eğitimi için veri setine nasıl ekleyeceğinizi açıklar.

## 📂 Klipleri Nereye Koymalıyım?

Klipleri **herhangi bir yere** koyabilirsiniz, ancak önerilen yapı:

```
Yetenek_Analizi/
├── data/
│   ├── input/
│   │   ├── clips/
│   │   │   ├── pass_clips/      ← Pass klipleri buraya
│   │   │   │   ├── pass1.mp4
│   │   │   │   ├── pass2.mp4
│   │   │   │   └── ...
│   │   │   └── basket_clips/    ← Basket klipleri buraya
│   │   │       ├── basket1.mp4
│   │   │       ├── basket2.mp4
│   │   │       └── ...
```

**Veya** klasörlerinizi başka bir yerde tutuyorsanız, script'e yolunu verebilirsiniz.

## 🚀 Adım Adım İşlem

### 1. Klipleri Hazırlayın

Klipleriniz hazır olmalı:
- Format: `.mp4`, `.avi`, `.mov`, `.mkv` (`.mp4` önerilir)
- Her klip sadece **bir olay** içermeli (ya pas ya basket)
- Klipler kısa olabilir (1-5 saniye ideal)

### 2. Script'i Çalıştırın

```bash
python tools/add_clips_to_dataset.py \
    --pass-clips "C:/path/to/pass_clips" \
    --basket-clips "C:/path/to/basket_clips"
```

**Örnekler:**

Eğer klipler `data/input/clips/` altındaysa:
```bash
python tools/add_clips_to_dataset.py \
    --pass-clips "data/input/clips/pass_clips" \
    --basket-clips "data/input/clips/basket_clips"
```

Eğer klipler başka bir yerdeyse (tam yol verin):
```bash
python tools/add_clips_to_dataset.py \
    --pass-clips "C:/Users/LOQ/Videos/pass_clips" \
    --basket-clips "C:/Users/LOQ/Videos/basket_clips"
```

### 3. Ne Olacak?

Script şunları yapacak:

1. **Her klip için video analizi** yapılır (SAM3 + tracking)
2. **Otomatik etiket** oluşturulur (tüm video pas veya basket olarak)
3. **Feature extraction** yapılır
4. **Mevcut veri setine eklenir** (`data/dataset/features.json`)

### 4. Modeli Yeniden Eğitin

Klipler eklendikten sonra:

```bash
python tools/train_model.py --features data/dataset/features.json
```

### 5. Grafikleri Yeniden Oluşturun

```bash
python tools/create_training_graphs.py --epochs 30
python tools/show_confusion_matrix.py
```

## ⚙️ İleri Seviye Seçenekler

### Farklı Çıktı Dosyası

Mevcut `features.json`'ı korumak için yeni dosya oluştur:

```bash
python tools/add_clips_to_dataset.py \
    --pass-clips "data/input/clips/pass_clips" \
    --basket-clips "data/input/clips/basket_clips" \
    --output "data/dataset/features_with_clips.json"
```

### Sadece Yeni Klipler (Mevcut Veriyi Koru)

```bash
python tools/add_clips_to_dataset.py \
    --pass-clips "data/input/clips/pass_clips" \
    --basket-clips "data/input/clips/basket_clips" \
    --existing-features "data/dataset/features.json" \
    --output "data/dataset/features_combined.json"
```

## ❓ Sık Sorulan Sorular

### Kaç klip eklemeliyim?

- **Minimum**: 10-20 klip her sınıf için (toplam 20-40)
- **İdeal**: 50+ klip her sınıf için (toplam 100+)
- **Daha fazla veri = daha iyi model performansı**

### Klip süresi ne kadar olmalı?

- **Kısa klipler**: 1-3 saniye (ideal)
- **Orta klipler**: 3-5 saniye (kabul edilebilir)
- **Uzun klipler**: 5+ saniye (mümkünse kırpın)

### İşlem ne kadar sürer?

- **Her klip için**: ~30-60 saniye (video analizi)
- **10 klip**: ~5-10 dakika
- **50 klip**: ~25-50 dakika

### Hata alırsam ne yapmalıyım?

1. Video formatını kontrol edin (`.mp4` önerilir)
2. Video dosyalarının bozuk olmadığından emin olun
3. Yeterli disk alanı olduğundan emin olun
4. Hata mesajını okuyun ve gerekirse log dosyasına bakın

## 📊 Sonuç

Klipler eklendikten sonra:

✅ Daha fazla eğitim verisi  
✅ Daha doğru model  
✅ Daha iyi grafikler  
✅ Daha yüksek accuracy  

**Not**: Her yeni veri ekledikten sonra modeli yeniden eğitmeyi unutmayın!


