# 3. Metodoloji

## 3.1. Deneysel Kurulumlar

Bu çalışmada, basketbol video analizi için oluşturulan görüntü veri kümesi, eğitim için %80 ve test için %20 olacak şekilde iki kategoriye ayrılmıştır. Stratified split yöntemi kullanılarak sınıf dengesi korunmuştur (random_state=42). Eğitim alt kümesi, Gradient Boosting Classifier tabanlı olay sınıflandırma modelini eğitmek için kullanılmıştır. Bu çalışmada kullanılan modelin hiperparametreleri Tablo 2'de ayrıntılı olarak verilmiştir.

Çalışma Python programlama dili ve scikit-learn kütüphanesi kullanılarak gerçekleştirilmiştir. Model eğitimi ve değerlendirme işlemleri için yerel bir geliştirme ortamı kullanılmıştır. Özellik çıkarımı aşamasında OpenCV ve SAM3 (Segment Anything Model 3) tabanlı nesne tespiti ve takip sistemleri kullanılmıştır.

### Tablo 2: Model Hiperparametreleri

#### Normal Model (Gradient Boosting Classifier)
- **Ağaç Sayısı (n_estimators)**: 50
- **Maksimum Derinlik (max_depth)**: 5
- **Öğrenme Hızı (learning_rate)**: 0.1
- **Min Örnek Split (min_samples_split)**: 5
- **Kayıp Fonksiyonu (loss)**: Log-Loss (Cross-Entropy)
- **Rastgele Durum (random_state)**: 42

#### Regularized Model (Overfitting Önleme)
- **Ağaç Sayısı (n_estimators)**: 30
- **Maksimum Derinlik (max_depth)**: 3
- **Öğrenme Hızı (learning_rate)**: 0.1
- **Min Örnek Split (min_samples_split)**: 10
- **Min Örnek Yaprak (min_samples_leaf)**: 5
- **Alt Örnek Oranı (subsample)**: 0.8
- **Maksimum Özellik Sayısı (max_features)**: sqrt
- **Kayıp Fonksiyonu (loss)**: Log-Loss (Cross-Entropy)
- **Rastgele Durum (random_state)**: 42

### Ön İşleme
Özellik vektörleri StandardScaler kullanılarak normalize edilmiştir. Eksik değerler (NaN) 0 ile doldurulmuştur. Her bir video penceresi için oyuncu pozisyonları, top konumu, hareket hızları ve mesafe metrikleri gibi 30+ özellik çıkarılmıştır.

### Veri Artırma (Data Augmentation)
Eğitim veri kümesinin boyutunu artırmak için, mevcut video klip örneklerine parlaklık ayarlama, yatay çevirme ve döndürme gibi dönüşümler uygulanarak yeni örnekler oluşturulmuştur.


