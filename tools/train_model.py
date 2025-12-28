"""
ML Model Eğitim Scripti

Bu script, manuel olarak etiketlenmiş video klip örneklerinden çıkarılan
feature'ları kullanarak basket ve pas olaylarını sınıflandıran ML modelini eğitir.

Eğitim Süreci:
1. Feature JSON dosyasını yükler (extract_features.py ile oluşturulmuş)
2. Train/test split yapar (%80 train, %20 test)
3. Feature'ları normalize eder (StandardScaler)
4. Gradient Boosting Classifier ile model eğitir
5. Test set üzerinde metrikleri hesaplar (accuracy, precision, recall, F1)
6. Eğitilmiş modeli pickle formatında kaydeder

Kullanım:
    # Varsayılan parametrelerle
    python tools/train_model.py
    
    # Özel feature dosyası ile
    python tools/train_model.py --features data/dataset/features_augmented.json
    
    # Regularized model için (overfitting önleme)
    python tools/train_model_regularized.py
"""

import argparse
from pathlib import Path
import sys

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier


def main():
    parser = argparse.ArgumentParser(description="Basketbol olay sınıflandırıcı model eğitimi")
    parser.add_argument(
        "--features",
        type=str,
        default="data/dataset/features.json",
        help="Feature dosyası yolu (default: data/dataset/features.json)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gradient_boosting",
        choices=["random_forest", "gradient_boosting"],
        help="Model tipi (default: gradient_boosting - optimize edilmiş)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set oranı (default: 0.2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/event_classifier.pkl",
        help="Çıktı model dosyası yolu (default: data/models/event_classifier.pkl)"
    )
    
    args = parser.parse_args()
    
    # Dosya kontrolü
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"HATA: Feature dosyası bulunamadı: {features_path}")
        print(f"Önce feature extraction çalıştırın:")
        print(f"  python tools/extract_features.py")
        return 1
    
    # Model oluştur ve eğit
    classifier = EventClassifier(model_type=args.model_type)
    results = classifier.train(
        features_file=str(features_path),
        test_size=args.test_size
    )
    
    # Modeli kaydet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(str(output_path))
    
    print(f"\n{'='*60}")
    print("EĞİTİM TAMAMLANDI")
    print(f"{'='*60}")
    print(f"Model kaydedildi: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

