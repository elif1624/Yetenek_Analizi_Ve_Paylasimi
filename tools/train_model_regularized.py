"""
Regularized (Düzenli) Model Eğitimi
Ezberlemeyi azaltmak için regularization ile model eğitir
"""

import argparse
from pathlib import Path
import sys

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier
from sklearn.ensemble import GradientBoostingClassifier


class RegularizedEventClassifier(EventClassifier):
    """Regularization eklenmiş EventClassifier"""
    
    def _create_model(self):
        """Regularization ile model oluştur"""
        if self.model_type == "gradient_boosting":
            # Daha az kompleks, regularization ile
            self.model = GradientBoostingClassifier(
                n_estimators=30,  # Daha az ağaç (50'den 30'a)
                max_depth=3,      # Daha sığ ağaçlar (5'ten 3'e)
                learning_rate=0.1,
                min_samples_split=10,  # Daha fazla örnek gerekli (5'ten 10'a)
                min_samples_leaf=5,    # Leaf'lerde en az 5 örnek
                subsample=0.8,         # Her ağaç için %80 örnek kullan
                random_state=42,
                max_features='sqrt'    # Feature'ların sqrt'i kadar kullan
            )
        else:
            super()._create_model()


def main():
    parser = argparse.ArgumentParser(description="Regularized Model Eğitimi")
    parser.add_argument(
        "--features",
        type=str,
        default="data/dataset/features.json",
        help="Feature dosyası yolu"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/event_classifier_regularized.pkl",
        help="Çıktı model dosyası yolu"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set oranı"
    )
    
    args = parser.parse_args()
    
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"HATA: Feature dosyası bulunamadı: {features_path}")
        return 1
    
    print("="*70)
    print("REGULARIZED MODEL EĞİTİMİ")
    print("="*70)
    print("\nRegularization parametreleri:")
    print("  - Daha az ağaç (30 yerine 50)")
    print("  - Daha sığ ağaçlar (max_depth=3)")
    print("  - Daha fazla min_samples (10)")
    print("  - Subsample (her ağaç %80 örnek kullanır)")
    print("  - Max features (sqrt kullanım)")
    print("\nBu parametreler ezberlemeyi azaltır ama performansı biraz düşürebilir.")
    print("="*70)
    print()
    
    # Regularized model oluştur ve eğit
    classifier = RegularizedEventClassifier(model_type="gradient_boosting")
    results = classifier.train(
        features_file=str(features_path),
        test_size=args.test_size
    )
    
    # Modeli kaydet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(str(output_path))
    
    print(f"\n{'='*70}")
    print("REGULARIZED MODEL EĞİTİMİ TAMAMLANDI")
    print(f"{'='*70}")
    print(f"Model kaydedildi: {output_path}")
    print("\nBu modeli test etmek için:")
    print(f"  python tools/check_overfitting.py --model {output_path}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    exit(main())



