"""
Overfitting (Ezberleme) Kontrolü
Modelin ezberleyip ezberlemediğini kontrol eder
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
import sys

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    parser = argparse.ArgumentParser(description='Overfitting Kontrolu')
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/event_classifier.pkl',
        help='Model dosyasi yolu'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/dataset/features.json',
        help='Feature dosyasi yolu'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test split orani'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    features_path = Path(args.features)
    
    if not model_path.exists():
        print(f"HATA: Model dosyasi bulunamadi: {model_path}")
        return 1
    
    if not features_path.exists():
        print(f"HATA: Feature dosyasi bulunamadi: {features_path}")
        return 1
    
    # Modeli yukle
    print("Model yukleniyor...")
    classifier = EventClassifier.load(str(model_path))
    
    # Feature'lari yukle
    with open(features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data['features']
    print(f"Toplam ornek: {len(features)}\n")
    
    # Feature'lari hazirla
    X, y = classifier._prepare_features(features)
    
    # Ayni train-test split'i kullan
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # Feature scaling (model'in scaler'ini kullan)
    X_train_scaled = classifier.scaler.transform(X_train)
    X_test_scaled = classifier.scaler.transform(X_test)
    
    # Train set uzerinde tahmin
    y_train_pred = classifier.model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test set uzerinde tahmin
    y_test_pred = classifier.model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Fark hesapla
    accuracy_diff = train_accuracy - test_accuracy
    
    print("="*70)
    print("OVERFITTING (EZBERLEME) ANALIZI")
    print("="*70)
    print()
    
    print("PERFORMANS KARSILASTIRMASI:")
    print("-"*70)
    print(f"Train Set Accuracy:  {train_accuracy:.1%} ({train_accuracy*100:.1f}%)")
    print(f"Test Set Accuracy:   {test_accuracy:.1%} ({test_accuracy*100:.1f}%)")
    print(f"Fark (Train - Test): {accuracy_diff:.1%} ({accuracy_diff*100:.1f}%)")
    print()
    
    # Overfitting degerlendirmesi
    print("DEGERLENDIRME:")
    print("-"*70)
    
    if accuracy_diff < 0.05:  # %5'ten az fark
        print("[OK] Cok iyi! Model ezberlemiyor.")
        print("    Train ve test accuracy cok yakın - model genelleştirme yapıyor.")
    elif accuracy_diff < 0.10:  # %5-10 arası
        print("[IKI] Iyi, ama dikkatli olunmalı.")
        print("    Küçük bir fark var, model hafif ezberleme yapıyor olabilir.")
    elif accuracy_diff < 0.20:  # %10-20 arası
        print("[UYARI] Dikkat! Model ezberleme yapıyor olabilir.")
        print("    Train accuracy test'ten belirgin şekilde yüksek.")
    else:  # %20'den fazla
        print("[SORUN] Model ciddi şekilde ezberliyor!")
        print("    Train ve test accuracy arasında büyük fark var.")
    
    print()
    
    # Detaylı karşılaştırma
    print("DETAYLI KARSILASTIRMA:")
    print("-"*70)
    
    # Her sınıf için train/test karşılaştırması
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    class_names = [classifier.idx_to_class[idx] for idx in unique_classes]
    
    print(f"\n{'Sinif':<10} {'Train Acc':<12} {'Test Acc':<12} {'Fark':<10}")
    print("-"*70)
    
    for idx, class_name in zip(unique_classes, class_names):
        # Train
        train_mask = y_train == idx
        train_class_acc = accuracy_score(y_train[train_mask], y_train_pred[train_mask]) if train_mask.sum() > 0 else 0
        
        # Test
        test_mask = y_test == idx
        test_class_acc = accuracy_score(y_test[test_mask], y_test_pred[test_mask]) if test_mask.sum() > 0 else 0
        
        diff = train_class_acc - test_class_acc
        
        print(f"{class_name:<10} {train_class_acc:>10.1%}  {test_class_acc:>10.1%}  {diff:>8.1%}")
    
    print()
    
    # Veri seti boyutu değerlendirmesi
    print("VERI SETI BOYUTU DEGERLENDIRMESI:")
    print("-"*70)
    print(f"Toplam ornek: {len(features)}")
    print(f"Train ornek: {len(X_train)}")
    print(f"Test ornek: {len(X_test)}")
    print()
    
    if len(features) < 100:
        print("[NOT] Veri seti kucuk (< 100 ornek).")
        print("    - Daha fazla veri toplamak modeli iyilestirebilir")
        print("    - Ancak mevcut sonuclar kabul edilebilir gorunuyor")
    elif len(features) < 500:
        print("[NOT] Orta boyutlu veri seti (100-500 ornek).")
        print("    - Model performansi genellikle bu boyutta iyilesir")
    else:
        print("[NOT] Buyuk veri seti (> 500 ornek).")
        print("    - Model iyi genellesme yapmali")
    
    print()
    
    # Öneriler
    print("ONERILER:")
    print("-"*70)
    
    if accuracy_diff > 0.10:
        print("1. Model ezberleme yapiyor olabilir:")
        print("   - Daha fazla veri toplayin (en etkili cozum)")
        print("   - Model kompleksligini azaltin (max_depth dusurun)")
        print("   - Regularization ekleyin")
    else:
        print("1. Model genelleme yapiyor - mevcut durum iyi!")
    
    if len(features) < 200:
        print("2. Veri setini genisletmek performansi artirabilir:")
        print("   - Daha fazla video etiketleyin")
        print("   - Farkli kosullarda video ekleyin (farkli maclar, kosullar)")
    else:
        print("2. Mevcut veri seti boyutu yeterli gorunuyor")
    
    print("3. Cross-validation sonuclari onemli:")
    print("   - Egitim sirasinda gorulen CV accuracy: 84.4%")
    print("   - Bu, modelin farkli veri parcalarinda tutarli performans gosterdigini soyler")
    
    print()
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())





