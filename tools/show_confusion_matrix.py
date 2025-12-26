"""
Confusion Matrix Görselleştirme Aracı
Eğitilmiş modelin confusion matrix'ini gösterir
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier
from sklearn.metrics import confusion_matrix, classification_report


def load_model_and_features(model_path: Path, features_path: Path):
    """Model ve feature'ları yükle"""
    # Modeli yükle (classmethod olarak çağrılmalı)
    classifier = EventClassifier.load(str(model_path))
    
    # Feature'ları yükle
    with open(features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return classifier, data['features']


def create_confusion_matrix_display(y_test, y_pred, class_names):
    """Confusion matrix'i daha okunabilir formatta göster"""
    cm = confusion_matrix(y_test, y_pred)
    
    # DataFrame olarak oluştur
    cm_df = pd.DataFrame(cm, 
                        index=[f'Gerçek: {name}' for name in class_names],
                        columns=[f'Tahmin: {name}' for name in class_names])
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX (Karışıklık Matrisi)")
    print("="*70)
    print("\nAçıklama:")
    print("- Satırlar: Gerçek değerler (Actual)")
    print("- Sütunlar: Tahmin edilen değerler (Predicted)")
    print("- Diagonal (köşegen) değerler: DOĞRU tahminler")
    print("- Diagonal dışı değerler: YANLIŞ tahminler")
    print("\n" + "-"*70)
    print(cm_df.to_string())
    print("-"*70)
    
    # Detaylı analiz
    print("\nDetayli Analiz:")
    print("-"*70)
    total = cm.sum()
    correct = cm.trace()
    incorrect = total - correct
    
    print(f"Toplam ornek: {total}")
    print(f"[OK] Dogru tahmin: {correct} ({correct/total*100:.1f}%)")
    print(f"[X] Yanlis tahmin: {incorrect} ({incorrect/total*100:.1f}%)")
    print()
    
    # Her sınıf için detay
    for i, class_name in enumerate(class_names):
        total_class = cm[i, :].sum()
        correct_class = cm[i, i]
        incorrect_class = total_class - correct_class
        
        if total_class > 0:
            accuracy_class = correct_class / total_class * 100
            print(f"{class_name.upper()}:")
            print(f"  Toplam: {total_class} ornek")
            print(f"  [OK] Dogru tahmin: {correct_class} ({accuracy_class:.1f}%)")
            print(f"  [X] Yanlis tahmin: {incorrect_class}")
            
            # Hangi sınıfa yanlış tahmin edilmiş?
            for j, other_class in enumerate(class_names):
                if i != j and cm[i, j] > 0:
                    print(f"    -> {cm[i, j]} ornek '{other_class}' olarak yanlis tahmin edilmis")
            print()


def main():
    parser = argparse.ArgumentParser(description='Confusion Matrix Görselleştirme')
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/event_classifier.pkl',
        help='Model dosyası yolu'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/dataset/features.json',
        help='Feature dosyası yolu (test için)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test split oranı (model eğitimindeki ile aynı olmalı)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state (model eğitimindeki ile aynı olmalı)'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    features_path = Path(args.features)
    
    if not model_path.exists():
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return 1
    
    if not features_path.exists():
        print(f"HATA: Feature dosyası bulunamadı: {features_path}")
        return 1
    
    print("Model ve veriler yükleniyor...")
    classifier, features = load_model_and_features(model_path, features_path)
    
    # Feature'ları hazırla (model eğitimindeki gibi)
    from sklearn.model_selection import train_test_split
    
    # Event type'ları label'a çevir
    event_types = [f['event_type'] for f in features]
    y_all = np.array([classifier.class_to_idx.get(et, 0) for et in event_types])
    
    # Aynı train-test split'i kullan
    indices = np.arange(len(features))
    _, test_indices = train_test_split(
        indices,
        test_size=args.test_split,
        random_state=args.random_state,
        stratify=y_all
    )
    
    test_features = [features[i] for i in test_indices]
    
    # Test set için tahmin yap
    print(f"\nTest set: {len(test_features)} örnek")
    print("Tahminler yapılıyor...\n")
    
    y_test = []
    y_pred = []
    
    for feature in test_features:
        true_label = feature['event_type']
        predicted_label, _ = classifier.predict(feature)
        
        y_test.append(classifier.class_to_idx[true_label])
        y_pred.append(classifier.class_to_idx[predicted_label])
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Gerçek sınıf isimlerini al
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    class_names = [classifier.idx_to_class[idx] for idx in unique_classes]
    
    # Confusion matrix göster
    create_confusion_matrix_display(y_test, y_pred, class_names)
    
    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred, 
                              labels=unique_classes,
                              target_names=class_names, 
                              zero_division=0))
    
    return 0


if __name__ == "__main__":
    exit(main())

