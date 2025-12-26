"""
Eğitilmiş model ile tahmin testi
SADECE TEST SET ile test yapar (train set'i kullanmaz)
"""

import json
import argparse
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier


def main():
    parser = argparse.ArgumentParser(description="Eğitilmiş model ile tahmin testi (SADECE TEST SET)")
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/event_classifier.pkl",
        help="Model dosyası yolu (default: data/models/event_classifier.pkl)"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/dataset/features.json",
        help="Test için feature dosyası (default: data/dataset/features.json)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set oranı (eğitimle aynı olmalı, default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (eğitimle aynı olmalı, default: 42)"
    )
    
    args = parser.parse_args()
    
    # Model yükle
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        print(f"Önce model eğitin:")
        print(f"  python tools/train_model.py")
        return 1
    
    print(f"Model yükleniyor: {model_path}")
    classifier = EventClassifier.load(str(model_path))
    print("Model yüklendi.\n")
    
    # Feature'ları yükle
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"HATA: Feature dosyası bulunamadı: {features_path}")
        return 1
    
    with open(features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_features = data['features']
    print(f"Toplam {len(all_features)} örnek yüklendi.")
    
    # Train-test split (eğitimle aynı şekilde)
    # Event type'lara göre label oluştur
    event_types = [f['event_type'] for f in all_features]
    class_to_idx = {cls: idx for idx, cls in enumerate(classifier.classes)}
    y_all = np.array([class_to_idx.get(et, 0) for et in event_types])
    
    # Aynı split'i kullan (eğitimle aynı random_state)
    _, test_indices = train_test_split(
        range(len(all_features)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_all
    )
    
    test_features = [all_features[i] for i in test_indices]
    
    print(f"Test set: {len(test_features)} örnek (eğitimde kullanılmamış)")
    print(f"Train set: {len(all_features) - len(test_features)} örnek (bu test edilmeyecek)\n")
    
    # Tahmin yap (SADECE TEST SET)
    correct = 0
    total = len(test_features)
    
    print(f"{'='*70}")
    print("TEST SET TAHMIN SONUÇLARI (Eğitimde kullanılmamış veriler)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Gerçek':<10} {'Tahmin':<10} {'Güven':<8} {'Doğru':<8}")
    print("-" * 70)
    
    for i, feature in enumerate(test_features, 1):
        true_label = feature['event_type']
        predicted_label, confidence = classifier.predict(feature)
        
        is_correct = (true_label == predicted_label)
        if is_correct:
            correct += 1
        
        status = "OK" if is_correct else "X"
        print(f"{i:<4} {true_label:<10} {predicted_label:<10} {confidence:.3f}    {status}")
    
    accuracy = correct / total
    
    print(f"{'='*70}")
    print(f"\nTest Set Sonuçları:")
    print(f"  Toplam: {total} örnek (eğitimde kullanılmamış)")
    print(f"  Doğru: {correct} örnek")
    print(f"  Yanlış: {total - correct} örnek")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"{'='*70}\n")
    
    # Event bazlı analiz
    print("EVENT BAZLI ANALİZ (Test Set)")
    print(f"{'='*70}")
    
    event_stats = {}
    for feature in test_features:
        event_type = feature['event_type']
        if event_type not in event_stats:
            event_stats[event_type] = {'total': 0, 'correct': 0}
        
        event_stats[event_type]['total'] += 1
        predicted_label, _ = classifier.predict(feature)
        if predicted_label == event_type:
            event_stats[event_type]['correct'] += 1
    
    for event_type, stats in sorted(event_stats.items()):
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"{event_type.upper():<10} {stats['correct']}/{stats['total']} = {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
