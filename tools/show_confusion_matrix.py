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
import matplotlib.pyplot as plt
import seaborn as sns

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def load_model_and_features(model_path: Path, features_path: Path):
    """Model ve feature'ları yükle"""
    # Modeli yükle (classmethod olarak çağrılmalı)
    classifier = EventClassifier.load(str(model_path))
    
    # Feature'ları yükle
    with open(features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return classifier, data['features']


def create_confusion_matrix_visualization(y_test, y_pred, class_names, output_path=None):
    """Confusion matrix'i görsel olarak oluştur (fotoğraftaki gibi)"""
    # Sadece test setinde bulunan sınıfları kullan
    unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
    class_indices = sorted(unique_test_classes.tolist())
    
    # Sadece mevcut sınıflar için confusion matrix oluştur
    cm = confusion_matrix(y_test, y_pred, labels=class_indices)
    filtered_class_names = [class_names[i] for i in class_indices]
    
    # Türkçe sınıf isimleri
    turkish_names = {'basket': 'Basket', 'pas': 'Pas', 'blok': 'Blok'}
    display_names = [turkish_names.get(name, name.capitalize()) for name in filtered_class_names]
    
    # Figür oluştur
    plt.figure(figsize=(10, 8))
    
    # Heatmap oluştur
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=display_names, yticklabels=display_names,
                cbar_kws={'label': 'Adet'}, linewidths=1, linecolor='gray')
    
    plt.title('Karmaşıklık Matrisi', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Gerçek', fontsize=12, fontweight='bold')
    plt.xlabel('Tahmin Edilen', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Kaydet veya göster
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nGörsel kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return cm


def create_confusion_matrix_display(y_test, y_pred, class_names):
    """Confusion matrix'i daha okunabilir formatta göster"""
    # Sadece test setinde bulunan sınıfları kullan
    unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
    class_indices = sorted(unique_test_classes.tolist())
    cm = confusion_matrix(y_test, y_pred, labels=class_indices)
    filtered_class_names = [class_names[i] for i in class_indices]
    
    # DataFrame olarak oluştur
    cm_df = pd.DataFrame(cm, 
                        index=[f'Gerçek: {name}' for name in filtered_class_names],
                        columns=[f'Tahmin: {name}' for name in filtered_class_names])
    
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
    for i, class_name in enumerate(filtered_class_names):
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
            for j, other_class in enumerate(filtered_class_names):
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
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Görsel çıktı dosyası yolu (örn: confusion_matrix.png)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Sadece görsel oluştur, konsola yazdırma'
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
    
    # Gerçek sınıf isimlerini al (tüm sınıfları dahil et)
    all_class_indices = sorted(classifier.class_to_idx.values())
    class_names = [classifier.idx_to_class[idx] for idx in all_class_indices]
    
    # Confusion matrix görselini oluştur
    output_path = args.output
    if not output_path:
        # Varsayılan çıktı yolu
        output_dir = Path("data/results/metrics_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "confusion_matrix.png"
    
    print("\nGörsel confusion matrix oluşturuluyor...")
    cm = create_confusion_matrix_visualization(y_test, y_pred, class_names, str(output_path))
    
    if not args.no_display:
        # Konsola yazdır
        create_confusion_matrix_display(y_test, y_pred, class_names)
        
        # Classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        # Sadece test setinde bulunan sınıflar için report
        unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
        report_class_indices = sorted(unique_test_classes.tolist())
        report_class_names = [class_names[i] for i in report_class_indices]
        
        print(classification_report(y_test, y_pred, 
                                  labels=report_class_indices,
                                  target_names=report_class_names, 
                                  zero_division=0))
    
    return 0


if __name__ == "__main__":
    exit(main())

