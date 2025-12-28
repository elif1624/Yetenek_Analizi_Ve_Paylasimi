"""
Sınıflandırma Metrikleri Tablosu Oluşturma
Her kategori için Precision, Recall, F1-Score değerlerini görsel tablo olarak oluşturur
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier
from sklearn.model_selection import train_test_split

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def create_metrics_table(y_test, y_pred, class_names, class_indices, output_path=None):
    """
    Her kategori için Precision, Recall, F1-Score tablosu oluştur
    
    Args:
        y_test: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri listesi
        class_indices: Sınıf index'leri (labels için)
        output_path: Çıktı dosyası yolu
    """
    # Her sınıf için metrikleri hesapla (sadece mevcut sınıflar için)
    precision_scores = precision_score(y_test, y_pred, labels=class_indices, 
                                       average=None, zero_division=0)
    recall_scores = recall_score(y_test, y_pred, labels=class_indices, 
                                 average=None, zero_division=0)
    f1_scores = f1_score(y_test, y_pred, labels=class_indices, 
                        average=None, zero_division=0)
    
    # Genel doğruluk
    overall_accuracy = accuracy_score(y_test, y_pred) * 100
    
    # Türkçe sınıf isimleri
    turkish_names = {'basket': 'Basket', 'pas': 'Pas', 'blok': 'Blok'}
    display_names = [turkish_names.get(name, name.capitalize()) for name in class_names]
    
    # Tablo oluştur
    fig, ax = plt.subplots(figsize=(12, len(class_names) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Veri hazırla
    table_data = []
    for i, class_name in enumerate(display_names):
        table_data.append([
            class_name,
            f'{precision_scores[i]:.2f}',
            f'{recall_scores[i]:.2f}',
            f'{f1_scores[i]:.2f}'
        ])
    
    # Tablo başlıkları
    headers = ['Kategori', 'Kesinlik\n(Precision)', 'Duyarlılık\n(Recall)', 'F1-Skor\n(F1-Score)']
    
    # Tablo oluştur
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.1, 1, 0.9])
    
    # Tablo stilini ayarla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Başlık satırını vurgula
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Diğer satırları vurgula (alternatif renkler)
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#D9D9D9')
    
    # Genel doğruluk ekle
    title_text = f'Şekil X. Modelin test veri seti üzerinde değerlendirilmesinden ortaya çıkan sınıflandırma metrikleri.\n\nGenel Doğruluk (Overall Accuracy): % {overall_accuracy:.1f}'
    
    plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrikler tablosu kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Konsola da yazdır
    print("\n" + "="*70)
    print("SINIFLANDIRMA METRİKLERİ")
    print("="*70)
    print(f"{'Kategori':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    for i, class_name in enumerate(display_names):
        print(f"{class_name:<15} {precision_scores[i]:<12.2f} {recall_scores[i]:<12.2f} {f1_scores[i]:<12.2f}")
    print("-"*70)
    print(f"{'Genel Doğruluk':<15} {overall_accuracy:.1f}%")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Sınıflandırma Metrikleri Tablosu Oluştur')
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
        help='Test split oranı'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Görsel çıktı dosyası yolu'
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
    classifier = EventClassifier.load(str(model_path))
    
    # Feature'ları yükle
    with open(features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data['features']
    
    # Feature'ları hazırla
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
    
    # Sadece test setinde bulunan sınıfları kullan
    unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
    class_indices = sorted(unique_test_classes.tolist())
    filtered_class_names = [class_names[i] for i in class_indices]
    
    # Metrikler tablosunu oluştur
    output_path = args.output
    if not output_path:
        output_dir = Path("data/results/metrics_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "classification_metrics_table.png"
    
    print("\nMetrikler tablosu oluşturuluyor...")
    create_metrics_table(y_test, y_pred, filtered_class_names, class_indices, str(output_path))
    
    return 0


if __name__ == "__main__":
    exit(main())

