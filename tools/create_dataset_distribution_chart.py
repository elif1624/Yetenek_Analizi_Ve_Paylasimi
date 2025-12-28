"""
Veri Seti Dağılım Grafiği Oluşturma
Train/Test/Validation dağılımı ve kategori dağılımını gösterir
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import sys
import numpy as np

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def create_dataset_distribution_visualization(features_file: Path, test_split: float = 0.2, output_path=None):
    """
    Veri seti dağılım grafiği oluştur (pasta grafiği + tablo)
    """
    # Verileri yükle
    with open(features_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data['features']
    total_samples = len(features)
    
    # Train/Test/Validation split hesapla
    train_size = int(total_samples * (1 - test_split))
    test_size = total_samples - train_size
    
    # Validation split (train'den %20 alıyoruz genelde, ama burada train/test var)
    # Bu örnekte validation = test olarak gösteriyoruz (gerçekte train içinden alınır)
    validation_size = 0  # Train-test split kullandığımız için validation ayrı değil
    train_only_size = train_size
    
    # Kategori dağılımı
    event_counts = Counter([f['event_type'] for f in features])
    
    # Türkçe kategori isimleri
    category_names = {
        'basket': 'Basket',
        'pas': 'Pas',
        'blok': 'Blok'
    }
    
    # Figür oluştur (2 subplot: pasta grafiği + tablo)
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Pasta Grafiği (üstte)
    ax1 = plt.subplot(2, 1, 1)
    
    # Veri split değerleri
    sizes = [train_only_size, test_size]
    labels = ['Eğitim', 'Test']
    colors = ['#FF6B9D', '#4ECDC4']  # Magenta, Light Blue (görseldeki gibi)
    
    # Eğer validation varsa ekle
    if validation_size > 0:
        sizes.insert(1, validation_size)
        labels.insert(1, 'Doğrulama')
        colors.insert(1, '#95E1D3')  # Green
    
    # Pasta grafiği
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Yüzde metinlerini daha okunabilir yap
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # Başlık
    title_text = f'Şekil X. Veri setinden rastgele elde edilen {total_samples} örneğin eğitim ve test şeklinde ayrılması'
    ax1.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # Legend (eğer gerekirse)
    # ax1.legend(wedges, [f'{label}: {size}' for label, size in zip(labels, sizes)], 
    #           title="Dağılım", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # 2. Tablo (altta)
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('tight')
    ax2.axis('off')
    
    # Tablo verileri
    table_data = []
    sorted_categories = sorted(event_counts.items(), key=lambda x: x[0])
    
    for category, count in sorted_categories:
        category_tr = category_names.get(category, category.capitalize())
        percentage = (count / total_samples) * 100
        table_data.append([category_tr, count, f'{percentage:.1f}%'])
    
    # Toplam satırı
    table_data.append(['TOPLAM', total_samples, '100.0%'])
    
    # Tablo başlıkları
    headers = ['Kategori', 'Örnek Sayısı', 'Yüzde']
    
    # Tablo oluştur
    table = ax2.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Tablo stilini ayarla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Başlık satırını vurgula
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.08)
    
    # Son satırı (TOPLAM) vurgula
    last_row = len(table_data)
    for j in range(len(headers)):
        cell = table[(last_row, j)]
        cell.set_facecolor('#D9E1F2')
        cell.set_text_props(weight='bold')
        cell.set_height(0.08)
    
    # Diğer satırları vurgula (alternatif renkler)
    for i in range(1, last_row):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#D9D9D9')
            cell.set_height(0.06)
    
    # Tablo başlığı
    table_title = f'Tablo X. Ayrılan {total_samples} örnekli veri setinin kategorileri ve örnek sayıları.'
    ax2.set_title(table_title, fontsize=14, fontweight='bold', pad=20, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Veri seti dağılım grafiği kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Konsola da yazdır
    print("\n" + "="*70)
    print("VERİ SETİ DAĞILIMI")
    print("="*70)
    print(f"\nToplam örnek: {total_samples}")
    print(f"\nTrain/Test Dağılımı:")
    print(f"  Eğitim: {train_only_size} (%{(train_only_size/total_samples)*100:.1f})")
    print(f"  Test: {test_size} (%{(test_size/total_samples)*100:.1f})")
    print(f"\nKategori Dağılımı:")
    print("-"*70)
    print(f"{'Kategori':<15} {'Sayı':<10} {'Yüzde':<10}")
    print("-"*70)
    for category, count in sorted_categories:
        category_tr = category_names.get(category, category.capitalize())
        percentage = (count / total_samples) * 100
        print(f"{category_tr:<15} {count:<10} {percentage:.1f}%")
    print("-"*70)
    print(f"{'TOPLAM':<15} {total_samples:<10} 100.0%")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Veri Seti Dağılım Grafiği Oluştur')
    parser.add_argument(
        '--features',
        type=str,
        default='data/dataset/features.json',
        help='Feature dosyası yolu'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test split oranı'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Görsel çıktı dosyası yolu'
    )
    
    args = parser.parse_args()
    
    features_path = Path(args.features)
    
    if not features_path.exists():
        print(f"HATA: Feature dosyası bulunamadı: {features_path}")
        return 1
    
    # Çıktı dosyası
    output_path = args.output
    if not output_path:
        output_dir = Path("data/results/metrics_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset_distribution.png"
    
    print("Veri seti dağılım grafiği oluşturuluyor...")
    create_dataset_distribution_visualization(features_path, args.test_split, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())


