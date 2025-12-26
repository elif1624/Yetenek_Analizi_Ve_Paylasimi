"""
Data Augmentation (Veri Çoğaltma) Aracı
Mevcut feature'lara noise ekleyerek veri setini genişletir
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import copy


def augment_feature(feature: Dict, noise_level: float = 0.05) -> Dict:
    """
    Bir feature'a küçük noise ekleyerek yeni bir örnek oluştur
    
    Args:
        feature: Orijinal feature dictionary
        noise_level: Noise seviyesi (0.05 = %5 değişiklik)
    
    Returns:
        Augment edilmiş yeni feature
    """
    # Kopya oluştur
    augmented = copy.deepcopy(feature)
    
    # Metadata'ları koru (bunlar değişmez)
    exclude_keys = [
        'event_type',  # Label değişmez
        'start_time', 'end_time',  # Zaman bilgisi değişmez
        'frame_count',  # Frame sayısı değişmez
        'ball_y_trend',  # Kategorik
        'has_ball_data'  # Boolean
    ]
    
    # Sayısal değerlere noise ekle
    for key, value in augmented.items():
        if key in exclude_keys:
            continue
        
        if isinstance(value, (int, float)) and value != 0:
            # Küçük rastgele noise ekle
            noise = np.random.normal(0, abs(value) * noise_level)
            augmented[key] = value + noise
            
            # Negatif değerleri önle (bazı feature'lar negatif olamaz)
            if key in ['num_players', 'num_frames_with_players', 'frame_count']:
                augmented[key] = max(0, augmented[key])
            elif 'avg' in key or 'std' in key or 'min' in key or 'max' in key:
                # Pozisyon ve istatistik feature'ları için abs kullan
                augmented[key] = abs(augmented[key])
    
    return augmented


def augment_dataset(features: List[Dict], augmentation_factor: float = 1.0, noise_level: float = 0.05) -> List[Dict]:
    """
    Veri setini augment et
    
    Args:
        features: Orijinal feature listesi
        augmentation_factor: Her örnek için kaç yeni örnek oluşturulacak (1.0 = her örnek için 1 yeni)
        noise_level: Noise seviyesi
    
    Returns:
        Augment edilmiş feature listesi (orijinal + yeni örnekler)
    """
    augmented_features = features.copy()  # Orijinal örnekleri koru
    num_new = int(len(features) * augmentation_factor)
    
    print(f"Augmentation başlatılıyor...")
    print(f"  Orijinal örnek sayısı: {len(features)}")
    print(f"  Yeni örnek sayısı: {num_new}")
    print(f"  Noise seviyesi: {noise_level*100:.1f}%")
    print(f"  Toplam hedef: {len(features) + num_new}")
    
    # Rastgele örnekler seç ve augment et
    for _ in range(num_new):
        # Rastgele bir örnek seç
        original = np.random.choice(features)
        # Augment et
        augmented = augment_feature(original, noise_level)
        augmented_features.append(augmented)
    
    print(f"  Oluşturulan toplam örnek: {len(augmented_features)}")
    
    return augmented_features


def main():
    parser = argparse.ArgumentParser(description='Feature Augmentation')
    parser.add_argument(
        '--input',
        type=str,
        default='data/dataset/features.json',
        help='Girdi feature dosyası'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/dataset/features_augmented.json',
        help='Çıktı feature dosyası'
    )
    parser.add_argument(
        '--factor',
        type=float,
        default=1.0,
        help='Augmentation faktörü (1.0 = her örnek için 1 yeni örnek, 2.0 = her örnek için 2 yeni)'
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.05,
        help='Noise seviyesi (0.05 = %5 değişiklik, 0.1 = %10 değişiklik)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Random seed
    np.random.seed(args.seed)
    
    # Dosyaları yükle
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"HATA: Girdi dosyası bulunamadı: {input_path}")
        return 1
    
    print(f"Feature dosyası yükleniyor: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data['features']
    original_count = len(features)
    
    # Event dağılımını göster
    print(f"\nOrijinal veri seti:")
    event_counts = {}
    for f in features:
        event_type = f['event_type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count}")
    
    # Augment et
    print(f"\n{'='*70}")
    augmented_features = augment_dataset(features, args.factor, args.noise)
    
    # Yeni event dağılımını göster
    print(f"\nAugmented veri seti:")
    new_event_counts = {}
    for f in augmented_features:
        event_type = f['event_type']
        new_event_counts[event_type] = new_event_counts.get(event_type, 0) + 1
    
    for event_type, count in new_event_counts.items():
        print(f"  {event_type}: {count} (artış: +{count - event_counts[event_type]})")
    
    # Kaydet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'total_samples': len(augmented_features),
        'original_samples': original_count,
        'augmented_samples': len(augmented_features) - original_count,
        'augmentation_factor': args.factor,
        'noise_level': args.noise,
        'features': augmented_features
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("AUGMENTATION TAMAMLANDI")
    print(f"{'='*70}")
    print(f"Orijinal örnek: {original_count}")
    print(f"Yeni örnek: {len(augmented_features) - original_count}")
    print(f"Toplam örnek: {len(augmented_features)}")
    print(f"Artış: {((len(augmented_features) - original_count) / original_count * 100):.1f}%")
    print(f"\nKaydedildi: {output_path}")
    print(f"\nModel eğitimi için:")
    print(f"  python tools/train_model.py --features {output_path}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    exit(main())



