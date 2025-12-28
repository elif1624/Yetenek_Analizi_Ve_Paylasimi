"""
Feature extraction çıktısını görüntüleme aracı
"""

import json
import argparse
from pathlib import Path
from collections import Counter

def view_features(features_file: str):
    """Feature dosyasını görüntüle"""
    
    with open(features_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 70)
    print("FEATURE EXTRACTION ÇIKTISI")
    print("=" * 70)
    print(f"\nDosya: {features_file}")
    print(f"Toplam örnek sayısı: {data['total_samples']}")
    
    # Event type dağılımı
    event_types = [f['event_type'] for f in data['features']]
    event_counts = Counter(event_types)
    
    print(f"\nEvent dağılımı:")
    for event_type, count in event_counts.items():
        print(f"  {event_type.upper()}: {count} örnek")
    
    # Feature listesi
    if data['features']:
        print(f"\nFeature sayısı: {len(data['features'][0])} feature per örnek")
        print(f"\nFeature listesi:")
        for key in sorted(data['features'][0].keys()):
            print(f"  - {key}")
    
    # İlk 3 örnek detayları
    print(f"\n{'=' * 70}")
    print("İLK 3 ÖRNEK DETAYLARI")
    print("=" * 70)
    
    for i, feature in enumerate(data['features'][:3], 1):
        print(f"\n--- Örnek {i}: {feature['event_type'].upper()} ---")
        print(f"  Süre: {feature['duration']:.2f} saniye")
        print(f"  Frame sayısı: {feature['frame_count']}")
        print(f"  Zaman: {feature['start_time']:.2f}s - {feature['end_time']:.2f}s")
        print(f"  Oyuncu sayısı: {feature['num_players']}")
        print(f"  Oyuncu hareket (avg): {feature.get('player_avg_movement', 0):.2f}")
        print(f"  Top hızı (avg): {feature.get('ball_avg_speed', 0):.2f}")
        print(f"  Oyuncu-top mesafe (avg): {feature.get('player_ball_avg_distance', 0):.2f}")
        print(f"  Top verisi var mı: {feature.get('has_ball_data', False)}")
    
    # İstatistikler
    print(f"\n{'=' * 70}")
    print("İSTATİSTİKLER")
    print("=" * 70)
    
    # Her event type için ortalama değerler
    for event_type in event_counts.keys():
        event_features = [f for f in data['features'] if f['event_type'] == event_type]
        
        print(f"\n{event_type.upper()} ({len(event_features)} örnek):")
        print(f"  Ortalama süre: {sum(f['duration'] for f in event_features) / len(event_features):.2f} saniye")
        print(f"  Ortalama oyuncu sayısı: {sum(f['num_players'] for f in event_features) / len(event_features):.1f}")
        
        movements = [f.get('player_avg_movement', 0) for f in event_features if f.get('player_avg_movement', 0) > 0]
        if movements:
            print(f"  Ortalama oyuncu hareketi: {sum(movements) / len(movements):.2f}")
        
        ball_speeds = [f.get('ball_avg_speed', 0) for f in event_features if f.get('ball_avg_speed', 0) > 0]
        if ball_speeds:
            print(f"  Ortalama top hızı: {sum(ball_speeds) / len(ball_speeds):.2f}")
    
    print(f"\n{'=' * 70}")
    print(f"Tüm veriler: {features_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction çıktısını görüntüle")
    parser.add_argument(
        "--features",
        type=str,
        default="data/dataset/features.json",
        help="Feature dosyası yolu (default: data/dataset/features.json)"
    )
    
    args = parser.parse_args()
    
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"HATA: Dosya bulunamadı: {features_path}")
        exit(1)
    
    view_features(str(features_path))






