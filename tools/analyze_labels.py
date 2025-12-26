"""
Etiketli verileri analiz et ve istatistikleri göster
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List

def analyze_labels(label_file: Path):
    """Etiket dosyasını analiz et"""
    with open(label_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = data.get('labels', [])
    video_info = data.get('video_info', {})
    
    print("=" * 60)
    print("ETIKET ANALIZI")
    print("=" * 60)
    
    # Video bilgileri
    print(f"\nVideo Bilgileri:")
    print(f"  Dosya: {data.get('video_path', 'N/A')}")
    print(f"  Sure: {video_info.get('duration', 0):.2f} saniye")
    print(f"  FPS: {video_info.get('fps', 0):.2f}")
    print(f"  Frame sayisi: {video_info.get('frame_count', 0)}")
    
    # Etiket istatistikleri
    print(f"\nEtiket Istatistikleri:")
    print(f"  Toplam etiket: {len(labels)}")
    
    # Olay tipine göre sayım
    event_types = Counter([label['event_type'] for label in labels])
    print(f"\nOlay Tipine Gore Dagilim:")
    for event_type, count in event_types.most_common():
        print(f"  {event_type.upper()}: {count}")
    
    # Zaman dağılımı
    print(f"\nZaman Dagilimi:")
    if labels:
        start_times = [label['start_time'] for label in labels]
        end_times = [label['end_time'] for label in labels]
        durations = [label['end_time'] - label['start_time'] for label in labels]
        
        print(f"  Ilk olay: {min(start_times):.2f}s")
        print(f"  Son olay: {max(end_times):.2f}s")
        print(f"  Ortalama olay suresi: {sum(durations)/len(durations):.2f}s")
        print(f"  En kisa olay: {min(durations):.2f}s")
        print(f"  En uzun olay: {max(durations):.2f}s")
    
    # Frame dağılımı
    print(f"\nFrame Dagilimi:")
    if labels:
        start_frames = [label['start_frame'] for label in labels]
        end_frames = [label['end_frame'] for label in labels]
        frame_durations = [label['end_frame'] - label['start_frame'] for label in labels]
        
        print(f"  Ilk frame: {min(start_frames)}")
        print(f"  Son frame: {max(end_frames)}")
        print(f"  Ortalama frame sayisi: {sum(frame_durations)/len(frame_durations):.1f}")
        print(f"  En kisa: {min(frame_durations)} frame")
        print(f"  En uzun: {max(frame_durations)} frame")
    
    # Detaylı liste
    print(f"\nDetayli Liste:")
    print(f"{'No':<4} {'Tip':<8} {'Baslangic':<12} {'Bitis':<12} {'Sure':<8} {'Frame':<15}")
    print("-" * 60)
    for i, label in enumerate(labels, 1):
        duration = label['end_time'] - label['start_time']
        frame_range = f"{label['start_frame']}-{label['end_frame']}"
        print(f"{i:<4} {label['event_type']:<8} {label['start_time']:>10.2f}s {label['end_time']:>10.2f}s "
              f"{duration:>6.2f}s {frame_range:<15}")
    
    print("\n" + "=" * 60)
    
    return {
        'total_labels': len(labels),
        'event_types': dict(event_types),
        'video_info': video_info
    }


if __name__ == "__main__":
    label_file = Path("data/labels/nba_test_video_labels.json")
    
    if not label_file.exists():
        print(f"HATA: Etiket dosyasi bulunamadi: {label_file}")
    else:
        analyze_labels(label_file)

