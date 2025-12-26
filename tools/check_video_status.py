"""
Video analiz durumunu kontrol et
"""

from pathlib import Path
import json


def check_video_status(video_name: str):
    """Video analiz durumunu kontrol et"""
    video_path = Path("data/input") / video_name
    
    if not video_path.exists():
        print(f"[HATA] Video bulunamadi: {video_path}")
        return
    
    print(f"[OK] Video var: {video_path}")
    print(f"     Boyut: {video_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Analiz dosyasını kontrol et
    results_file = Path("data/results") / f"{video_path.stem}_final_analysis.json"
    
    if results_file.exists():
        print(f"\n[OK] Analiz tamamlandi: {results_file}")
        
        # İstatistikleri göster
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        stats = results.get('statistics', {})
        print(f"\n[ISTATISTIK]")
        print(f"  Toplam frame: {stats.get('total_frames_processed', 0)}")
        print(f"  Detection olan frame: {stats.get('frames_with_detections', 0)}")
        print(f"  Toplam detection: {stats.get('total_detections', 0)}")
        print(f"  Benzersiz track: {stats.get('unique_tracks', 0)}")
        print(f"  Uzun track (>=20): {stats.get('long_tracks', 0)}")
        
        # Etiket dosyasını kontrol et
        label_file = Path("data/labels") / f"{video_path.stem}_labels.json"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            event_count = len(labels.get('events', []))
            print(f"\n[ETIKET DURUMU]")
            print(f"  Toplam etiket: {event_count}")
            
            events_by_type = {}
            for event in labels.get('events', []):
                event_type = event.get('event_type', 'unknown')
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            for event_type, count in events_by_type.items():
                print(f"  {event_type.upper()}: {count}")
        else:
            print(f"\n[UYARI] Etiket dosyasi yok: {label_file}")
            print(f"  Etiketleme yapmak icin:")
            print(f"  python tools/labeling_tool.py {video_path}")
    else:
        print(f"\n[BEKLEMEDE] Analiz henuz tamamlanmadi")
        print(f"  Beklenen dosya: {results_file}")
        print(f"  Analiz calisiyor olabilir...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    else:
        video_name = "video_zubrmyzZ0hM.mp4"
    
    check_video_status(video_name)

