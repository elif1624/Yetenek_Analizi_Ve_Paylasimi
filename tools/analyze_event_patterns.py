"""
Etiketlerden pattern çıkar
Basket ve pas olaylarının özelliklerini analiz et
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict

def load_labels_and_analysis(label_file: Path, analysis_file: Path):
    """Etiketleri ve analiz sonuçlarını yükle"""
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    return labels_data, analysis_data


def extract_event_features(labels: List[Dict], analysis: Dict):
    """Etiketlerden feature çıkar"""
    
    # Frame bazlı tracking verileri
    frame_tracked = {
        fr['frame_number']: fr.get('tracked_players', [])
        for fr in analysis.get('frame_results', [])
    }
    
    frame_ball = {
        fr['frame_number']: fr.get('ball_position')
        for fr in analysis.get('frame_results', [])
    }
    
    fps = analysis.get('video_metadata', {}).get('fps', 30.0)
    extraction_fps = analysis.get('analysis_params', {}).get('extraction_fps', 3.0)
    
    features_by_type = defaultdict(list)
    
    for label in labels:
        event_type = label['event_type']
        start_frame = label['start_frame']
        end_frame = label['end_frame']
        duration = label['end_time'] - label['start_time']
        
        # Bu olay için tracking verilerini topla
        event_frames = []
        event_players = []
        event_ball_positions = []
        
        # Gerçek frame numaralarına çevir (extraction FPS'den)
        for frame_num in range(start_frame, end_frame + 1):
            # Extraction frame'den gerçek frame'e çevir
            time_sec = frame_num / extraction_fps
            real_frame = int(time_sec * fps)
            
            if real_frame in frame_tracked:
                event_frames.append(real_frame)
                event_players.extend(frame_tracked[real_frame])
                if real_frame in frame_ball and frame_ball[real_frame]:
                    event_ball_positions.append(frame_ball[real_frame])
        
        # Feature'ları hesapla
        features = {
            'duration': duration,
            'frame_count': end_frame - start_frame,
            'num_players': len(set(p.get('track_id') for p in event_players)),
            'has_ball_data': len(event_ball_positions) > 0
        }
        
        # Oyuncu pozisyonları
        if event_players:
            positions = []
            for player in event_players:
                if player.get('position'):
                    positions.append(player['position'])
            
            if positions:
                # Mesafe analizi
                if len(positions) >= 2:
                    distances = []
                    for i in range(1, len(positions)):
                        dx = positions[i][0] - positions[i-1][0]
                        dy = positions[i][1] - positions[i-1][1]
                        dist = np.sqrt(dx**2 + dy**2)
                        distances.append(dist)
                    
                    features['avg_movement'] = np.mean(distances) if distances else 0
                    features['max_movement'] = np.max(distances) if distances else 0
                else:
                    features['avg_movement'] = 0
                    features['max_movement'] = 0
        
        # Top pozisyonları
        if event_ball_positions:
            if len(event_ball_positions) >= 2:
                # Top hareketi
                ball_distances = []
                for i in range(1, len(event_ball_positions)):
                    dx = event_ball_positions[i][0] - event_ball_positions[i-1][0]
                    dy = event_ball_positions[i][1] - event_ball_positions[i-1][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    ball_distances.append(dist)
                
                features['ball_avg_speed'] = np.mean(ball_distances) if ball_distances else 0
                features['ball_max_speed'] = np.max(ball_distances) if ball_distances else 0
            else:
                features['ball_avg_speed'] = 0
                features['ball_max_speed'] = 0
        
        features_by_type[event_type].append(features)
    
    return features_by_type


def analyze_patterns(features_by_type: Dict[str, List[Dict]]):
    """Pattern'leri analiz et ve özetle"""
    
    print("=" * 60)
    print("OLAY PATTERN ANALİZİ")
    print("=" * 60)
    
    for event_type, features_list in features_by_type.items():
        if not features_list:
            continue
        
        print(f"\n{event_type.upper()} Olayları ({len(features_list)} adet):")
        print("-" * 60)
        
        # Süre analizi
        durations = [f['duration'] for f in features_list]
        print(f"Süre:")
        print(f"  Ortalama: {np.mean(durations):.2f}s")
        print(f"  Min: {np.min(durations):.2f}s")
        print(f"  Max: {np.max(durations):.2f}s")
        print(f"  Std: {np.std(durations):.2f}s")
        
        # Frame sayısı
        frame_counts = [f['frame_count'] for f in features_list]
        print(f"\nFrame Sayısı:")
        print(f"  Ortalama: {np.mean(frame_counts):.1f}")
        print(f"  Min: {int(np.min(frame_counts))}")
        print(f"  Max: {int(np.max(frame_counts))}")
        
        # Oyuncu hareketi
        movements = [f.get('avg_movement', 0) for f in features_list if 'avg_movement' in f]
        if movements:
            print(f"\nOyuncu Hareketi (Ortalama):")
            print(f"  Ortalama: {np.mean(movements):.1f} piksel/frame")
            print(f"  Min: {np.min(movements):.1f}")
            print(f"  Max: {np.max(movements):.1f}")
        
        # Top hızı
        ball_speeds = [f.get('ball_avg_speed', 0) for f in features_list if 'ball_avg_speed' in f and f.get('ball_avg_speed', 0) > 0]
        if ball_speeds:
            print(f"\nTop Hızı (Ortalama):")
            print(f"  Ortalama: {np.mean(ball_speeds):.1f} piksel/frame")
            print(f"  Min: {np.min(ball_speeds):.1f}")
            print(f"  Max: {np.max(ball_speeds):.1f}")
        
        # Top verisi var mı?
        has_ball = sum(1 for f in features_list if f.get('has_ball_data', False))
        print(f"\nTop Verisi:")
        print(f"  Top verisi olan: {has_ball}/{len(features_list)} ({has_ball/len(features_list)*100:.1f}%)")
    
    # Öneriler
    print("\n" + "=" * 60)
    print("ÖNERİLER")
    print("=" * 60)
    
    for event_type, features_list in features_by_type.items():
        if not features_list:
            continue
        
        durations = [f['duration'] for f in features_list]
        avg_duration = np.mean(durations)
        
        print(f"\n{event_type.upper()} için öneriler:")
        print(f"  - Ortalama süre: {avg_duration:.2f}s")
        print(f"  - Minimum süre threshold: {avg_duration * 0.5:.2f}s")
        print(f"  - Maksimum süre threshold: {avg_duration * 2.0:.2f}s")
        
        movements = [f.get('avg_movement', 0) for f in features_list if 'avg_movement' in f]
        if movements:
            avg_movement = np.mean(movements)
            print(f"  - Ortalama hareket: {avg_movement:.1f} piksel/frame")
            print(f"  - Hareket threshold: {avg_movement * 0.5:.1f} - {avg_movement * 2.0:.1f}")


def main():
    label_file = Path("data/labels/nba_test_video_labels.json")
    analysis_file = Path("data/results/nba_test_video_final_analysis.json")
    
    if not label_file.exists():
        print(f"HATA: Etiket dosyasi bulunamadi: {label_file}")
        return
    
    if not analysis_file.exists():
        print(f"HATA: Analiz dosyasi bulunamadi: {analysis_file}")
        return
    
    print("Etiketler ve analiz sonuçlari yükleniyor...")
    labels_data, analysis_data = load_labels_and_analysis(label_file, analysis_file)
    
    labels = labels_data.get('labels', [])
    print(f"Toplam {len(labels)} etiket yüklendi")
    
    print("\nPattern'ler çıkarılıyor...")
    features_by_type = extract_event_features(labels, analysis_data)
    
    analyze_patterns(features_by_type)


if __name__ == "__main__":
    main()

