"""
23 etiketten ML model için feature extraction
SAM3 + Tracking verilerini kullanarak feature vector'lar oluştur
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(label_file: Path, analysis_file: Path):
    """Etiketleri ve analiz sonuçlarını yükle"""
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    return labels_data, analysis_data


def extract_features_for_event(
    label: Dict,
    analysis: Dict,
    video_fps: float,
    extraction_fps: float
) -> Optional[Dict]:
    """
    Tek bir etiket için feature çıkar
    
    Returns:
        Feature dictionary veya None (eğer yeterli veri yoksa)
    """
    # Frame numaraları: Etiketler gerçek video frame, analiz extraction frame kullanıyor
    # Zamanı kullanarak extraction frame hesapla (daha doğru)
    time_start = label['start_time']
    time_end = label['end_time']
    start_frame_label = int(time_start * extraction_fps)  # Extraction frame
    end_frame_label = int(time_end * extraction_fps)  # Extraction frame
    
    # Zamanı hesapla
    time_start = label['start_time']
    time_end = label['end_time']
    
    # Frame bazlı tracking verileri (frame_number extraction frame numarası)
    frame_tracked = {
        fr['frame_number']: fr.get('tracked_players', [])
        for fr in analysis.get('frame_results', [])
    }
    
    frame_ball = {
        fr['frame_number']: fr.get('ball_position')
        for fr in analysis.get('frame_results', [])
    }
    
    # Bu olay için veri topla (extraction frame numaralarını direkt kullan)
    event_frames = []
    event_players = []
    event_ball_positions = []
    
    # Extraction frame numaralarını direkt kullan (analiz sonuçları da extraction frame)
    for extraction_frame in range(start_frame_label, end_frame_label + 1):
        if extraction_frame in frame_tracked:
            event_frames.append(extraction_frame)
            event_players.extend(frame_tracked[extraction_frame])
            if extraction_frame in frame_ball and frame_ball[extraction_frame]:
                event_ball_positions.append((extraction_frame, frame_ball[extraction_frame]))
    
    if len(event_frames) < 3:  # Yeterli veri yok
        return None
    
    # Feature'ları hesapla
    features = {
        # Temel bilgiler
        'event_type': label['event_type'],
        'duration': time_end - time_start,
        'frame_count': end_frame_label - start_frame_label,
        'start_time': time_start,
        'end_time': time_end,
        
        # Oyuncu feature'ları
        'num_players': len(set(p.get('track_id') for p in event_players if p.get('track_id'))),
        'num_frames_with_players': len(event_frames),
    }
    
    # Oyuncu pozisyonları
    if event_players:
        positions = []
        for player in event_players:
            if player.get('position'):
                positions.append(player['position'])
        
        if positions:
            # Pozisyon istatistikleri
            x_positions = [p[0] for p in positions]
            y_positions = [p[1] for p in positions]
            
            features['player_avg_x'] = np.mean(x_positions)
            features['player_avg_y'] = np.mean(y_positions)
            features['player_std_x'] = np.std(x_positions)
            features['player_std_y'] = np.std(y_positions)
            features['player_min_x'] = np.min(x_positions)
            features['player_max_x'] = np.max(x_positions)
            features['player_min_y'] = np.min(y_positions)
            features['player_max_y'] = np.max(y_positions)
            
            # Hareket analizi
            if len(positions) >= 2:
                movements = []
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    movements.append(dist)
                
                features['player_avg_movement'] = np.mean(movements) if movements else 0
                features['player_max_movement'] = np.max(movements) if movements else 0
                features['player_total_movement'] = sum(movements) if movements else 0
            else:
                features['player_avg_movement'] = 0
                features['player_max_movement'] = 0
                features['player_total_movement'] = 0
    
    # Top feature'ları
    if event_ball_positions:
        # Top pozisyonları (frame, position) formatında
        ball_positions = [pos for _, pos in sorted(event_ball_positions)]
        
        if len(ball_positions) >= 2:
            # Top pozisyon istatistikleri
            ball_x = [p[0] for p in ball_positions]
            ball_y = [p[1] for p in ball_positions]
            
            features['ball_avg_x'] = np.mean(ball_x)
            features['ball_avg_y'] = np.mean(ball_y)
            features['ball_std_x'] = np.std(ball_x)
            features['ball_std_y'] = np.std(ball_y)
            
            # Top hareketi
            ball_movements = []
            for i in range(1, len(ball_positions)):
                dx = ball_positions[i][0] - ball_positions[i-1][0]
                dy = ball_positions[i][1] - ball_positions[i-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                ball_movements.append(dist)
            
            features['ball_avg_speed'] = np.mean(ball_movements) if ball_movements else 0
            features['ball_max_speed'] = np.max(ball_movements) if ball_movements else 0
            features['ball_total_movement'] = sum(ball_movements) if ball_movements else 0
            
            # Top yön analizi
            if len(ball_positions) >= 3:
                # Y ekseni değişimi (yukarı/aşağı)
                y_changes = [ball_positions[i][1] - ball_positions[i-1][1] 
                           for i in range(1, len(ball_positions))]
                features['ball_avg_y_change'] = np.mean(y_changes)
                features['ball_y_trend'] = 'up' if np.mean(y_changes) < 0 else 'down'
            else:
                features['ball_avg_y_change'] = 0
                features['ball_y_trend'] = 'unknown'
        else:
            # Tek top pozisyonu
            features['ball_avg_x'] = ball_positions[0][0]
            features['ball_avg_y'] = ball_positions[0][1]
            features['ball_std_x'] = 0
            features['ball_std_y'] = 0
            features['ball_avg_speed'] = 0
            features['ball_max_speed'] = 0
            features['ball_total_movement'] = 0
            features['ball_avg_y_change'] = 0
            features['ball_y_trend'] = 'unknown'
        
        features['has_ball_data'] = True
    else:
        features['has_ball_data'] = False
        # Top verisi yok, default değerler
        features['ball_avg_x'] = 0
        features['ball_avg_y'] = 0
        features['ball_std_x'] = 0
        features['ball_std_y'] = 0
        features['ball_avg_speed'] = 0
        features['ball_max_speed'] = 0
        features['ball_total_movement'] = 0
        features['ball_avg_y_change'] = 0
        features['ball_y_trend'] = 'unknown'
    
    # Oyuncu-Top mesafesi (eğer ikisi de varsa)
    if event_players and event_ball_positions:
        # Her frame için oyuncu-top mesafesi
        distances = []
        for player in event_players:
            if player.get('position'):
                # En yakın top pozisyonunu bul
                player_pos = player['position']
                min_dist = float('inf')
                for _, ball_pos in event_ball_positions:
                    dist = np.sqrt((player_pos[0] - ball_pos[0])**2 + 
                                 (player_pos[1] - ball_pos[1])**2)
                    min_dist = min(min_dist, dist)
                if min_dist != float('inf'):
                    distances.append(min_dist)
        
        if distances:
            features['player_ball_avg_distance'] = np.mean(distances)
            features['player_ball_min_distance'] = np.min(distances)
            features['player_ball_max_distance'] = np.max(distances)
        else:
            features['player_ball_avg_distance'] = 0
            features['player_ball_min_distance'] = 0
            features['player_ball_max_distance'] = 0
    else:
        features['player_ball_avg_distance'] = 0
        features['player_ball_min_distance'] = 0
        features['player_ball_max_distance'] = 0
    
    return features


def extract_all_features(
    labels_file: Path,
    analysis_file: Path
) -> List[Dict]:
    """Tüm etiketler için feature çıkar"""
    
    labels_data, analysis_data = load_data(labels_file, analysis_file)
    
    labels = labels_data.get('labels', [])
    video_info = labels_data.get('video_info', {})
    video_fps = video_info.get('fps', 30.0)
    
    analysis_params = analysis_data.get('analysis_params', {})
    extraction_fps = analysis_params.get('extraction_fps', 3.0)
    
    print(f"Toplam {len(labels)} etiket işleniyor...")
    print(f"Video FPS: {video_fps:.2f}, Extraction FPS: {extraction_fps:.2f}")
    
    all_features = []
    skipped = 0
    
    for i, label in enumerate(labels, 1):
        features = extract_features_for_event(
            label, analysis_data, video_fps, extraction_fps
        )
        
        if features:
            all_features.append(features)
            print(f"  {i}. {label['event_type'].upper()}: Feature çıkarıldı "
                  f"({len(features)} feature)")
        else:
            skipped += 1
            print(f"  {i}. {label['event_type'].upper()}: Yeterli veri yok, atlandı")
    
    print(f"\nToplam: {len(all_features)} feature çıkarıldı, {skipped} atlandı")
    
    return all_features


def save_features(features: List[Dict], output_file: Path):
    """Feature'ları kaydet"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'total_samples': len(features),
        'features': features,
        'feature_names': list(features[0].keys()) if features else []
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nFeature'lar kaydedildi: {output_file}")
    print(f"Toplam {len(features)} örnek, {len(output_data['feature_names'])} feature")


def analyze_features(features: List[Dict]):
    """Feature'ları analiz et ve özetle"""
    
    if not features:
        print("Analiz edilecek feature yok")
        return
    
    print("\n" + "=" * 60)
    print("FEATURE ANALİZİ")
    print("=" * 60)
    
    # Olay tipine göre grupla
    by_type = defaultdict(list)
    for feat in features:
        by_type[feat['event_type']].append(feat)
    
    for event_type, type_features in by_type.items():
        print(f"\n{event_type.upper()} ({len(type_features)} örnek):")
        print("-" * 60)
        
        # Numeric feature'ları analiz et
        numeric_features = [
            'duration', 'player_avg_movement', 'ball_avg_speed',
            'player_ball_avg_distance', 'num_players'
        ]
        
        for feat_name in numeric_features:
            if feat_name in type_features[0]:
                values = [f[feat_name] for f in type_features if f.get(feat_name) is not None]
                if values:
                    print(f"  {feat_name}:")
                    print(f"    Ortalama: {np.mean(values):.2f}")
                    print(f"    Min: {np.min(values):.2f}")
                    print(f"    Max: {np.max(values):.2f}")
                    print(f"    Std: {np.std(values):.2f}")
        
        # Top verisi var mı?
        has_ball = sum(1 for f in type_features if f.get('has_ball_data', False))
        print(f"\n  Top verisi: {has_ball}/{len(type_features)} ({has_ball/len(type_features)*100:.1f}%)")


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML model için feature extraction')
    parser.add_argument('--labels', type=str, default='data/labels/nba_test_video_labels.json',
                       help='Etiket dosyası')
    parser.add_argument('--analysis', type=str, default='data/results/nba_test_video_final_analysis.json',
                       help='Analiz sonuçları dosyası')
    parser.add_argument('--output', type=str, default='data/dataset/features.json',
                       help='Çıktı dosyası')
    
    args = parser.parse_args()
    
    label_file = Path(args.labels)
    analysis_file = Path(args.analysis)
    output_file = Path(args.output)
    
    if not label_file.exists():
        print(f"HATA: Etiket dosyasi bulunamadi: {label_file}")
        sys.exit(1)
    
    if not analysis_file.exists():
        print(f"HATA: Analiz dosyasi bulunamadi: {analysis_file}")
        print(f"Önce video analizi yapin: python analyze_video_final.py")
        sys.exit(1)
    
    # Feature extraction
    print("Feature extraction baslatiliyor...")
    features = extract_all_features(label_file, analysis_file)
    
    if not features:
        print("HATA: Hiç feature çıkarılamadı!")
        sys.exit(1)
    
    # Analiz
    analyze_features(features)
    
    # Kaydet
    save_features(features, output_file)
    
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION TAMAMLANDI")
    print("=" * 60)
    print(f"Toplam {len(features)} örnek hazır")
    print(f"ML model eğitimi için hazır: {output_file}")


if __name__ == "__main__":
    main()

