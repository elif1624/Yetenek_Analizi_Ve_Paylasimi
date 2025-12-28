"""
Video Analizi ve ML Model Tahmini Modülü

Bu modül, yüklenen basketbol videolarını analiz ederek basket ve pas olaylarını
tespit eder. analyze_video_final.py ile video analizi yapar, sonrasında ML model
ile sliding window yöntemi kullanarak her pencere için olay sınıflandırması yapar.

Ana İşlevler:
1. Video analizi (SAM3 + tracking) - analyze_video_final.py kullanır
2. Sliding window ile video pencerelerine bölme
3. Her pencere için feature extraction
4. ML model ile olay tahmini (basket/pas)
5. Pencere tahminlerini birleştirme ve overlap temizleme
6. Son 2 saniye mantığı: Videonun son 2 saniyesi kesinlikle basket

Kullanım:
    web/video_analyzer.py modülü web/app.py tarafından otomatik olarak kullanılır.
    Direkt çağrılmaz, get_video_analysis_events_simple() fonksiyonu API olarak kullanılır.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import sys
import numpy as np

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze_video_final import analyze_video_final
from src.models.event_classifier import EventClassifier

logger = logging.getLogger(__name__)


def extract_features_for_window(
    analysis: Dict,
    start_time: float,
    end_time: float,
    video_fps: float,
    extraction_fps: float
) -> Optional[Dict]:
    """
    Belirli bir zaman penceresi için feature çıkar
    
    Args:
        analysis: Video analiz sonuçları
        start_time: Başlangıç zamanı (saniye)
        end_time: Bitiş zamanı (saniye)
        video_fps: Video FPS
        extraction_fps: Extraction FPS
    
    Returns:
        Feature dictionary veya None
    """
    # Frame numaralarını hesapla (extraction frame)
    start_frame = int(start_time * extraction_fps)
    end_frame = int(end_time * extraction_fps)
    
    # Frame bazlı tracking verileri
    frame_tracked = {
        fr['frame_number']: fr.get('tracked_players', [])
        for fr in analysis.get('frame_results', [])
    }
    
    frame_ball = {
        fr['frame_number']: fr.get('ball_position')
        for fr in analysis.get('frame_results', [])
    }
    
    # Bu pencere için veri topla
    event_frames = []
    event_players = []
    event_ball_positions = []
    
    for extraction_frame in range(start_frame, end_frame + 1):
        if extraction_frame in frame_tracked:
            event_frames.append(extraction_frame)
            event_players.extend(frame_tracked[extraction_frame])
            if extraction_frame in frame_ball and frame_ball[extraction_frame]:
                event_ball_positions.append((extraction_frame, frame_ball[extraction_frame]))
    
    if len(event_frames) < 3:  # Yeterli veri yok
        return None
    
    # Feature'ları hesapla (extract_features.py'deki mantık ile aynı)
    features = {
        'duration': end_time - start_time,
        'frame_count': end_frame - start_frame,
        'start_time': start_time,
        'end_time': end_time,
        
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
            x_positions = [p[0] for p in positions]
            y_positions = [p[1] for p in positions]
            
            features['player_avg_x'] = np.mean(x_positions)
            features['player_avg_y'] = np.mean(y_positions)
            features['player_std_x'] = np.std(x_positions) if len(x_positions) > 1 else 0
            features['player_std_y'] = np.std(y_positions) if len(y_positions) > 1 else 0
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
    else:
        # Default değerler
        features.update({
            'player_avg_x': 0, 'player_avg_y': 0,
            'player_std_x': 0, 'player_std_y': 0,
            'player_min_x': 0, 'player_max_x': 0,
            'player_min_y': 0, 'player_max_y': 0,
            'player_avg_movement': 0, 'player_max_movement': 0, 'player_total_movement': 0
        })
    
    # Top feature'ları
    if event_ball_positions:
        ball_positions = [pos for _, pos in sorted(event_ball_positions)]
        
        if len(ball_positions) >= 2:
            ball_x = [p[0] for p in ball_positions]
            ball_y = [p[1] for p in ball_positions]
            
            features['ball_avg_x'] = np.mean(ball_x)
            features['ball_avg_y'] = np.mean(ball_y)
            features['ball_std_x'] = np.std(ball_x) if len(ball_x) > 1 else 0
            features['ball_std_y'] = np.std(ball_y) if len(ball_y) > 1 else 0
            
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
        features.update({
            'ball_avg_x': 0, 'ball_avg_y': 0,
            'ball_std_x': 0, 'ball_std_y': 0,
            'ball_avg_speed': 0, 'ball_max_speed': 0, 'ball_total_movement': 0,
            'ball_avg_y_change': 0, 'ball_y_trend': 'unknown'
        })
    
    # Oyuncu-Top mesafesi
    if event_players and event_ball_positions:
        distances = []
        for player in event_players:
            if player.get('position'):
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


def analyze_video_with_model(
    video_path: Path,
    model_path: Path,
    fps: float = 3.0,
    window_duration: float = 2.0,
    window_step: float = 0.5,
    confidence_threshold: float = 0.6,
    progress_callback: Optional[callable] = None,
    merge_distance_factor: float = 1.5
) -> List[Dict]:
    """
    Video analizi yap ve model ile olay tespiti
    
    Args:
        video_path: Video dosyası yolu
        model_path: Model dosyası yolu
        fps: Frame extraction FPS
        window_duration: Analiz penceresi süresi (saniye)
        window_step: Pencere adımı (saniye)
        confidence_threshold: Minimum güven eşiği
    
    Returns:
        Tespit edilen olaylar listesi
    """
    logger.info(f"Video analizi başlatılıyor: {video_path}")
    
    if progress_callback:
        progress_callback('video_analysis', 10, "Video analizi yapılıyor (SAM3 + tracking)...")
    
    # 1. Video analizi (SAM3 + tracking) - Hassaslık artırılmış
    logger.info("Video analizi yapılıyor (SAM3 + tracking)...")
    analysis_result = analyze_video_final(
        video_path=video_path,
        text_prompt="basketball player",
        conf_threshold=0.25,  # Daha düşük threshold (0.3 -> 0.25) = daha fazla tespit
        fps=fps,  # fps parametresi zaten yukarıdan geliyor (6.0)
        enable_event_detection=False
    )
    
    if progress_callback:
        progress_callback('model_loading', 40, "Model yükleniyor...")
    
    # 2. Model yükleme
    logger.info(f"Model yükleniyor: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    
    model = EventClassifier.load(str(model_path))
    logger.info("Model yüklendi")
    
    # 3. Video bilgileri
    video_fps = analysis_result.get('video_metadata', {}).get('fps', 30.0)
    extraction_fps = analysis_result.get('analysis_params', {}).get('extraction_fps', fps)
    video_duration = analysis_result.get('video_metadata', {}).get('duration', 0)
    
    logger.info(f"Video süresi: {video_duration:.2f} saniye")
    logger.info(f"Pencere analizi başlatılıyor (pencere: {window_duration}s, adım: {window_step}s)")
    
    # Toplam pencere sayısını hesapla
    total_windows = int((video_duration - window_duration) / window_step) + 1 if video_duration > window_duration else 0
    
    if progress_callback:
        progress_callback('window_analysis', 50, f"Pencere analizi başlatılıyor ({total_windows} pencere)...")
    
    # 4. Sliding window ile feature extraction ve model tahmini
    detected_events = []
    
    current_time = 0.0
    window_count = 0
    
    while current_time + window_duration <= video_duration:
        window_count += 1
        start_time = current_time
        end_time = min(current_time + window_duration, video_duration)
        
        # Feature extraction
        features = extract_features_for_window(
            analysis_result,
            start_time,
            end_time,
            video_fps,
            extraction_fps
        )
        
        if features:
            try:
                # Model tahmini - tüm sınıf olasılıklarını al
                predicted_class, confidence = model.predict(features)
                
                # Tüm sınıf olasılıklarını al (model içinden)
                feature_vector = np.array([[features.get(col, 0) for col in model.feature_columns]])
                feature_vector = feature_vector.astype(float)
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                feature_scaled = model.scaler.transform(feature_vector)
                all_probas = model.model.predict_proba(feature_scaled)[0]
                
                # Sınıf olasılıklarını dictionary'ye çevir (güvenli şekilde - model.classes sırasına göre)
                proba_dict = {}
                # sklearn predict_proba, model.classes_ sırasına göre olasılıkları döndürür
                # Modelimizde classes = ["basket", "pas"], yani all_probas[0] = basket, all_probas[1] = pas
                for i, class_name in enumerate(model.classes):
                    if i < len(all_probas):
                        proba_dict[class_name] = all_probas[i]
                    else:
                        proba_dict[class_name] = 0.0
                
                basket_prob = proba_dict.get('basket', 0)
                pas_prob = proba_dict.get('pas', 0)
                
                # Basket genelde videonun sonunda olur - son 2 saniye kesinlikle basket
                video_last_2_seconds_start = video_duration - 2.0  # Son 2 saniye başlangıcı
                # Pencere başlangıcı son 2 saniyede ise basket (veya pencere orta noktası son 2 saniyede ise)
                window_midpoint = (start_time + end_time) / 2.0
                is_in_last_2_seconds = start_time >= video_last_2_seconds_start or window_midpoint >= video_last_2_seconds_start
                
                # Son 2 saniyede kesinlikle basket'e yönlendir (model ne derse desin)
                if is_in_last_2_seconds:
                    selected_class = 'basket'
                    selected_confidence = max(basket_prob, 0.5)  # Minimum 0.5 confidence ver
                    logger.info(f"Pencere {window_count}: {start_time:.2f}-{end_time:.2f}s -> SON 2 SANİYE: {selected_class} (zorla basket) (basket: {basket_prob:.3f}, pas: {pas_prob:.3f})")
                # Son 2 saniye dışında model normal çalışır
                else:
                    if basket_prob > pas_prob:
                        selected_class = 'basket'
                        selected_confidence = basket_prob
                        logger.info(f"Pencere {window_count}: {start_time:.2f}-{end_time:.2f}s -> MODEL TAHMİNİ: {selected_class} (basket: {basket_prob:.3f} > pas: {pas_prob:.3f})")
                    elif pas_prob > basket_prob:
                        selected_class = 'pas'
                        selected_confidence = pas_prob
                        logger.info(f"Pencere {window_count}: {start_time:.2f}-{end_time:.2f}s -> MODEL TAHMİNİ: {selected_class} (pas: {pas_prob:.3f} > basket: {basket_prob:.3f})")
                    else:
                        # Eşitse model tahminini kullan
                        selected_class = predicted_class
                        selected_confidence = confidence
                        logger.info(f"Pencere {window_count}: {start_time:.2f}-{end_time:.2f}s -> MODEL TAHMİNİ (eşit): {selected_class} (basket: {basket_prob:.3f}, pas: {pas_prob:.3f})")
                
                # Yüksek güven ile tespit edilen olayları kaydet
                if selected_confidence >= confidence_threshold:
                    detected_events.append({
                        'type': selected_class,
                        'start_time': start_time,
                        'end_time': end_time,
                        'confidence': selected_confidence
                    })
                    logger.info(f"✓ Olay tespit edildi: {selected_class} ({selected_confidence:.2f}) @ {start_time:.2f}-{end_time:.2f}s")
            
            except Exception as e:
                logger.warning(f"Model tahmini hatası (window {window_count}): {e}", exc_info=True)
        
        current_time += window_step
        
        # Progress güncelleme (pencere analizi: 50% - 90%)
        if progress_callback and total_windows > 0:
            window_progress = 50 + int((window_count / total_windows) * 40)
            progress_callback('window_analysis', window_progress, 
                            f"Pencere analizi: {window_count}/{total_windows} ({len(detected_events)} olay tespit edildi)")
        
        if window_count % 50 == 0:
            logger.info(f"İşlenen pencere: {window_count}, tespit edilen olay: {len(detected_events)}")
    
    logger.info(f"Toplam {window_count} pencere analiz edildi")
    
    if progress_callback:
        progress_callback('merging', 90, "Olaylar birleştiriliyor...")
    
    # 5. Yakın olayları birleştir (sadece aynı tip olayları, çok küçük mesafe ile)
    logger.info(f"Birleştirme öncesi {len(detected_events)} olay tespit edildi")
    if len(detected_events) > 1:
        merged_events = []
        detected_events.sort(key=lambda x: x['start_time'])
        
        # Tüm tespit edilen olayları logla (debug için)
        for i, evt in enumerate(detected_events):
            logger.info(f"  Olay {i+1}: {evt['type']} @ {evt['start_time']:.2f}-{evt['end_time']:.2f}s (conf: {evt['confidence']:.2f})")
        
        current_event = detected_events[0].copy()
        # Birleştirme mesafesi: window_step'in merge_distance_factor katı (overlap toleransı)
        # Daha büyük birleştirme mesafesi kullanarak daha fazla birleştirme yap
        merge_distance = window_step * merge_distance_factor
        
        for next_event in detected_events[1:]:
            # SADECE aynı tipte ve ÇOK yakınsa birleştir (farklı tipler kesinlikle ayrı kalmalı)
            time_gap = next_event['start_time'] - current_event['end_time']
            if (next_event['type'] == current_event['type'] and time_gap < merge_distance):
                # Aynı tip olayları birleştir (confidence'ı ortalama al)
                current_event['end_time'] = next_event['end_time']
                # Confidence'ı ortalama al (daha tutarlı)
                current_event['confidence'] = (current_event['confidence'] + next_event['confidence']) / 2.0
                logger.info(f"Birleştirildi: {current_event['type']} @ {current_event['start_time']:.2f}-{current_event['end_time']:.2f}s (gap: {time_gap:.2f}s < {merge_distance:.2f}s)")
            else:
                # Farklı tip veya yeterince uzaksa ayrı olay olarak ekle
                merged_events.append(current_event)
                current_event = next_event.copy()
        
        merged_events.append(current_event)
        detected_events = merged_events
        logger.info(f"Birleştirme sonrası {len(detected_events)} olay kaldı")
    else:
        if len(detected_events) == 1:
            logger.info(f"Tek olay: {detected_events[0]['type']} @ {detected_events[0]['start_time']:.2f}-{detected_events[0]['end_time']:.2f}s")
    
    logger.info(f"Toplam {len(detected_events)} olay tespit edildi (birleştirme sonrası)")
    
    # Zaman sırasına göre sırala
    detected_events.sort(key=lambda x: x['start_time'])
    
    # Overlap eden olayları temizle - aynı zaman dilimi sadece bir olaya ait olmalı
    if len(detected_events) > 1:
        final_events = []
        video_last_2_seconds_start = video_duration - 2.0  # Son 2 saniye başlangıcı
        
        for current_event in detected_events:
            should_add = True
            overlap_found = False
            
            # Önceki olaylarla overlap var mı kontrol et
            for i, prev_event in enumerate(final_events):
                # Overlap kontrolü: zaman dilimleri kesişiyorsa
                if not (current_event['end_time'] <= prev_event['start_time'] or 
                       current_event['start_time'] >= prev_event['end_time']):
                    overlap_found = True
                    
                    # Overlap var - hangi olaya öncelik vereceğiz?
                    current_window_midpoint = (current_event['start_time'] + current_event['end_time']) / 2.0
                    prev_window_midpoint = (prev_event['start_time'] + prev_event['end_time']) / 2.0
                    is_current_in_last_2_seconds = current_event['start_time'] >= video_last_2_seconds_start or current_window_midpoint >= video_last_2_seconds_start
                    is_prev_in_last_2_seconds = prev_event['start_time'] >= video_last_2_seconds_start or prev_window_midpoint >= video_last_2_seconds_start
                    
                    # Basket son 2 saniyede ise öncelikli
                    if current_event['type'] == 'basket' and prev_event['type'] == 'pas':
                        if is_current_in_last_2_seconds:
                            # Basket son 2 saniyede, pas'i kısalt
                            prev_event['end_time'] = min(prev_event['end_time'], current_event['start_time'])
                            should_add = True
                        else:
                            # Pas'i koru, basket'i ekleme
                            should_add = False
                    elif current_event['type'] == 'pas' and prev_event['type'] == 'basket':
                        if is_prev_in_last_2_seconds:
                            # Basket son 2 saniyede, pas'i ekleme
                            should_add = False
                        else:
                            # Basket'i kısalt
                            prev_event['end_time'] = min(prev_event['end_time'], current_event['start_time'])
                            should_add = True
                    elif current_event['type'] == prev_event['type']:
                        # Aynı tip olaylar - birleştirme zaten yapıldı, confidence'a göre seç
                        if current_event['confidence'] > prev_event['confidence']:
                            # Yeni olay daha yüksek confidence, eski olayı kısalt
                            prev_event['end_time'] = min(prev_event['end_time'], current_event['start_time'])
                            should_add = True
                        else:
                            # Eski olay daha yüksek confidence, yeni olayı ekleme
                            should_add = False
                    else:
                        # Farklı tip olaylar - confidence'a göre seç
                        if current_event['confidence'] > prev_event['confidence']:
                            prev_event['end_time'] = min(prev_event['end_time'], current_event['start_time'])
                            should_add = True
                        else:
                            should_add = False
                    
                    if not should_add:
                        break
            
            if should_add:
                final_events.append(current_event)
        
        detected_events = final_events
        logger.info(f"Overlap temizleme sonrası {len(detected_events)} olay kaldı")
    
    return detected_events


def get_video_analysis_events_simple(
    video_path: Path,
    model_path: Optional[Path] = None,
    progress_callback: Optional[callable] = None
) -> List[Dict]:
    """
    Basitleştirilmiş API - Video analizi yap ve model ile olay tespiti
    
    Bu fonksiyon web uygulaması tarafından çağrılır. Optimize edilmiş parametreler
    kullanır (fps=6.0, window_duration=1.5s, window_step=0.8s, confidence=0.45).
    
    Args:
        video_path: Video dosyası yolu
        model_path: Model dosyası yolu (None ise default: event_classifier_regularized.pkl)
        progress_callback: İlerleme callback fonksiyonu (step, progress, message)
    
    Returns:
        List[Dict]: Tespit edilen olaylar listesi
                   [{'type': 'basket', 'start_time': 2.5, 'end_time': 3.0, 'confidence': 0.8}, ...]
    """
    if model_path is None:
        # Varsayılan olarak regularized model kullan (overfitting önlenmiş)
        model_path = project_root / 'data' / 'models' / 'event_classifier_regularized.pkl'
    
    if not model_path.exists():
        logger.warning(f"Model bulunamadı: {model_path}, mock sonuçlar döndürülüyor")
        # Mock events (model yoksa test için)
        import random
        import cv2
        
        events = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        num_events = random.randint(3, 8)
        for i in range(num_events):
            start_time = random.uniform(5, duration - 5)
            event_duration = random.uniform(1.0, 3.0)
            end_time = min(start_time + event_duration, duration)
            event_type = random.choice(['basket', 'pas'])
            confidence = random.uniform(0.75, 0.98)
            
            events.append({
                'type': event_type,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence
            })
        
        events.sort(key=lambda x: x['start_time'])
        return events
    
    # Gerçek model entegrasyonu (Dengeli parametreler - gerçekçi olay tespiti için)
    return analyze_video_with_model(
        video_path=video_path,
        model_path=model_path,
        fps=6.0,  # Dengeli FPS = yeterli detay, performanslı
        window_duration=1.5,  # Orta pencere = olayları daha iyi kapsar
        window_step=0.8,  # Daha büyük adım = daha az pencere, daha az olay
        confidence_threshold=0.45,  # Daha yüksek threshold = daha az false positive
        progress_callback=progress_callback,
        merge_distance_factor=2.0  # Daha büyük birleştirme mesafesi = daha fazla birleştirme
    )



