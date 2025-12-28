"""
Ana Video Analiz Modülü - Basketbol Video Analizi

Bu modül, basketbol videolarını analiz ederek oyuncu tespiti, takibi ve özellik çıkarımı yapar.
SAM3 (Segment Anything Model 3) kullanarak oyuncuları tespit eder ve Kalman filter tabanlı
tracking sistemi ile aynı oyuncuya aynı ID'yi atar.

Ana İşlevler:
- Oyuncu tespiti (SAM3 ile text prompt tabanlı)
- Oyuncu takibi (Kalman filter + IoU matching)
- Top tespiti (ML model için feature extraction)
- Forma numarası tanıma (isteğe bağlı)
- Analiz sonuçlarını JSON ve CSV formatında kaydetme

Kullanım:
    python analyze_video_final.py --video data/input/video.mp4 --fps 5.0
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json
from tqdm import tqdm
import cv2

from src.config.settings import settings
from src.utils.helpers import setup_logging
from src.video.processor import VideoProcessor
from src.ai.detection import PlayerDetector
from src.ai.tracking_improved import PlayerTracker, TrackedPlayer
from src.ai.jersey_number_recognizer import JerseyNumberRecognizer
from src.ai.ball_detector import BallDetector
# Event detection kaldırıldı - sadece manuel etiketleme kullanılacak

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


def analyze_video_final(
    video_path: Path,
    text_prompt: str = "basketball player",
    conf_threshold: float = 0.3,
    fps: float = 5.0,
    max_frames: int = None,
    enable_jersey_recognition: bool = False,
    enable_event_detection: bool = False  # Rule-based kaldırıldı, sadece manuel etiketleme
) -> Dict[str, Any]:
    """
    Ana video analiz fonksiyonu - Basketbol videolarını analiz eder
    
    Bu fonksiyon videodaki her frame'i işleyerek:
    1. SAM3 ile oyuncuları tespit eder (text prompt: "basketball player")
    2. Kalman filter tabanlı tracking ile aynı oyuncuya aynı ID'yi atar
    3. Top pozisyonunu tespit eder (ML model için feature extraction için)
    4. İsteğe bağlı olarak forma numarası tanır
    
    Analiz sonuçları JSON ve CSV formatında kaydedilir.
    
    Args:
        video_path: Analiz edilecek video dosyasının yolu
        text_prompt: SAM3 için text prompt (varsayılan: "basketball player")
        conf_threshold: Tespit güven eşiği (0.0-1.0 arası, düşük değer = daha fazla tespit)
        fps: Saniyede kaç frame analiz edileceği (performans için düşük değerler kullanılabilir)
        max_frames: Maksimum frame sayısı (None ise tüm video analiz edilir, test için kullanılır)
        enable_jersey_recognition: Forma numarası tanıma aktif olsun mu (OCR ile, yavaş)
        enable_event_detection: Rule-based olay tespiti (kullanılmıyor, ML model kullanılıyor)
    
    Returns:
        Dict: Analiz sonuçları içeren dictionary (metadata, frame_results, track_statistics, vb.)
    """
    logger.info("="*60)
    logger.info("FINAL VIDEO ANALİZİ - GELİŞTİRİLMİŞ TRACKING")
    logger.info("="*60)
    logger.info(f"Video: {video_path}")
    logger.info(f"Text Prompt: '{text_prompt}' (sadece basketbol oyuncuları)")
    logger.info(f"Tracking: Aktif (aynı oyuncu = aynı ID)")
    logger.info("Olay Tespiti: Kapalı (Sadece manuel etiketleme kullanılacak)")
    
    # ========== 1. Video Yükleme ve Metadata ==========
    # Video dosyasını yükler ve temel bilgileri (süre, çözünürlük, FPS) alır
    processor = VideoProcessor(video_path, fps=fps)
    metadata = processor.video_loader.get_metadata()
    
    logger.info(f"\nVideo Metadata:")
    logger.info(f"  Süre: {metadata['duration']:.2f} saniye")
    logger.info(f"  Çözünürlük: {metadata['width']}x{metadata['height']}")
    
    # ========== 2. Oyuncu Tespiti (SAM3) ==========
    # SAM3 modelini kullanarak videodaki basketbol oyuncularını tespit eder
    # Text prompt ile sadece "basketball player" objelerini bulur
    logger.info("\nSAM3 model yükleniyor...")
    detector = PlayerDetector(
        sam3_model=None,
        text_prompt=text_prompt,  # "basketball player" - sadece oyuncular
        conf_threshold=conf_threshold,
        use_local=True  # Yerel SAM3 modeli kullan
    )
    logger.info("Oyuncu Detector hazır!")
    
    # ========== 3. Top Tespiti ==========
    # Top pozisyonunu tespit eder (ML model için feature extraction'da kullanılır)
    # Aynı SAM3 modelini paylaşarak performans optimizasyonu yapar
    ball_detector = BallDetector(sam3_model=detector.sam3_model, conf_threshold=0.3)
    logger.info("Top Detector hazır! (ML model için feature extraction) - Threshold: 0.3")
    
    # ========== 4. Oyuncu Takibi (Tracking) ==========
    # Kalman filter tabanlı tracking sistemi - Aynı oyuncuya aynı ID'yi atar
    # IoU (Intersection over Union) matching ile oyuncuları frame'ler arasında eşleştirir
    tracker = PlayerTracker(
        iou_threshold=0.4,  # Daha yüksek threshold = daha az yeni track oluşturur (stabilite için)
        max_disappeared=20,  # Oyuncu 20 frame boyunca görünmese bile track'i koru
        frame_rate=metadata['fps']
    )
    logger.info("Geliştirilmiş Tracker hazır! (Kalman filter + improved matching)")
    
    # ========== 5. Forma Numarası Tanıma (İsteğe Bağlı) ==========
    # OCR (Optical Character Recognition) ile oyuncu forma numaralarını tanır
    # Yavaş olduğu için varsayılan olarak kapalı, gerekirse açılabilir
    jersey_recognizer = None
    if enable_jersey_recognition:
        try:
            jersey_recognizer = JerseyNumberRecognizer()
            logger.info("Jersey recognizer hazır!")
        except Exception as e:
            logger.warning(f"Jersey recognizer başlatılamadı: {e}")
            enable_jersey_recognition = False
    
    # ========== 6. Frame İşleme Döngüsü ==========
    # Her frame için: detection -> tracking -> top tespiti -> sonuç kaydetme
    all_tracked_players = []  # Tüm frame'lerdeki takip edilen oyuncular
    frame_results = []  # Her frame için analiz sonuçları
    total_detections = 0  # Toplam tespit sayısı (istatistik için)
    ball_positions_by_frame = {}  # frame_num -> (x, y) veya None (top pozisyonları)
    
    logger.info("\nFrame işleme başlıyor...")
    logger.info("-"*60)
    
    frame_iterator = processor.get_frames()
    if max_frames:
        frame_iterator = list(frame_iterator)[:max_frames]
    
    for frame_num, frame in tqdm(frame_iterator, desc="Processing"):
        try:
            frame_time = frame_num / fps
            
            # 6.1. Oyuncu Tespiti: SAM3 ile frame'deki oyuncuları bul
            detections = detector.detect(frame)
            total_detections += len(detections)
            
            # 6.2. Oyuncu Takibi: Tespit edilen oyuncuları önceki frame'lerdeki oyuncularla eşleştir
            # Kalman filter ile pozisyon tahmini yapar ve IoU ile eşleştirme yapar
            tracked_players = tracker.update(detections, frame, frame_num)
            
            # 6.3. Top Tespiti: ML model için feature extraction için top pozisyonunu bul
            # Performans optimizasyonu: Her 5 frame'de bir tespit yap, diğerlerinde interpolasyon kullan
            ball_position = None
            if ball_detector:
                ball_detection_interval = 5  # Her 5 frame'de bir top tespiti yap
                if frame_num % ball_detection_interval == 0:
                    try:
                        ball_position = ball_detector.get_ball_position(frame)
                        ball_positions_by_frame[frame_num] = ball_position
                    except Exception as e:
                        logger.debug(f"Frame {frame_num}: Top tespiti hatası: {e}")
                        ball_positions_by_frame[frame_num] = None
                else:
                    # Son tespit edilen pozisyonu kullan (basit interpolasyon - performans için)
                    last_detected_frame = (frame_num // ball_detection_interval) * ball_detection_interval
                    ball_position = ball_positions_by_frame.get(last_detected_frame)
                    ball_positions_by_frame[frame_num] = ball_position
            
            # 6.4. Forma Numarası Tanıma: Her 30 frame'de bir OCR ile forma numarasını tanı
            # Yavaş olduğu için sadece belirli frame'lerde çalıştırılır
            if enable_jersey_recognition and jersey_recognizer and frame_num % 30 == 0:
                for player in tracked_players:
                    if player.jersey_number is None:  # Henüz numara tanınmadıysa
                        jersey_num = jersey_recognizer.recognize_number(frame, player.bbox)
                        if jersey_num:
                            tracker.assign_jersey_number(player.track_id, jersey_num)
            
            # 6.5. Frame Sonuçlarını Kaydet: Her frame için tespit edilen oyuncuları ve top pozisyonunu kaydet
            frame_result = {
                'frame_number': frame_num,
                'time_seconds': frame_time,
                'num_detections': len(detections),
                'num_tracked': len(tracked_players),
                'ball_position': list(ball_position) if ball_position else None,
                'tracked_players': []
            }
            
            for player in tracked_players:
                frame_result['tracked_players'].append({
                    'track_id': player.track_id,
                    'bbox': list(player.bbox),
                    'confidence': player.confidence,
                    'position': list(player.position) if player.position else None,
                    'jersey_number': player.jersey_number
                })
                all_tracked_players.append({
                    'frame': frame_num,
                    'time': frame_time,
                    'track_id': player.track_id,
                    'bbox': list(player.bbox),
                    'confidence': player.confidence,
                    'position': list(player.position) if player.position else None,
                    'jersey_number': player.jersey_number
                })
            
            frame_results.append(frame_result)
            
            # İlerleme
            if frame_num % 30 == 0 and len(tracked_players) > 0:
                unique_tracks = len(set(p.track_id for p in tracked_players))
                logger.info(f"Frame {frame_num} ({frame_time:.1f}s): {len(tracked_players)} oyuncu, {unique_tracks} unique track")
        
        except Exception as e:
            logger.error(f"Frame {frame_num} hatası: {e}")
            continue
    
    # ========== 7. Olay Tespiti ==========
    # Rule-based olay tespiti kaldırıldı - ML model kullanılıyor
    # Manuel etiketleme ile training verisi oluşturulur, ML model ile tahmin yapılır
    detected_events = []
    logger.info("Olay tespiti: Manuel etiketleme ile yapılacak")
    
    logger.info("\n" + "="*60)
    logger.info("ANALİZ TAMAMLANDI")
    logger.info("="*60)
    
    # ========== 8. İstatistik Hesaplama ==========
    # Her track için istatistikler hesaplanır (track uzunluğu, süre, forma numarası vb.)
    track_stats = {}
    for track_id in tracker.get_active_tracks():
        trajectory = tracker.get_player_trajectory(track_id)
        if trajectory:
            track_stats[track_id] = {
                'total_frames': len(trajectory),  # Bu track'te kaç frame göründü
                'first_frame': trajectory[0].frame_number,  # İlk göründüğü frame
                'last_frame': trajectory[-1].frame_number,  # Son göründüğü frame
                'duration_frames': trajectory[-1].frame_number - trajectory[0].frame_number,
                'duration_seconds': (trajectory[-1].frame_number - trajectory[0].frame_number) / fps,
                'jersey_number': trajectory[0].jersey_number if trajectory[0].jersey_number else None
            }
    
    # İstatistikler
    frames_with_tracks = sum(1 for fr in frame_results if fr['num_tracked'] > 0)
    unique_tracks = len(track_stats)
    long_tracks = sum(1 for stats in track_stats.values() if stats['total_frames'] >= 20)
    
    logger.info(f"Toplam işlenen frame: {len(frame_results)}")
    logger.info(f"Toplam detection: {total_detections}")
    logger.info(f"Benzersiz track sayısı: {unique_tracks}")
    logger.info(f"Uzun track sayısı (>=20 frame): {long_tracks}")
    
    if track_stats:
        avg_track_length = sum(s['total_frames'] for s in track_stats.values()) / len(track_stats)
        logger.info(f"Ortalama track uzunluğu: {avg_track_length:.1f} frame")
    
    if detected_events:
        logger.info(f"\nTespit edilen olaylar:")
        for event in detected_events:
            logger.info(f"  - {event.event_type.upper()}: Track ID {event.track_id}, "
                       f"Frame {event.frame_start}-{event.frame_end}, "
                       f"Confidence: {event.confidence:.2f}")
    
    # ========== 9. Sonuçları Derleme ==========
    # Tüm analiz sonuçlarını bir dictionary'de topla (JSON kaydetme için)
    results = {
        'video_path': str(video_path),
        'video_metadata': metadata,
        'analysis_params': {
            'text_prompt': text_prompt,
            'conf_threshold': conf_threshold,
            'extraction_fps': fps,
            'max_frames': max_frames,
            'tracking_enabled': True,
            'jersey_recognition_enabled': enable_jersey_recognition,
            'event_detection_enabled': False  # Rule-based kaldırıldı
        },
        'statistics': {
            'total_frames_processed': len(frame_results),
            'frames_with_detections': frames_with_tracks,
            'total_detections': total_detections,
            'unique_tracks': unique_tracks,
            'long_tracks': long_tracks,
            'avg_track_length': avg_track_length if track_stats else 0,
            'total_events_detected': len(detected_events)
        },
        'track_statistics': track_stats,
        'detected_events': [
            {
                'event_type': e.event_type,
                'track_id': e.track_id,
                'frame_start': e.frame_start,
                'frame_end': e.frame_end,
                'confidence': e.confidence,
                'position': list(e.position) if e.position else None,
                'metadata': e.metadata
            }
            for e in detected_events
        ],
        'frame_results': frame_results,
        'all_tracked_players': all_tracked_players
    }
    
    # ========== 10. Sonuçları Kaydetme ==========
    # JSON formatında detaylı analiz sonuçlarını kaydet (ML model için feature extraction'da kullanılır)
    results_path = settings.results_dir / f"{video_path.stem}_final_analysis.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSonuçlar kaydedildi: {results_path}")
    
    # CSV formatında oyuncu pozisyonlarını kaydet (Excel'de analiz için)
    import pandas as pd
    if all_tracked_players:
        df = pd.DataFrame(all_tracked_players)
        csv_path = settings.results_dir / f"{video_path.stem}_final_tracked.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV kaydedildi: {csv_path}")
    
    # Olaylar CSV
    if detected_events:
        events_data = [
            {
                'event_type': e.event_type,
                'track_id': e.track_id,
                'frame_start': e.frame_start,
                'frame_end': e.frame_end,
                'time_start': e.frame_start / fps,
                'time_end': e.frame_end / fps,
                'confidence': e.confidence,
                'position_x': e.position[0] if e.position else None,
                'position_y': e.position[1] if e.position else None
            }
            for e in detected_events
        ]
        df_events = pd.DataFrame(events_data)
        events_csv_path = settings.results_dir / f"{video_path.stem}_events.csv"
        df_events.to_csv(events_csv_path, index=False)
        logger.info(f"Olaylar CSV kaydedildi: {events_csv_path}")
    
    return results


def create_final_visualization(
    video_path: Path,
    results: Dict[str, Any],
    output_path: Path = None
):
    """
    Analiz sonuçlarını görselleştiren video oluşturur
    
    Görselleştirmede:
    - Her oyuncu için bounding box ve track ID gösterilir
    - Farklı oyuncular için farklı renkler kullanılır
    - Forma numarası varsa gösterilir
    - Top pozisyonu sarı daire ile gösterilir
    - Olaylar (basket, pas) üstte metin olarak gösterilir
    
    Args:
        video_path: Orijinal video dosyası
        results: analyze_video_final() fonksiyonundan dönen sonuçlar
        output_path: Çıktı video dosyası yolu (None ise otomatik oluşturulur)
    """
    if output_path is None:
        output_path = settings.output_dir / f"{video_path.stem}_final_tracked.mp4"
    
    logger.info(f"Görselleştirmeli video oluşturuluyor: {output_path}")
    
    processor = VideoProcessor(video_path, fps=results['analysis_params']['extraction_fps'])
    metadata = processor.video_loader.get_metadata()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        results['analysis_params']['extraction_fps'],
        (metadata['width'], metadata['height'])
    )
    
    frame_tracked = {
        fr['frame_number']: fr['tracked_players'] 
        for fr in results['frame_results']
    }
    frame_ball_positions = {
        fr['frame_number']: fr.get('ball_position')
        for fr in results['frame_results']
    }
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ] * 20  # Daha fazla renk
    
    for frame_num, frame in tqdm(processor.get_frames(), desc="Creating video"):
        vis_frame = frame.copy()
        tracked_players = frame_tracked.get(frame_num, [])
        
        for player_data in tracked_players:
            x1, y1, x2, y2 = map(int, player_data['bbox'])
            track_id = player_data['track_id']
            jersey_num = player_data.get('jersey_number')
            
            color = colors[track_id % len(colors)]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id}"
            if jersey_num is not None:
                label += f" #{jersey_num}"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis_frame,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            if player_data.get('position'):
                cx, cy = map(int, player_data['position'])
                cv2.circle(vis_frame, (cx, cy), 5, color, -1)
        
        # Top pozisyonu
        ball_pos = frame_ball_positions.get(frame_num)
        if ball_pos:
            bx, by = map(int, ball_pos)
            cv2.circle(vis_frame, (bx, by), 8, (0, 255, 255), -1)  # Sarı top
            cv2.circle(vis_frame, (bx, by), 8, (0, 0, 0), 2)  # Siyah kenar
        
        # Olayları göster
        frame_events = [e for e in results.get('detected_events', []) 
                       if e['frame_start'] <= frame_num <= e['frame_end']]
        if frame_events:
            event_text = ", ".join([e['event_type'].upper() for e in frame_events])
            cv2.putText(
                vis_frame,
                f"EVENT: {event_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        out.write(vis_frame)
    
    out.release()
    processor.video_loader.close()
    logger.info(f"Video hazır: {output_path}")


def main():
    """
    Komut satırı arayüzü - Video analizini başlatır
    
    Kullanım örnekleri:
        # Varsayılan video ile
        python analyze_video_final.py
        
        # Özel video ile
        python analyze_video_final.py --video data/input/my_video.mp4
        
        # Düşük FPS ile (daha hızlı)
        python analyze_video_final.py --fps 3.0
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Final video analizi")
    parser.add_argument(
        '--video',
        type=str,
        default="data/input/nba_test_video.mp4",
        help='Video dosyası yolu'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=3.0,
        help='Frame extraction FPS (varsayılan: 3.0)'
    )
    
    args = parser.parse_args()
    video_path = Path(args.video)
    
    if not video_path.exists():
        logger.error(f"Video bulunamadı: {video_path}")
        return
    
    # Final analiz
    results = analyze_video_final(
        video_path=video_path,
        text_prompt="basketball player",  # Sadece basketbol oyuncuları
        conf_threshold=0.4,  # Biraz artırıldı (daha az false positive)
        fps=args.fps,  # Komut satırından gelen FPS
        max_frames=None,
        enable_jersey_recognition=False,
        enable_event_detection=False  # Rule-based kaldırıldı, sadece manuel etiketleme
    )
    
    # Görselleştirme
    if results['statistics']['unique_tracks'] > 0:
        logger.info("\nGörselleştirmeli video oluşturuluyor...")
        create_final_visualization(video_path, results)
        
        logger.info("\n" + "="*60)
        logger.info("TRACKING KALİTE RAPORU")
        logger.info("="*60)
        logger.info(f"Benzersiz track: {results['statistics']['unique_tracks']}")
        logger.info(f"Uzun track (>=20 frame): {results['statistics']['long_tracks']}")
        logger.info(f"Ortalama track uzunluğu: {results['statistics']['avg_track_length']:.1f} frame")
        
        if results['statistics']['long_tracks'] > results['statistics']['unique_tracks'] * 0.3:
            logger.info("✅ Tracking kalitesi İYİ - Aynı oyuncu aynı ID'ye sahip")
        else:
            logger.info("⚠️  Tracking kalitesi düşük olabilir")
        
        # Olay tespiti raporu
        if results.get('detected_events'):
            logger.info("\n" + "="*60)
            logger.info("OLAY TESPİTİ RAPORU")
            logger.info("="*60)
            logger.info(f"Toplam tespit edilen olay: {results['statistics']['total_events_detected']}")
            
            events_by_type = {}
            for event in results['detected_events']:
                event_type = event['event_type']
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            for event_type, count in events_by_type.items():
                logger.info(f"  {event_type.upper()}: {count}")


if __name__ == "__main__":
    main()

