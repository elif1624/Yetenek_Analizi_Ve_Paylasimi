"""
Olay tespitlerini video üzerinde görselleştir
Hem otomatik tespitleri hem de manuel etiketleri gösterir
"""

import cv2
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_analysis_results(results_file: Path) -> Dict:
    """Analiz sonuçlarını yükle"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_labels(label_file: Path) -> Dict:
    """Etiket dosyasını yükle"""
    with open(label_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def draw_event_on_frame(
    frame: np.ndarray,
    event: Dict,
    color: tuple,
    label: str,
    fps: float
):
    """Frame üzerine olay çiz"""
    frame_start = event.get('frame_start', event.get('start_frame', 0))
    frame_end = event.get('frame_end', event.get('end_frame', 0))
    current_frame = event.get('current_frame', 0)
    
    # Eğer mevcut frame bu olayın aralığında değilse çizme
    if current_frame < frame_start or current_frame > frame_end:
        return
    
    # Olay tipine göre renk
    event_type = event.get('event_type', 'unknown')
    
    # Zaman bilgisi
    time_start = frame_start / fps
    time_end = frame_end / fps
    current_time = current_frame / fps
    
    # Frame üzerine bilgi yaz
    info_text = f"{label}: {event_type.upper()}"
    time_text = f"{time_start:.1f}s - {time_end:.1f}s"
    confidence_text = f"Conf: {event.get('confidence', 0):.2f}" if 'confidence' in event else ""
    
    # Arka plan kutusu
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Metin boyutları
    (text_width1, text_height1), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(time_text, font, font_scale * 0.8, thickness)
    
    # Arka plan
    cv2.rectangle(frame,
                 (10, 10),
                 (max(text_width1, text_width2) + 20, y_offset + text_height1 + text_height2 + 20),
                 (0, 0, 0), -1)
    
    # Metin
    cv2.putText(frame, info_text, (15, y_offset + text_height1),
               font, font_scale, color, thickness)
    cv2.putText(frame, time_text, (15, y_offset + text_height1 + text_height2 + 5),
               font, font_scale * 0.8, color, thickness)
    
    if confidence_text:
        cv2.putText(frame, confidence_text, (15, y_offset + text_height1 + text_height2 * 2 + 10),
                   font, font_scale * 0.7, color, thickness)
    
    # Frame'in etrafına renkli çerçeve
    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, 5)


def create_event_visualization(
    video_path: Path,
    analysis_results: Dict,
    labels: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    fps: float = 3.0
):
    """Olayları video üzerinde görselleştir"""
    
    # Video aç
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"HATA: Video açılamadı: {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {total_frames} frame, {video_fps:.2f} FPS")
    
    # Output video writer
    if output_path is None:
        output_path = Path("data/output") / f"{video_path.stem}_events_visualized.mp4"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
    
    # Olayları al
    detected_events = analysis_results.get('detected_events', [])
    print(f"Tespit edilen olay sayisi: {len(detected_events)}")
    
    # Analiz parametreleri
    extraction_fps = analysis_results.get('analysis_params', {}).get('extraction_fps', 3.0)
    print(f"Analiz FPS: {extraction_fps}, Video FPS: {video_fps}")
    
    # Frame başına olayları grupla
    events_by_frame = {}
    
    # Tespit edilen olaylar - Frame numaralarını gerçek video frame'lerine çevir
    for event in detected_events:
        # Analiz sırasında extraction_fps kullanıldı, gerçek video FPS'e çevir
        analysis_frame_start = event.get('frame_start', 0)
        analysis_frame_end = event.get('frame_end', 0)
        
        # Zamanı hesapla (analiz frame'lerinden)
        time_start = analysis_frame_start / extraction_fps
        time_end = analysis_frame_end / extraction_fps
        
        # Gerçek video frame numaralarına çevir
        real_frame_start = int(time_start * video_fps)
        real_frame_end = int(time_end * video_fps)
        
        # Frame aralığını genişlet (daha görünür olsun)
        for frame_num in range(max(0, real_frame_start - 5), min(total_frames, real_frame_end + 5)):
            if frame_num not in events_by_frame:
                events_by_frame[frame_num] = {'detected': [], 'ground_truth': []}
            # Gerçek frame numaralarını event'e ekle
            event_copy = event.copy()
            event_copy['real_frame_start'] = real_frame_start
            event_copy['real_frame_end'] = real_frame_end
            events_by_frame[frame_num]['detected'].append(event_copy)
    
    # Etiketleri al
    ground_truth_events = []
    if labels:
        ground_truth_events = labels.get('labels', [])
        print(f"Etiketlenmis olay sayisi: {len(ground_truth_events)}")
    
    # Etiketlenmiş olaylar
    for event in ground_truth_events:
        frame_start = event.get('start_frame', 0)
        frame_end = event.get('end_frame', 0)
        for frame_num in range(frame_start, frame_end + 1):
            if frame_num not in events_by_frame:
                events_by_frame[frame_num] = {'detected': [], 'ground_truth': []}
            events_by_frame[frame_num]['ground_truth'].append(event)
    
    # Video işle
    frame_num = 0
    print("\nVideo işleniyor...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Bu frame'deki olaylar
        frame_events = events_by_frame.get(frame_num, {'detected': [], 'ground_truth': []})
        
        # Tespit edilen olayları çiz (YEŞİL)
        for event in frame_events['detected']:
            event_with_frame = event.copy()
            event_with_frame['current_frame'] = frame_num
            # Gerçek frame numaralarını kullan
            if 'real_frame_start' in event:
                event_with_frame['frame_start'] = event['real_frame_start']
                event_with_frame['frame_end'] = event['real_frame_end']
            draw_event_on_frame(frame, event_with_frame, (0, 255, 0), "TESPIT", video_fps)
        
        # Etiketlenmiş olayları çiz (KIRMIZI)
        for event in frame_events['ground_truth']:
            event_with_frame = event.copy()
            event_with_frame['current_frame'] = frame_num
            event_with_frame['frame_start'] = event.get('start_frame', 0)
            event_with_frame['frame_end'] = event.get('end_frame', 0)
            draw_event_on_frame(frame, event_with_frame, (0, 0, 255), "ETIKET", video_fps)
        
        # Frame numarası ve zaman
        time_sec = frame_num / video_fps
        info_text = f"Frame: {frame_num} | Time: {time_sec:.2f}s"
        cv2.putText(frame, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Legend
        legend_y = height - 60
        cv2.rectangle(frame, (10, legend_y - 20), (300, legend_y + 30), (0, 0, 0), -1)
        cv2.putText(frame, "YEŞİL: Otomatik Tespit", (15, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "KIRMIZI: Manuel Etiket", (15, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        out.write(frame)
        frame_num += 1
        
        if frame_num % 100 == 0:
            print(f"  İşlenen frame: {frame_num}/{total_frames}")
    
    cap.release()
    out.release()
    
    print(f"\nGörselleştirme tamamlandı: {output_path}")
    print(f"Toplam frame: {frame_num}")
    print(f"Tespit edilen olay: {len(detected_events)}")
    print(f"Etiketlenmis olay: {len(ground_truth_events)}")


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Olay tespitlerini video üzerinde görselleştir')
    parser.add_argument('--video', type=str, default='data/input/nba_test_video.mp4',
                       help='Video dosya yolu')
    parser.add_argument('--results', type=str, default='data/results/nba_test_video_final_analysis.json',
                       help='Analiz sonuçları JSON dosyası')
    parser.add_argument('--labels', type=str, default='data/labels/nba_test_video_labels.json',
                       help='Etiket dosyası (opsiyonel)')
    parser.add_argument('--output', type=str, default=None,
                       help='Çıktı video dosyası (opsiyonel)')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    results_file = Path(args.results)
    label_file = Path(args.labels) if args.labels else None
    output_path = Path(args.output) if args.output else None
    
    if not video_path.exists():
        print(f"HATA: Video bulunamadi: {video_path}")
        sys.exit(1)
    
    if not results_file.exists():
        print(f"HATA: Analiz sonuçlari bulunamadi: {results_file}")
        sys.exit(1)
    
    # Sonuçları yükle
    print("Analiz sonuçlari yükleniyor...")
    analysis_results = load_analysis_results(results_file)
    
    # Etiketleri yükle
    labels = None
    if label_file and label_file.exists():
        print("Etiketler yükleniyor...")
        labels = load_labels(label_file)
    
    # Görselleştirme oluştur
    create_event_visualization(
        video_path=video_path,
        analysis_results=analysis_results,
        labels=labels,
        output_path=output_path
    )


if __name__ == "__main__":
    main()

