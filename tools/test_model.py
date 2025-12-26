"""
Model test scripti - Etiketli verilerle model performansını ölç
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

# Proje root'unu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.metrics import MetricsCalculator
from analyze_video_final import analyze_video_final
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ground_truth(label_file: Path) -> Dict:
    """Etiket dosyasını yükle"""
    with open(label_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_model_on_labeled_video(
    video_path: Path,
    label_file: Path,
    iou_threshold: float = 0.5
):
    """Etiketli video üzerinde modeli test et"""
    
    logger.info(f"Video analiz ediliyor: {video_path}")
    logger.info(f"Etiket dosyasi: {label_file}")
    
    # 1. Video analizi yap (sadece tracking, event detection yok)
    logger.info("Video analizi baslatiliyor (tracking verileri için)...")
    analysis_result = analyze_video_final(
        video_path=video_path,
        enable_event_detection=False,  # Rule-based kaldırıldı
        fps=3.0  # Hız için düşük FPS
    )
    
    # 2. Rule-based event detection kaldırıldı
    # Şimdilik sadece manuel etiketlerle çalışıyoruz
    # ML model eğitildikten sonra burada model tahminleri kullanılacak
    detected_events = []
    logger.info("Rule-based event detection kaldirildi. ML model eğitildikten sonra burada kullanilacak.")
    
    # 3. Ground truth'u yükle
    ground_truth_data = load_ground_truth(label_file)
    ground_truth_labels = ground_truth_data.get('labels', [])
    video_info = ground_truth_data.get('video_info', {})
    fps = video_info.get('fps', 30.0)
    
    logger.info(f"Etiketlenmis olay sayisi: {len(ground_truth_labels)}")
    
    # 4. Metrikleri hesapla
    metrics_calc = MetricsCalculator(iou_threshold=iou_threshold)
    metrics = metrics_calc.calculate_metrics(
        ground_truth=ground_truth_labels,
        detected=detected_events,
        fps=fps
    )
    
    # 5. Sonuçları göster
    metrics_calc.print_metrics(metrics)
    
    # 6. Detaylı karşılaştırma
    print("\n" + "=" * 60)
    print("DETAYLI KARSILASTIRMA")
    print("=" * 60)
    
    matches = metrics_calc.match_events(ground_truth_labels, detected_events, fps)
    
    print(f"\nEslestirilen Olaylar ({len(matches)}):")
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match.ground_truth['event_type'].upper()}:")
        print(f"   Etiket:    {match.ground_truth['start_time']:.2f}s - {match.ground_truth['end_time']:.2f}s "
              f"(Frame {match.ground_truth['start_frame']}-{match.ground_truth['end_frame']})")
        print(f"   Tespit:    {match.detected.frame_start/fps:.2f}s - {match.detected.frame_end/fps:.2f}s "
              f"(Frame {match.detected.frame_start}-{match.detected.frame_end})")
        print(f"   IoU:       {match.iou:.3f}")
        print(f"   Tip:       {match.match_type}")
        print(f"   Confidence: {match.detected.confidence:.3f}")
    
    # False Positives
    matched_detected = set()
    for match in matches:
        for i, det in enumerate(detected_events):
            if (det.frame_start == match.detected.frame_start and
                det.frame_end == match.detected.frame_end):
                matched_detected.add(i)
                break
    
    false_positives = [detected_events[i] for i in range(len(detected_events)) if i not in matched_detected]
    if false_positives:
        print(f"\nFalse Positives ({len(false_positives)}):")
        for i, fp in enumerate(false_positives, 1):
            print(f"  {i}. {fp.event_type.upper()}: Frame {fp.frame_start}-{fp.frame_end} "
                  f"({fp.frame_start/fps:.2f}s - {fp.frame_end/fps:.2f}s), "
                  f"Confidence: {fp.confidence:.3f}")
    
    # False Negatives
    matched_gt = set()
    for match in matches:
        for i, gt in enumerate(ground_truth_labels):
            if (gt['start_frame'] == match.ground_truth['start_frame'] and
                gt['end_frame'] == match.ground_truth['end_frame']):
                matched_gt.add(i)
                break
    
    false_negatives = [ground_truth_labels[i] for i in range(len(ground_truth_labels)) if i not in matched_gt]
    if false_negatives:
        print(f"\nFalse Negatives ({len(false_negatives)}):")
        for i, fn in enumerate(false_negatives, 1):
            print(f"  {i}. {fn['event_type'].upper()}: Frame {fn['start_frame']}-{fn['end_frame']} "
                  f"({fn['start_time']:.2f}s - {fn['end_time']:.2f}s)")
    
    # Sonuçları kaydet
    output_file = Path("data/results") / f"{video_path.stem}_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'video_path': str(video_path),
        'label_file': str(label_file),
        'metrics': {
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'accuracy': metrics.accuracy,
            'tp': metrics.true_positives,
            'fp': metrics.false_positives,
            'fn': metrics.false_negatives
        },
        'per_event_metrics': metrics.per_event_metrics,
        'iou_threshold': iou_threshold
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSonuclar kaydedildi: {output_file}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model test scripti')
    parser.add_argument('--video', type=str, default='data/input/nba_test_video.mp4',
                       help='Video dosya yolu')
    parser.add_argument('--labels', type=str, default='data/labels/nba_test_video_labels.json',
                       help='Etiket dosya yolu')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU eşik değeri (0-1 arası)')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    label_file = Path(args.labels)
    
    if not video_path.exists():
        print(f"HATA: Video bulunamadi: {video_path}")
        sys.exit(1)
    
    if not label_file.exists():
        print(f"HATA: Etiket dosyasi bulunamadi: {label_file}")
        sys.exit(1)
    
    test_model_on_labeled_video(video_path, label_file, args.iou)

