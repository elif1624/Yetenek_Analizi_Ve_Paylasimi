"""
Klipleri (pass_clips ve basket_clips) veri setine ekleme aracı
Her klip için otomatik analiz yapar ve feature extraction yapar
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import logging

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))  # tools klasörünü de ekle

from analyze_video_final import analyze_video_final
from extract_features import extract_features_for_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_label_for_clip(video_path: Path, event_type: str, analysis: Dict) -> Dict:
    """
    Klip için otomatik etiket oluştur (tüm video olay olarak işaretlenir)
    """
    video_metadata = analysis.get('video_metadata', {})
    duration = video_metadata.get('duration', 0.0)
    
    label = {
        'event_type': event_type,
        'start_time': 0.0,
        'end_time': duration,
        'start_frame': 0,
        'end_frame': int(duration * video_metadata.get('fps', 30.0)),
        'confidence': 1.0,
        'source': 'clip',
        'video_path': str(video_path)
    }
    
    return label


def process_clip(video_path: Path, event_type: str, output_dir: Path) -> Dict:
    """
    Bir klip için analiz yap ve feature çıkar
    
    Returns:
        Feature dictionary veya None
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"İşleniyor: {video_path.name} ({event_type})")
    logger.info(f"{'='*60}")
    
    # Video analizi yap
    logger.info("Video analizi yapılıyor...")
    try:
        analysis = analyze_video_final(
            video_path=video_path,
            text_prompt="basketball player",
            conf_threshold=0.25,
            fps=10.0,
            enable_event_detection=False
        )
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")
        return None
    
    # Etiket oluştur (tüm video olay olarak)
    label = create_label_for_clip(video_path, event_type, analysis)
    
    # Feature extraction
    logger.info("Feature extraction yapılıyor...")
    video_metadata = analysis.get('video_metadata', {})
    video_fps = video_metadata.get('fps', 30.0)
    extraction_fps = analysis.get('analysis_params', {}).get('extraction_fps', 10.0)
    
    try:
        features = extract_features_for_event(
            label=label,
            analysis=analysis,
            video_fps=video_fps,
            extraction_fps=extraction_fps
        )
        
        if features:
            logger.info(f"✓ Feature çıkarıldı: {len(features)} feature")
            
            # Analiz ve etiket dosyalarını kaydet (isteğe bağlı)
            video_stem = video_path.stem
            analysis_path = output_dir / "results" / f"{video_stem}_final_analysis.json"
            label_path = output_dir / "labels" / f"{video_stem}_labels.json"
            
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Analiz sonuçlarını kaydet
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            # Etiket dosyasını kaydet (standart format)
            label_data = {
                'video_path': str(video_path),
                'video_info': video_metadata,
                'labels': [label]
            }
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, indent=2, ensure_ascii=False)
            
            return features
        else:
            logger.warning(f"✗ Yeterli veri yok, feature çıkarılamadı")
            return None
            
    except Exception as e:
        logger.error(f"Feature extraction hatası: {e}", exc_info=True)
        return None


def process_clips_directory(clips_dir: Path, event_type: str, output_dir: Path) -> List[Dict]:
    """
    Bir klasördeki tüm klipleri işle
    
    Returns:
        Feature listesi
    """
    if not clips_dir.exists():
        logger.warning(f"Klasör bulunamadı: {clips_dir}")
        return []
    
    # Video dosyalarını bul
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(clips_dir.glob(f'*{ext}'))
        video_files.extend(clips_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        logger.warning(f"Video dosyası bulunamadı: {clips_dir}")
        return []
    
    logger.info(f"\n{len(video_files)} klip bulundu ({event_type}): {clips_dir}")
    
    all_features = []
    success_count = 0
    fail_count = 0
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\n[{i}/{len(video_files)}] {video_path.name}")
        
        features = process_clip(video_path, event_type, output_dir)
        
        if features:
            all_features.append(features)
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"\n{'-'*60}")
    logger.info(f"Tamamlandı: {success_count} başarılı, {fail_count} başarısız")
    logger.info(f"{'-'*60}")
    
    return all_features


def merge_features(existing_features_path: Path, new_features: List[Dict], output_path: Path):
    """
    Yeni feature'ları mevcut feature'larla birleştir
    """
    # Mevcut feature'ları yükle
    existing_features = []
    if existing_features_path.exists():
        with open(existing_features_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_features = data.get('features', [])
        logger.info(f"Mevcut feature sayısı: {len(existing_features)}")
    
    # Yeni feature'ları ekle
    all_features = existing_features + new_features
    logger.info(f"Yeni feature sayısı: {len(new_features)}")
    logger.info(f"Toplam feature sayısı: {len(all_features)}")
    
    # Event dağılımını göster
    event_counts = {}
    for f in all_features:
        event_type = f.get('event_type', 'unknown')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    logger.info(f"\nEvent dağılımı:")
    for event_type, count in sorted(event_counts.items()):
        logger.info(f"  {event_type}: {count}")
    
    # Kaydet
    output_data = {
        'total_samples': len(all_features),
        'features': all_features
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Birleştirilmiş feature'lar kaydedildi: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Klipleri veri setine ekle')
    parser.add_argument(
        '--pass-clips',
        type=str,
        required=True,
        help='Pass klipleri klasörü yolu'
    )
    parser.add_argument(
        '--basket-clips',
        type=str,
        required=True,
        help='Basket klipleri klasörü yolu'
    )
    parser.add_argument(
        '--existing-features',
        type=str,
        default='data/dataset/features.json',
        help='Mevcut feature dosyası yolu'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/dataset/features.json',
        help='Çıktı feature dosyası yolu (birleştirilmiş)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Analiz ve etiket çıktı dizini'
    )
    
    args = parser.parse_args()
    
    pass_clips_dir = Path(args.pass_clips)
    basket_clips_dir = Path(args.basket_clips)
    existing_features_path = Path(args.existing_features)
    output_path = Path(args.output)
    output_dir = Path(args.output_dir)
    
    # Klasörleri kontrol et
    if not pass_clips_dir.exists():
        logger.error(f"Pass klipleri klasörü bulunamadı: {pass_clips_dir}")
        return 1
    
    if not basket_clips_dir.exists():
        logger.error(f"Basket klipleri klasörü bulunamadı: {basket_clips_dir}")
        return 1
    
    logger.info("="*70)
    logger.info("KLİPLER VERİ SETİNE EKLENİYOR")
    logger.info("="*70)
    logger.info(f"Pass klipleri: {pass_clips_dir}")
    logger.info(f"Basket klipleri: {basket_clips_dir}")
    logger.info(f"Mevcut features: {existing_features_path}")
    logger.info(f"Çıktı: {output_path}")
    logger.info("="*70)
    
    # Pass kliplerini işle
    pass_features = process_clips_directory(pass_clips_dir, 'pas', output_dir)
    
    # Basket kliplerini işle
    basket_features = process_clips_directory(basket_clips_dir, 'basket', output_dir)
    
    # Tüm feature'ları birleştir
    all_new_features = pass_features + basket_features
    
    if all_new_features:
        merge_features(existing_features_path, all_new_features, output_path)
        logger.info(f"\n{'='*70}")
        logger.info("İŞLEM TAMAMLANDI!")
        logger.info(f"{'='*70}")
        logger.info(f"Toplam {len(all_new_features)} yeni feature eklendi")
        logger.info(f"Güncellenmiş feature dosyası: {output_path}")
        logger.info(f"\nModeli yeniden eğitmek için:")
        logger.info(f"  python tools/train_model.py --features {output_path}")
        return 0
    else:
        logger.error("\nHiç feature çıkarılamadı!")
        return 1


if __name__ == "__main__":
    exit(main())

