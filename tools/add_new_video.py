"""
Yeni video ekleme ve işleme scripti
Video yüklenir, analiz edilir, etiketleme için hazırlanır
"""

import argparse
from pathlib import Path
import sys
import subprocess

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_video_exists(video_path: Path) -> bool:
    """Video dosyasının var olup olmadığını kontrol et"""
    if not video_path.exists():
        print(f"HATA: Video bulunamadı: {video_path}")
        return False
    return True


def copy_video_to_input(video_path: Path, output_name: str = None) -> Path:
    """Videoyu data/input klasörüne kopyala"""
    input_dir = Path("data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    if output_name is None:
        output_name = video_path.name
    
    output_path = input_dir / output_name
    
    # Eğer aynı isimde dosya varsa, numara ekle
    counter = 1
    original_output = output_path
    while output_path.exists():
        stem = original_output.stem
        suffix = original_output.suffix
        output_path = input_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    
    # Dosyayı kopyala
    import shutil
    shutil.copy2(video_path, output_path)
    print(f"Video kopyalandi: {output_path}")
    
    return output_path


def run_video_analysis(video_path: Path):
    """Video analizini çalıştır"""
    print(f"\n{'='*60}")
    print("VIDEO ANALİZİ BAŞLATILIYOR")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Bu işlem birkaç dakika sürebilir...\n")
    
    # analyze_video_final.py'yi çalıştır
    cmd = [
        sys.executable,
        "analyze_video_final.py",
        "--video", str(video_path),
        "--fps", "3.0"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n{'='*60}")
        print("VIDEO ANALİZİ TAMAMLANDI")
        print(f"{'='*60}")
        return True
    else:
        print(f"\nHATA: Video analizi başarısız oldu")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Yeni video ekle ve işleme hazırla"
    )
    parser.add_argument(
        'video',
        type=str,
        help='Video dosyası yolu (yerel dosya veya YouTube URL)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Video için özel isim (opsiyonel)'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Analizi atla, sadece videoyu kopyala'
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    # Eğer URL ise, önce indir
    if args.video.startswith('http'):
        print(f"YouTube URL algilandi: {args.video}")
        print("Video indiriliyor...")
        
        # video_downloader.py'yi kullan
        from src.utils.video_downloader import download_youtube_video
        
        try:
            downloaded_path = download_youtube_video(args.video)
            video_path = Path(downloaded_path)
            print(f"Video indirildi: {video_path}")
        except Exception as e:
            print(f"HATA: Video indirilemedi: {e}")
            return 1
    else:
        # Yerel dosya kontrolü
        if not check_video_exists(video_path):
            return 1
    
    # Videoyu input klasörüne kopyala
    input_video = copy_video_to_input(video_path, args.name)
    
    # Analiz yap (eğer atlanmadıysa)
    if not args.skip_analysis:
        success = run_video_analysis(input_video)
        if not success:
            print("\nUYARI: Analiz başarısız oldu, ama video hazır.")
            print("Manuel olarak analiz edebilirsiniz:")
            print(f"  python analyze_video_final.py --video {input_video}")
    
    print(f"\n{'='*60}")
    print("VIDEO HAZIR!")
    print(f"{'='*60}")
    print(f"Video: {input_video}")
    print(f"\nSonraki adimlar:")
    print(f"1. Etiketleme:")
    print(f"   python tools/labeling_tool.py {input_video}")
    print(f"\n2. Feature extraction (etiketleme sonrası):")
    print(f"   python tools/extract_features.py")
    print(f"\n3. Model eğitimi (yeterli veri toplandığında):")
    print(f"   python tools/train_model.py")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())






