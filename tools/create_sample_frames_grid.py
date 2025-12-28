"""
Veri Setinden Örnek Frame Grid Görseli Oluşturma
Video frame'lerinden rastgele örnekler alıp grid halinde gösterir
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys
import json
from collections import defaultdict

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def extract_frame_from_video(video_path: Path, frame_number: int = None):
    """
    Videodan bir frame çıkar
    
    Args:
        video_path: Video dosyası yolu
        frame_number: Frame numarası (None ise rastgele)
    
    Returns:
        Frame (numpy array) veya None
    """
    if not video_path.exists():
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Frame numarasını belirle
    if frame_number is None:
        frame_number = random.randint(0, total_frames - 1)
    else:
        frame_number = min(frame_number, total_frames - 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # BGR'den RGB'ye çevir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    return None


def get_sample_videos_from_features(features_file: Path, samples_per_category: int = 9):
    """
    Feature dosyasından örnek videoları al
    
    Args:
        features_file: Feature JSON dosyası
        samples_per_category: Her kategoriden kaç örnek alınacak
    
    Returns:
        Dict: {category: [video_paths]}
    """
    with open(features_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data['features']
    
    # Kategorilere göre grupla
    videos_by_category = defaultdict(set)
    
    for feature in features:
        event_type = feature.get('event_type', 'unknown')
        # Video path'i feature'dan al (eğer varsa)
        video_info = feature.get('video_path', '')
        
        # Label dosyalarından video path'lerini bul
        # Önce label dosyalarını kontrol et
        label_dir = Path('data/labels')
        if label_dir.exists():
            for label_file in label_dir.glob('*_labels.json'):
                with open(label_file, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    video_path = label_data.get('video_path', '')
                    if video_path and Path(video_path).exists():
                        videos_by_category[event_type].add(video_path)
    
    # Eğer label dosyalarından bulamazsak, input klasöründen al
    input_dir = Path('data/input')
    if input_dir.exists():
        video_files = list(input_dir.glob('*.mp4'))
        if video_files:
            # Kategorilere rastgele dağıt
            categories = list(videos_by_category.keys()) if videos_by_category else ['basket', 'pas']
            for video_file in video_files[:samples_per_category * len(categories)]:
                category = random.choice(categories)
                videos_by_category[category].add(str(video_file))
    
    # Her kategoriden örnek sayısını sınırla
    result = {}
    for category, video_set in videos_by_category.items():
        video_list = list(video_set)[:samples_per_category]
        result[category] = video_list
    
    return result


def get_clip_videos(use_original_only=True):
    """
    Clip klasörlerinden video dosyalarını al
    
    Args:
        use_original_only: Sadece orijinal klipleri kullan (augment edilmemiş)
    """
    clips_by_category = defaultdict(list)
    
    # Pass clips
    pass_clips_dir = Path('data/input/clips/pass_clips')
    if pass_clips_dir.exists():
        if use_original_only:
            # Sadece orijinal klipleri al (augment edilmemiş olanlar)
            # Orijinal: pass_001.mp4, pass_002.mp4 gibi (brightness, flip, rotate olmayan)
            for video_file in pass_clips_dir.glob('pass_*.mp4'):
                filename = video_file.name
                # Augment edilmemiş klipler: brightness, flip, rotate içermeyenler
                if '_brightness' not in filename and '_flip' not in filename and '_rotate' not in filename:
                    clips_by_category['pas'].append(str(video_file))
        else:
            # Tüm klipleri al
            for video_file in pass_clips_dir.glob('*.mp4'):
                clips_by_category['pas'].append(str(video_file))
    
    # Basket clips
    basket_clips_dir = Path('data/input/clips/basket_clips')
    if basket_clips_dir.exists():
        if use_original_only:
            # Sadece orijinal klipleri al (augment edilmemiş olanlar)
            for video_file in basket_clips_dir.glob('basket_*.mp4'):
                filename = video_file.name
                # Augment edilmemiş klipler: brightness, flip, rotate içermeyenler
                if '_brightness' not in filename and '_flip' not in filename and '_rotate' not in filename:
                    clips_by_category['basket'].append(str(video_file))
        else:
            # Tüm klipleri al
            for video_file in basket_clips_dir.glob('*.mp4'):
                clips_by_category['basket'].append(str(video_file))
    
    return clips_by_category


def create_sample_frames_grid(clips_by_category: dict, samples_per_category: int = 9, output_path=None):
    """
    Örnek frame'lerden grid görseli oluştur
    
    Args:
        clips_by_category: {category: [video_paths]}
        samples_per_category: Her kategoriden kaç örnek
        output_path: Çıktı dosyası yolu
    """
    # Türkçe kategori isimleri
    category_names = {
        'basket': 'Basket',
        'pas': 'Pas',
        'blok': 'Blok'
    }
    
    # Kategorileri sırala
    categories = sorted(clips_by_category.keys())
    
    if not categories:
        print("HATA: Hiç video bulunamadı!")
        return
    
    # Grid boyutları
    num_rows = len(categories)
    num_cols = samples_per_category
    
    # Figür oluştur
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
    
    # Tek satır için axes'i düzleştir
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    frame_idx = 0
    
    for cat_idx, category in enumerate(categories):
        category_tr = category_names.get(category, category.capitalize())
        video_files = clips_by_category[category]
        
        # Rastgele örnekler seç
        if len(video_files) > samples_per_category:
            selected_videos = random.sample(video_files, samples_per_category)
        else:
            selected_videos = video_files
        
        for vid_idx, video_path in enumerate(selected_videos):
            if frame_idx >= len(axes):
                break
            
            ax = axes[frame_idx]
            
            # Frame çıkar
            frame = extract_frame_from_video(Path(video_path))
            
            if frame is not None:
                ax.imshow(frame)
                ax.axis('off')
                
                # İlk satırda kategori ismini göster
                if vid_idx == 0:
                    ax.text(0.5, -0.1, category_tr, transform=ax.transAxes,
                           ha='center', va='top', fontsize=12, fontweight='bold')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10)
            
            frame_idx += 1
    
    # Boş axes'leri kapat
    for i in range(frame_idx, len(axes)):
        axes[i].axis('off')
    
    # Başlık
    total_samples = sum(len(videos) for videos in clips_by_category.values())
    title_text = f'Şekil X. Veri setinden rastgele seçilen örnek görüntüler.'
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Örnek frame grid görseli kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Örnek Frame Grid Görseli Oluştur')
    parser.add_argument(
        '--samples-per-category',
        type=int,
        default=9,
        help='Her kategoriden kaç örnek alınacak (default: 9)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Görsel çıktı dosyası yolu'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/dataset/features.json',
        help='Feature dosyasi yolu (video path\'leri icin)'
    )
    
    args = parser.parse_args()
    
    # Önce clip klasörlerinden al (sadece orijinal klipleri kullan - daha net görseller için)
    clips_by_category = get_clip_videos(use_original_only=True)
    
    # Eğer clip yoksa, feature dosyasından dene
    if not clips_by_category or sum(len(v) for v in clips_by_category.values()) == 0:
        print("Clip klasörlerinde video bulunamadı, feature dosyasından kontrol ediliyor...")
        features_file = Path(args.features)
        if features_file.exists():
            clips_by_category = get_sample_videos_from_features(features_file, args.samples_per_category)
    
    # Hala yoksa, input klasöründen al
    if not clips_by_category or sum(len(v) for v in clips_by_category.values()) == 0:
        print("Input klasöründen video alınıyor...")
        input_dir = Path('data/input')
        if input_dir.exists():
            video_files = list(input_dir.glob('*.mp4'))
            if video_files:
                # Kategorilere rastgele dağıt
                categories = ['basket', 'pas']
                for video_file in video_files[:args.samples_per_category * len(categories)]:
                    category = random.choice(categories)
                    if category not in clips_by_category:
                        clips_by_category[category] = []
                    clips_by_category[category].append(str(video_file))
    
    if not clips_by_category or sum(len(v) for v in clips_by_category.values()) == 0:
        print("HATA: Hiç video dosyası bulunamadı!")
        print("Video dosyalarını şu klasörlere koyabilirsiniz:")
        print("  - data/input/clips/pass_clips/")
        print("  - data/input/clips/basket_clips/")
        print("  - data/input/")
        return 1
    
    print(f"Bulunan videolar:")
    for category, videos in clips_by_category.items():
        print(f"  {category}: {len(videos)} video")
    
    # Çıktı dosyası
    output_path = args.output
    if not output_path:
        output_dir = Path("data/results/metrics_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "sample_frames_grid.png"
    
    print(f"\nÖrnek frame grid görseli oluşturuluyor...")
    create_sample_frames_grid(clips_by_category, args.samples_per_category, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())

