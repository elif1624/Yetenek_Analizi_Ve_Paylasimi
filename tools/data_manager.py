"""
Veri Yönetim Sistemi
Etiketlenmiş verileri organize eder, train/test split yapar
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random
import shutil

@dataclass
class DatasetInfo:
    """Dataset bilgileri"""
    total_videos: int
    total_labels: int
    labels_by_type: Dict[str, int]
    train_videos: int
    test_videos: int
    validation_videos: int


class DataManager:
    """Veri yönetim sistemi"""
    
    def __init__(self, labels_dir: Path = Path("data/labels")):
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_dir = Path("data/dataset")
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        self.validation_dir = self.dataset_dir / "validation"
        
        # Dizinleri oluştur
        for dir_path in [self.train_dir, self.test_dir, self.validation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_all_labels(self) -> List[Dict]:
        """Tüm etiket dosyalarını yükle"""
        label_files = list(self.labels_dir.glob("*_labels.json"))
        all_labels = []
        
        for label_file in label_files:
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_labels.append(data)
        
        return all_labels
    
    def get_statistics(self) -> DatasetInfo:
        """Dataset istatistiklerini hesapla"""
        all_labels = self.load_all_labels()
        
        total_videos = len(all_labels)
        total_labels = sum(len(data.get('labels', [])) for data in all_labels)
        
        labels_by_type = {}
        for data in all_labels:
            for label in data.get('labels', []):
                event_type = label.get('event_type', 'unknown')
                labels_by_type[event_type] = labels_by_type.get(event_type, 0) + 1
        
        return DatasetInfo(
            total_videos=total_videos,
            total_labels=total_labels,
            labels_by_type=labels_by_type,
            train_videos=0,
            test_videos=0,
            validation_videos=0
        )
    
    def split_dataset(
        self, 
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        validation_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """Dataset'i train/test/validation olarak böl"""
        
        if abs(train_ratio + test_ratio + validation_ratio - 1.0) > 0.01:
            raise ValueError("Oranlar toplamı 1.0 olmalı")
        
        all_labels = self.load_all_labels()
        
        if len(all_labels) < 3:
            print(f"UYARI: Sadece {len(all_labels)} video var. En az 3 video gerekli.")
            return
        
        # Rastgele sırala
        random.seed(random_seed)
        random.shuffle(all_labels)
        
        # Böl
        total = len(all_labels)
        train_count = int(total * train_ratio)
        test_count = int(total * test_ratio)
        validation_count = total - train_count - test_count
        
        train_data = all_labels[:train_count]
        test_data = all_labels[train_count:train_count + test_count]
        validation_data = all_labels[train_count + test_count:]
        
        # Kopyala
        self._copy_to_split(train_data, self.train_dir)
        self._copy_to_split(test_data, self.test_dir)
        self._copy_to_split(validation_data, self.validation_dir)
        
        print(f"Dataset bolundu:")
        print(f"  Train: {len(train_data)} video")
        print(f"  Test: {len(test_data)} video")
        print(f"  Validation: {len(validation_data)} video")
        
        return {
            'train': len(train_data),
            'test': len(test_data),
            'validation': len(validation_data)
        }
    
    def _copy_to_split(self, data_list: List[Dict], target_dir: Path):
        """Verileri hedef dizine kopyala"""
        for data in data_list:
            video_path = Path(data['video_path'])
            label_file = self.labels_dir / f"{video_path.stem}_labels.json"
            
            if label_file.exists():
                shutil.copy2(label_file, target_dir / label_file.name)
    
    def validate_labels(self) -> Dict[str, List[str]]:
        """Etiketleri doğrula"""
        all_labels = self.load_all_labels()
        errors = {
            'missing_video': [],
            'invalid_time': [],
            'invalid_event_type': [],
            'empty_labels': []
        }
        
        valid_event_types = {'basket', 'pas'}
        
        for data in all_labels:
            video_path = Path(data.get('video_path', ''))
            if not video_path.exists():
                errors['missing_video'].append(str(video_path))
            
            labels = data.get('labels', [])
            if not labels:
                errors['empty_labels'].append(str(video_path))
            
            for label in labels:
                # Event type kontrolü
                event_type = label.get('event_type', '')
                if event_type not in valid_event_types:
                    errors['invalid_event_type'].append(
                        f"{video_path.name}: {event_type}"
                    )
                
                # Zaman kontrolü
                start_time = label.get('start_time', 0)
                end_time = label.get('end_time', 0)
                if end_time <= start_time:
                    errors['invalid_time'].append(
                        f"{video_path.name}: {start_time:.2f}s - {end_time:.2f}s"
                    )
        
        return errors
    
    def print_statistics(self):
        """İstatistikleri yazdır"""
        stats = self.get_statistics()
        
        print("="*60)
        print("DATASET ISTATISTIKLERI")
        print("="*60)
        print(f"Toplam Video: {stats.total_videos}")
        print(f"Toplam Etiket: {stats.total_labels}")
        print(f"\nEtiket Tipine Gore:")
        for event_type, count in sorted(stats.labels_by_type.items()):
            print(f"  {event_type.upper()}: {count}")
        print("="*60)
    
    def export_for_training(self, output_path: Path = Path("data/dataset/dataset_info.json")):
        """Eğitim için export et"""
        all_labels = self.load_all_labels()
        
        dataset_info = {
            'total_videos': len(all_labels),
            'labels': []
        }
        
        for data in all_labels:
            video_info = {
                'video_path': data['video_path'],
                'video_info': data.get('video_info', {}),
                'labels': data.get('labels', [])
            }
            dataset_info['labels'].append(video_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset bilgileri export edildi: {output_path}")


def main():
    """Ana fonksiyon"""
    manager = DataManager()
    
    # İstatistikler
    manager.print_statistics()
    
    # Doğrulama
    print("\nEtiketler dogrulanıyor...")
    errors = manager.validate_labels()
    
    has_errors = False
    for error_type, error_list in errors.items():
        if error_list:
            has_errors = True
            print(f"\n{error_type.upper()} hatalari ({len(error_list)}):")
            for error in error_list[:5]:  # İlk 5'i göster
                print(f"  - {error}")
            if len(error_list) > 5:
                print(f"  ... ve {len(error_list) - 5} tane daha")
    
    if not has_errors:
        print("Hic hata bulunamadi!")
    
    # Split (eğer yeterli video varsa)
    all_labels = manager.load_all_labels()
    if len(all_labels) >= 3:
        print("\nDataset bolunuyor...")
        manager.split_dataset()
        manager.export_for_training()
    else:
        print(f"\nUYARI: Dataset bolmek icin en az 3 video gerekli (su anda {len(all_labels)})")


if __name__ == "__main__":
    main()

