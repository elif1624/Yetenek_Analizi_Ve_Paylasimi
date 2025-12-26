"""
Manuel Etiketleme Aracı
Basketbol videolarındaki olayları manuel olarak işaretlemek için
"""

import cv2
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# FFmpeg/h264 uyarılarını gizle
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Sadece kritik hataları göster
# stderr'i yönlendirerek h264 uyarılarını gizle
if hasattr(sys.stderr, 'fileno'):
    try:
        # Windows'ta stderr'i dev/null'a yönlendir
        if os.name == 'nt':
            sys.stderr = open(os.devnull, 'w')
    except:
        pass

@dataclass
class EventLabel:
    """Etiketlenmiş olay"""
    event_type: str  # "basket", "pas"
    start_time: float  # Saniye cinsinden
    end_time: float
    start_frame: int
    end_frame: int
    player_track_id: Optional[int] = None  # İlgili oyuncu ID (varsa)
    confidence: float = 1.0  # Manuel etiketleme için genelde 1.0
    notes: Optional[str] = None  # Ek notlar
    position: Optional[Tuple[float, float]] = None  # Olay pozisyonu (varsa)


class LabelingTool:
    """Manuel etiketleme aracı"""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Video bilgileri
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps
        
        # Etiketler
        self.labels: List[EventLabel] = []
        self.current_frame = 0
        self.playing = False
        self.current_event_start = None
        self.current_event_type = None
        
        # Output path
        self.output_path = Path("data/labels") / f"{video_path.stem}_labels.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Video yuklendi: {video_path.name}")
        print(f"  Sure: {self.duration:.1f} saniye")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Frame sayisi: {self.frame_count}")
        print(f"\nKontroller:")
        print(f"  SPACE: Oynat/Duraklat")
        print(f"  LEFT/RIGHT veya A/D: Frame ileri/geri")
        print(f"  B: Basket (Gol) etiketle")
        print(f"  P: Pas etiketle")
        print(f"  ENTER: Etiketi kaydet")
        print(f"  ESC veya Q: Cikis ve kaydet")
        print(f"  DEL: Son etiketi sil")
        print(f"  X: Mevcut frame'deki etiketleri sil")
        print(f"  L: Tum etiketleri listele")
        print(f"\nNOT: Pencereye tiklayip odaklanin, sonra tuslara basin!")
    
    def get_frame(self, frame_num: int) -> Optional[Tuple[int, any]]:
        """Belirli bir frame'i al"""
        if frame_num < 0 or frame_num >= self.frame_count:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            return (frame_num, frame)
        return None
    
    def frame_to_time(self, frame_num: int) -> float:
        """Frame numarasını zamana çevir"""
        return frame_num / self.fps
    
    def time_to_frame(self, time: float) -> int:
        """Zamanı frame numarasına çevir"""
        return int(time * self.fps)
    
    def draw_overlay(self, frame, frame_num: int) -> None:
        """Frame üzerine bilgileri çiz"""
        overlay = frame.copy()
        
        # Zaman bilgisi
        current_time = self.frame_to_time(frame_num)
        time_text = f"Time: {current_time:.2f}s / {self.duration:.2f}s"
        frame_text = f"Frame: {frame_num} / {self.frame_count}"
        
        cv2.putText(overlay, time_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, frame_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Etiket sayısı
        label_text = f"Etiketler: {len(self.labels)}"
        cv2.putText(overlay, label_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Mevcut olay (etiketleme sırasında)
        if self.current_event_start is not None:
            event_text = f"Etiketleniyor: {self.current_event_type.upper()}"
            start_time = self.frame_to_time(self.current_event_start)
            cv2.putText(overlay, event_text, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(overlay, f"Baslangic: {start_time:.2f}s", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Mevcut frame'deki olaylar (vurgulu göster)
        y_offset = 230
        current_frame_labels = []
        for label in self.labels:
            if label.start_frame <= frame_num <= label.end_frame:
                current_frame_labels.append(label)
                # Kırmızı arka plan ile vurgula
                event_info = f"{label.event_type.upper()}: {label.start_time:.1f}s-{label.end_time:.1f}s"
                # Arka plan
                (text_width, text_height), baseline = cv2.getTextSize(
                    event_info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(overlay, 
                            (5, y_offset - text_height - 5),
                            (text_width + 10, y_offset + baseline + 5),
                            (0, 0, 255), -1)  # Kırmızı arka plan
                # Metin
                cv2.putText(overlay, event_info, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 35
        
        # Yarı saydam overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    def start_event_labeling(self, event_type: str):
        """Olay etiketlemeye başla"""
        if self.current_event_start is not None:
            print(f"UYARI: Zaten bir olay etiketleniyor ({self.current_event_type})")
            return
        
        self.current_event_start = self.current_frame
        self.current_event_type = event_type
        print(f"{event_type.upper()} etiketleme basladi (Frame {self.current_frame}, {self.frame_to_time(self.current_frame):.2f}s)")
    
    def finish_event_labeling(self):
        """Olay etiketlemeyi bitir ve kaydet"""
        if self.current_event_start is None:
            print("UYARI: Etiketlenecek olay yok")
            return
        
        start_frame = self.current_event_start
        end_frame = self.current_frame
        
        if end_frame <= start_frame:
            print("UYARI: Bitis frame'i baslangictan once olamaz")
            return
        
        start_time = self.frame_to_time(start_frame)
        end_time = self.frame_to_time(end_frame)
        
        label = EventLabel(
            event_type=self.current_event_type,
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            confidence=1.0
        )
        
        self.labels.append(label)
        print(f"{self.current_event_type.upper()} etiketi kaydedildi: "
              f"{start_time:.2f}s - {end_time:.2f}s (Frame {start_frame}-{end_frame})")
        
        # Otomatik kaydet (veri kaybını önlemek için)
        self.save_labels()
        print(f"  -> Otomatik kaydedildi: {len(self.labels)} etiket")
        
        self.current_event_start = None
        self.current_event_type = None
    
    def delete_last_label(self):
        """Son etiketi sil"""
        if self.labels:
            deleted = self.labels.pop()
            print(f"Son etiket silindi: {deleted.event_type} ({deleted.start_time:.2f}s-{deleted.end_time:.2f}s)")
        else:
            print("Silinecek etiket yok")
    
    def delete_labels_at_frame(self, frame_num: int):
        """Mevcut frame'deki tüm etiketleri sil"""
        labels_to_remove = []
        
        for label in self.labels:
            if label.start_frame <= frame_num <= label.end_frame:
                labels_to_remove.append(label)
        
        if not labels_to_remove:
            print(f"Frame {frame_num}'de ({self.frame_to_time(frame_num):.2f}s) etiket bulunamadi")
            return
        
        for label in labels_to_remove:
            self.labels.remove(label)
            print(f"Etiket silindi: {label.event_type.upper()} "
                  f"({label.start_time:.2f}s-{label.end_time:.2f}s, Frame {label.start_frame}-{label.end_frame})")
        
        print(f"Toplam {len(labels_to_remove)} etiket silindi")
    
    def show_labels_list(self):
        """Tüm etiketleri listele"""
        if not self.labels:
            print("Henuz etiket yok")
            return
        
        print(f"\n{'='*60}")
        print(f"TOPLAM {len(self.labels)} ETIKET:")
        print(f"{'='*60}")
        
        for i, label in enumerate(self.labels, 1):
            # Mevcut frame bu etiketin içinde mi?
            is_current = label.start_frame <= self.current_frame <= label.end_frame
            marker = " <-- MEVCUT FRAME" if is_current else ""
            
            print(f"{i}. {label.event_type.upper():8s} | "
                  f"Frame: {label.start_frame:4d}-{label.end_frame:4d} | "
                  f"Zaman: {label.start_time:6.2f}s - {label.end_time:6.2f}s{marker}")
        
        print(f"{'='*60}\n")
    
    def save_labels(self):
        """Etiketleri kaydet"""
        output_data = {
            'video_path': str(self.video_path),
            'video_info': {
                'fps': self.fps,
                'frame_count': self.frame_count,
                'duration': self.duration,
                'width': self.width,
                'height': self.height
            },
            'labels': [asdict(label) for label in self.labels],
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nEtiketler kaydedildi: {self.output_path}")
        print(f"Toplam {len(self.labels)} etiket")
    
    def load_labels(self):
        """Kaydedilmiş etiketleri yükle"""
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.labels = [EventLabel(**label) for label in data.get('labels', [])]
            print(f"{len(self.labels)} etiket yuklendi")
    
    def run(self):
        """Ana döngü"""
        self.load_labels()
        
        window_name = 'Etiketleme Araci'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Pencereyi üste getir
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)  # Sonra normal yap (odaklanma için)
        
        frame_result = self.get_frame(self.current_frame)
        if frame_result is None:
            print("HATA: Frame alinamadi")
            return
        
        _, frame = frame_result
        
        while True:
            # Overlay çiz
            display_frame = frame.copy()
            self.draw_overlay(display_frame, self.current_frame)
            
            cv2.imshow('Etiketleme Araci', display_frame)
            
            # Windows'ta daha güvenilir tuş yakalama
            # waitKey(30) = 30ms bekleme, bu süre içinde tuş basılırsa yakala
            key_code = cv2.waitKey(30)
            
            # Eğer hiçbir tuş basılmadıysa (-1 döner)
            if key_code == -1:
                # Auto-play için frame güncellemesi
                if self.playing:
                    self.current_frame += 1
                    if self.current_frame >= self.frame_count:
                        self.current_frame = 0
                        self.playing = False
                    
                    frame_result = self.get_frame(self.current_frame)
                    if frame_result:
                        _, frame = frame_result
                continue
            
            # Tuş kodunu al (son 8 bit)
            key = key_code & 0xFF
            
            # Windows'ta ok tuşları için özel kontrol
            # Ok tuşları 224 ile başlar, sonraki byte gerçek kod
            if key_code == 224:  # Windows'ta ok tuşu başlangıcı
                key_code = cv2.waitKey(30)
                if key_code != -1:
                    key = key_code & 0xFF
            
            # Debug: Basılan tuşu göster (geliştirme için - aktif edilebilir)
            # print(f"Basilan tus kodu: {key_code}, key: {key} (char: {chr(key) if 32 <= key <= 126 else 'special'})")
            
            if key == ord('q') or key == 27:  # ESC veya 'q'
                print("Cikis yapiliyor...")
                break
            elif key == ord(' '):  # SPACE - Play/Pause
                self.playing = not self.playing
                print(f"Oynatma: {'Aktif' if self.playing else 'Duraklatildi'}")
            elif key == 81 or key == 2 or key == 0 or key == 75:  # LEFT arrow (Windows: 75)
                self.playing = False
                self.current_frame = max(0, self.current_frame - 1)
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
            elif key == 83 or key == 3 or key == 1 or key == 77:  # RIGHT arrow (Windows: 77)
                self.playing = False
                self.current_frame = min(self.frame_count - 1, self.current_frame + 1)
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
            elif key == ord('b') or key == ord('B'):  # Basket (Gol) - küçük/büyük harf
                self.start_event_labeling('basket')
            elif key == ord('p') or key == ord('P'):  # Pas - küçük/büyük harf
                self.start_event_labeling('pas')
            elif key == 13 or key == 10:  # ENTER - Finish labeling (Windows/Linux)
                self.finish_event_labeling()
            elif key == 8 or key == 127:  # DEL/Backspace - Delete last
                self.delete_last_label()
            elif key == ord('x') or key == ord('X'):  # X - Mevcut frame'deki etiketleri sil
                self.delete_labels_at_frame(self.current_frame)
            elif key == ord('l') or key == ord('L'):  # L - Etiket listesini göster
                self.show_labels_list()
            elif key == ord('a') or key == ord('A'):  # 'a' tuşu - Frame geri (alternatif)
                self.playing = False
                self.current_frame = max(0, self.current_frame - 10)
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
            elif key == ord('d') or key == ord('D'):  # 'd' tuşu - Frame ileri (alternatif)
                self.playing = False
                self.current_frame = min(self.frame_count - 1, self.current_frame + 10)
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
            
            # Auto-play için frame güncellemesi
            if self.playing:
                self.current_frame += 1
                if self.current_frame >= self.frame_count:
                    self.current_frame = 0
                    self.playing = False
                
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
            
        
        self.save_labels()
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Manuel Etiketleme Araci')
    parser.add_argument('video', type=str, help='Video dosya yolu')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"HATA: Video bulunamadi: {video_path}")
        return
    
    tool = LabelingTool(video_path)
    tool.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Kullanim: python labeling_tool.py <video_path>")
        print("Ornek: python labeling_tool.py data/input/nba_test_video.mp4")
    else:
        main()

