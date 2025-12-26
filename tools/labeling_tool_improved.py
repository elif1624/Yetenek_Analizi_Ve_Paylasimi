"""
İyileştirilmiş Manuel Etiketleme Aracı
- Hızlı navigasyon
- Batch işlemler
- Akıllı öneriler
"""

import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import numpy as np

# Mevcut labeling_tool.py'den import
from labeling_tool import EventLabel, LabelingTool


class ImprovedLabelingTool(LabelingTool):
    """İyileştirilmiş etiketleme aracı"""
    
    def __init__(self, video_path: Path):
        super().__init__(video_path)
        
        # İyileştirmeler
        self.jump_size = 30  # Frame atlama boyutu (varsayılan)
        self.last_event_type = None  # Son etiketlenen olay tipi
        self.event_templates = []  # Olay şablonları
        
        print(f"\nİYİLEŞTİRİLMİŞ ÖZELLİKLER:")
        print(f"  J/K: 30 frame ileri/geri atla")
        print(f"  U/I: 10 frame ileri/geri atla")
        print(f"  R: Son olay tipini tekrar kullan")
        print(f"  T: Olay şablonu kaydet")
        print(f"  Y: Şablonu uygula")
        print(f"  N: Sonraki benzer olaya git")
    
    def jump_frames(self, direction: int, size: int = None):
        """Frame atla"""
        jump = size or self.jump_size
        if direction > 0:
            self.current_frame = min(self.frame_count - 1, self.current_frame + jump)
        else:
            self.current_frame = max(0, self.current_frame - jump)
        
        frame_result = self.get_frame(self.current_frame)
        if frame_result:
            _, frame = frame_result
            return frame
        return None
    
    def repeat_last_event_type(self):
        """Son olay tipini tekrar kullan"""
        if self.last_event_type:
            self.start_event_labeling(self.last_event_type)
            print(f"Son olay tipi kullanildi: {self.last_event_type.upper()}")
        else:
            print("Daha once olay etiketlenmemis")
    
    def save_event_template(self):
        """Mevcut olayı şablon olarak kaydet"""
        if self.current_event_start is not None:
            print("UYARI: Devam eden etiketleme var, once bitirin")
            return
        
        if not self.labels:
            print("UYARI: Kaydedilecek etiket yok")
            return
        
        # Son etiketi şablon olarak kaydet
        last_label = self.labels[-1]
        template = {
            'event_type': last_label.event_type,
            'duration': last_label.end_time - last_label.start_time,
            'frame_duration': last_label.end_frame - last_label.start_frame
        }
        self.event_templates.append(template)
        print(f"Şablon kaydedildi: {template['event_type']} ({template['duration']:.2f}s)")
    
    def apply_template(self):
        """Şablonu uygula"""
        if not self.event_templates:
            print("UYARI: Kaydedilmis sablon yok")
            return
        
        # Son şablonu kullan
        template = self.event_templates[-1]
        self.start_event_labeling(template['event_type'])
        
        # Şablon süresine göre bitiş frame'ini tahmin et
        estimated_end_frame = self.current_frame + template['frame_duration']
        print(f"Şablon uygulandi: {template['event_type']}, "
              f"tahmini bitis: Frame {estimated_end_frame}")
    
    def find_next_similar_event(self):
        """Sonraki benzer olaya git"""
        if not self.labels:
            print("UYARI: Daha once etiketlenmis olay yok")
            return
        
        # Son etiketlenen olay tipini bul
        last_label = self.labels[-1]
        event_type = last_label.event_type
        
        # Aynı tip olayları bul
        similar_events = [l for l in self.labels if l.event_type == event_type]
        
        if len(similar_events) < 2:
            print(f"UYARI: Benzer olay bulunamadi ({event_type})")
            return
        
        # Son olaydan sonraki ilk benzer olaya git
        for label in similar_events:
            if label.start_frame > last_label.end_frame:
                self.current_frame = label.start_frame
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
                    print(f"Benzer olaya gidildi: {event_type.upper()} (Frame {self.current_frame})")
                    return frame
        
        print(f"UYARI: Sonraki benzer olay bulunamadi")
    
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
        self.last_event_type = self.current_event_type  # Son olay tipini kaydet
        
        print(f"{self.current_event_type.upper()} etiketi kaydedildi: "
              f"{start_time:.2f}s - {end_time:.2f}s (Frame {start_frame}-{end_frame})")
        
        self.current_event_start = None
        self.current_event_type = None
    
    def run(self):
        """Ana döngü - iyileştirilmiş"""
        self.load_labels()
        
        cv2.namedWindow('İyileştirilmiş Etiketleme Araci', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('İyileştirilmiş Etiketleme Araci', 1280, 720)
        
        frame_result = self.get_frame(self.current_frame)
        if frame_result is None:
            print("HATA: Frame alinamadi")
            return
        
        _, frame = frame_result
        
        while True:
            # Overlay çiz
            display_frame = frame.copy()
            self.draw_overlay(display_frame, self.current_frame)
            
            cv2.imshow('İyileştirilmiş Etiketleme Araci', display_frame)
            
            # Windows'ta daha güvenilir tuş yakalama
            key_code = cv2.waitKey(30)
            
            if key_code == -1:
                if self.playing:
                    self.current_frame += 1
                    if self.current_frame >= self.frame_count:
                        self.current_frame = 0
                        self.playing = False
                    
                    frame_result = self.get_frame(self.current_frame)
                    if frame_result:
                        _, frame = frame_result
                continue
            
            key = key_code & 0xFF
            
            # Windows'ta ok tuşları için özel kontrol
            if key_code == 224:
                key_code2 = cv2.waitKey(30)
                if key_code2 != -1:
                    key = key_code2 & 0xFF
            
            if key == ord('q') or key == 27:  # ESC veya 'q'
                print("Cikis yapiliyor...")
                break
            elif key == ord(' '):  # SPACE - Play/Pause
                self.playing = not self.playing
                print(f"Oynatma: {'Aktif' if self.playing else 'Duraklatildi'}")
            elif key == 75 or key == ord('a') or key == ord('A'):  # LEFT / A
                self.playing = False
                self.current_frame = max(0, self.current_frame - 1)
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
            elif key == 77 or key == ord('d') or key == ord('D'):  # RIGHT / D
                self.playing = False
                self.current_frame = min(self.frame_count - 1, self.current_frame + 1)
                frame_result = self.get_frame(self.current_frame)
                if frame_result:
                    _, frame = frame_result
            elif key == ord('j') or key == ord('J'):  # J - 30 frame geri
                self.playing = False
                frame = self.jump_frames(-1, 30)
            elif key == ord('k') or key == ord('K'):  # K - 30 frame ileri
                self.playing = False
                frame = self.jump_frames(1, 30)
            elif key == ord('u') or key == ord('U'):  # U - 10 frame geri
                self.playing = False
                frame = self.jump_frames(-1, 10)
            elif key == ord('i') or key == ord('I'):  # I - 10 frame ileri
                self.playing = False
                frame = self.jump_frames(1, 10)
            elif key == ord('b') or key == ord('B'):  # Basket
                self.start_event_labeling('basket')
            elif key == ord('p') or key == ord('P'):  # Pas
                self.start_event_labeling('pas')
            elif key == 13 or key == 10:  # ENTER
                self.finish_event_labeling()
            elif key == ord('r') or key == ord('R'):  # R - Son olay tipini tekrarla
                self.repeat_last_event_type()
            elif key == ord('t') or key == ord('T'):  # T - Şablon kaydet
                self.save_event_template()
            elif key == ord('y') or key == ord('Y'):  # Y - Şablon uygula
                self.apply_template()
            elif key == ord('n') or key == ord('N'):  # N - Sonraki benzer olay
                self.find_next_similar_event()
            elif key == 8 or key == 127:  # DEL
                self.delete_last_label()
            elif key == ord('x') or key == ord('X'):  # X - Mevcut frame'deki etiketleri sil
                self.delete_labels_at_frame(self.current_frame)
            elif key == ord('l') or key == ord('L'):  # L - Etiket listesi
                self.show_labels_list()
            
            # Auto-play
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
    parser = argparse.ArgumentParser(description='İyileştirilmiş Manuel Etiketleme Araci')
    parser.add_argument('video', type=str, help='Video dosya yolu')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"HATA: Video bulunamadi: {video_path}")
        return
    
    tool = ImprovedLabelingTool(video_path)
    tool.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Kullanim: python labeling_tool_improved.py <video_path>")
        print("Ornek: python labeling_tool_improved.py data/input/nba_test_video.mp4")
    else:
        main()

