"""
Model doğruluk metrikleri hesaplama
Precision, Recall, F1-Score, Confusion Matrix
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from ..ai.event_detection import BasketballEvent


@dataclass
class EventMatch:
    """Etiket ve tespit eşleşmesi"""
    ground_truth: Dict  # Etiketlenmiş olay
    detected: BasketballEvent  # Tespit edilen olay
    iou: float  # Intersection over Union (zaman bazlı)
    match_type: str  # "exact", "partial", "miss"


@dataclass
class MetricsResult:
    """Metrik sonuçları"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    per_event_metrics: Dict[str, Dict[str, float]]


class MetricsCalculator:
    """Doğruluk metrikleri hesaplayıcı"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: Zaman bazlı IoU eşik değeri (0-1 arası)
        """
        self.iou_threshold = iou_threshold
    
    def calculate_time_iou(
        self,
        start1: float, end1: float,
        start2: float, end2: float
    ) -> float:
        """Zaman bazlı IoU hesapla"""
        # Kesişim
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)
        
        # Birleşim
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = union_end - union_start
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def match_events(
        self,
        ground_truth: List[Dict],
        detected: List[BasketballEvent],
        fps: float
    ) -> List[EventMatch]:
        """
        Etiketlenmiş olayları tespit edilen olaylarla eşleştir
        
        Args:
            ground_truth: Etiketlenmiş olaylar (JSON formatı)
            detected: Tespit edilen olaylar
            fps: Video FPS
        
        Returns:
            Eşleşmeler listesi
        """
        matches = []
        used_detected = set()
        
        # Her etiketlenmiş olay için en iyi eşleşmeyi bul
        for gt in ground_truth:
            gt_start_time = gt['start_time']
            gt_end_time = gt['end_time']
            gt_type = gt['event_type']
            
            best_match = None
            best_iou = 0
            best_detected_idx = -1
            
            for i, det in enumerate(detected):
                if i in used_detected:
                    continue
                
                # Frame'leri zamana çevir
                det_start_time = det.frame_start / fps
                det_end_time = det.frame_end / fps
                
                # Aynı olay tipi mi?
                if det.event_type != gt_type:
                    continue
                
                # IoU hesapla
                iou = self.calculate_time_iou(
                    gt_start_time, gt_end_time,
                    det_start_time, det_end_time
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
                    best_detected_idx = i
            
            # Eşik değerinden yüksekse eşleştir
            if best_iou >= self.iou_threshold:
                match_type = "exact" if best_iou >= 0.8 else "partial"
                matches.append(EventMatch(
                    ground_truth=gt,
                    detected=best_match,
                    iou=best_iou,
                    match_type=match_type
                ))
                used_detected.add(best_detected_idx)
        
        return matches
    
    def calculate_metrics(
        self,
        ground_truth: List[Dict],
        detected: List[BasketballEvent],
        fps: float
    ) -> MetricsResult:
        """
        Metrikleri hesapla
        
        Args:
            ground_truth: Etiketlenmiş olaylar
            detected: Tespit edilen olaylar
            fps: Video FPS
        
        Returns:
            Metrik sonuçları
        """
        matches = self.match_events(ground_truth, detected, fps)
        
        # True Positives: Eşleşen olaylar
        tp = len(matches)
        
        # False Positives: Tespit edilen ama eşleşmeyen olaylar
        matched_detected_indices = set()
        for match in matches:
            # detected listesindeki index'i bul
            for i, det in enumerate(detected):
                if (det.frame_start == match.detected.frame_start and
                    det.frame_end == match.detected.frame_end and
                    det.event_type == match.detected.event_type):
                    matched_detected_indices.add(i)
                    break
        
        fp = len(detected) - len(matched_detected_indices)
        
        # False Negatives: Etiketlenmiş ama tespit edilmeyen olaylar
        fn = len(ground_truth) - tp
        
        # Metrikler
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        # Olay tipine göre metrikler
        per_event_metrics = {}
        event_types = set([gt['event_type'] for gt in ground_truth])
        
        for event_type in event_types:
            gt_filtered = [gt for gt in ground_truth if gt['event_type'] == event_type]
            det_filtered = [det for det in detected if det.event_type == event_type]
            
            matches_filtered = self.match_events(gt_filtered, det_filtered, fps)
            tp_event = len(matches_filtered)
            fp_event = len(det_filtered) - tp_event
            fn_event = len(gt_filtered) - tp_event
            
            precision_event = tp_event / (tp_event + fp_event) if (tp_event + fp_event) > 0 else 0.0
            recall_event = tp_event / (tp_event + fn_event) if (tp_event + fn_event) > 0 else 0.0
            f1_event = 2 * (precision_event * recall_event) / (precision_event + recall_event) if (precision_event + recall_event) > 0 else 0.0
            
            per_event_metrics[event_type] = {
                'precision': precision_event,
                'recall': recall_event,
                'f1_score': f1_event,
                'tp': tp_event,
                'fp': fp_event,
                'fn': fn_event
            }
        
        return MetricsResult(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            per_event_metrics=per_event_metrics
        )
    
    def print_metrics(self, metrics: MetricsResult):
        """Metrikleri yazdır"""
        print("\n" + "=" * 60)
        print("MODEL DOGRULUK METRIKLERI")
        print("=" * 60)
        
        print(f"\nGenel Metrikler:")
        print(f"  Precision: {metrics.precision:.3f} ({metrics.precision*100:.1f}%)")
        print(f"  Recall:    {metrics.recall:.3f} ({metrics.recall*100:.1f}%)")
        print(f"  F1-Score:  {metrics.f1_score:.3f} ({metrics.f1_score*100:.1f}%)")
        print(f"  Accuracy:  {metrics.accuracy:.3f} ({metrics.accuracy*100:.1f}%)")
        
        print(f"\nDetaylar:")
        print(f"  True Positives (TP):  {metrics.true_positives}")
        print(f"  False Positives (FP): {metrics.false_positives}")
        print(f"  False Negatives (FN): {metrics.false_negatives}")
        
        print(f"\nOlay Tipine Gore Metrikler:")
        for event_type, event_metrics in metrics.per_event_metrics.items():
            print(f"\n  {event_type.upper()}:")
            print(f"    Precision: {event_metrics['precision']:.3f} ({event_metrics['precision']*100:.1f}%)")
            print(f"    Recall:    {event_metrics['recall']:.3f} ({event_metrics['recall']*100:.1f}%)")
            print(f"    F1-Score:  {event_metrics['f1_score']:.3f} ({event_metrics['f1_score']*100:.1f}%)")
            print(f"    TP: {event_metrics['tp']}, FP: {event_metrics['fp']}, FN: {event_metrics['fn']}")
        
        print("\n" + "=" * 60)

