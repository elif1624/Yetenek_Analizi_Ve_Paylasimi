"""SAM3 modeli - Transformers ile yerel kullanım"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Optional, List, Dict, Any, Tuple
import logging

try:
    from transformers import Sam3Model, Sam3Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..config.settings import settings

logger = logging.getLogger(__name__)


class SAM3Local:
    """SAM3 modeli - Transformers ile yerel kullanım"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """
        Args:
            model_name: Model adı (default: facebook/sam3)
            device: cuda veya cpu
            api_token: Hugging Face API token (gated model için gerekli)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers kütüphanesi gerekli! pip install transformers")
        
        self.model_name = model_name or settings.huggingface_model_name
        self.api_token = api_token or settings.huggingface_api_token
        
        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"SAM3 model yükleniyor: {self.model_name} (device: {self.device})")
        
        try:
            # Token ile login (gated model için)
            if self.api_token:
                from huggingface_hub import login
                login(token=self.api_token)
            
            # Model ve processor yükle
            self.processor = Sam3Processor.from_pretrained(
                self.model_name,
                token=self.api_token
            )
            self.model = Sam3Model.from_pretrained(
                self.model_name,
                token=self.api_token
            ).to(self.device)
            
            self.model.eval()  # Evaluation mode
            
            logger.info("SAM3 model başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"SAM3 model yüklenemedi: {e}")
            raise
    
    def segment_with_text(
        self,
        image: np.ndarray,
        text: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Text prompt ile segmentasyon
        
        Args:
            image: OpenCV BGR image array
            text: Text prompt (e.g., "basketball player")
            threshold: Confidence threshold
            mask_threshold: Mask threshold
            
        Returns:
            Segmentasyon sonuçları
        """
        # BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Process inputs
        inputs = self.processor(
            images=pil_image,
            text=text,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        try:
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
        except Exception as e:
            logger.warning(f"Post-processing hatası: {e}, raw outputs kullanılıyor")
            # Raw outputs'u kullan
            results = {
                'masks': outputs.pred_masks if hasattr(outputs, 'pred_masks') else None,
                'boxes': outputs.pred_boxes if hasattr(outputs, 'pred_boxes') else None,
                'scores': outputs.pred_scores if hasattr(outputs, 'pred_scores') else None
            }
        
        # Detection formatına çevir
        detections = []
        
        # Farklı formatları kontrol et
        if isinstance(results, dict):
            if 'masks' in results and 'boxes' in results and 'scores' in results:
                masks = results['masks']
                boxes = results['boxes']
                scores = results['scores']
                
                # Tensor ise numpy'a çevir
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu()
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu()
                if hasattr(masks, 'cpu'):
                    masks = masks.cpu()
                
                num_detections = len(boxes) if hasattr(boxes, '__len__') else 0
                
                for i in range(num_detections):
                    # Box format: tensor veya numpy array [x1, y1, x2, y2]
                    if hasattr(boxes[i], 'numpy'):
                        box = boxes[i].numpy()
                    elif hasattr(boxes[i], 'tolist'):
                        box = boxes[i].tolist()
                    else:
                        box = boxes[i]
                    
                    if hasattr(scores[i], 'numpy'):
                        score = float(scores[i].numpy())
                    elif hasattr(scores[i], 'item'):
                        score = float(scores[i].item())
                    else:
                        score = float(scores[i])
                    
                    if hasattr(masks[i], 'numpy'):
                        mask = masks[i].numpy()
                    elif hasattr(masks[i], 'tolist'):
                        mask = masks[i]
                    else:
                        mask = masks[i]
                    
                    detections.append({
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'confidence': score,
                        'class_id': 0,
                        'class_name': 'person',
                        'mask': mask
                    })
        
        logger.debug(f"SAM3 detection: {len(detections)} tespit edildi")
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'raw_results': results
        }
    
    def segment_with_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Point prompt ile segmentasyon
        
        Args:
            image: OpenCV BGR image array
            points: (x, y) koordinatları listesi
            labels: Her point için label (1: foreground, 0: background)
            
        Returns:
            Segmentasyon sonuçları
        """
        if labels is None:
            labels = [1] * len(points)
        
        # BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        # Point format: [[[x, y], ...]]
        input_points = [[points]]
        input_labels = [[labels]]
        
        # Process
        inputs = self.processor(
            images=pil_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        # Post-process
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]
        
        return {'masks': masks}
    
    def segment_with_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        Box prompt ile segmentasyon
        
        Args:
            image: OpenCV BGR image array
            box: (x1, y1, x2, y2) koordinatları
            
        Returns:
            Segmentasyon sonuçları
        """
        # BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        # Box format: [[[x1, y1, x2, y2]]]
        input_boxes = [[box]]
        input_boxes_labels = [[1]]  # Positive box
        
        inputs = self.processor(
            images=pil_image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        return results
