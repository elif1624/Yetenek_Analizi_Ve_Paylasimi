"""SAM3 Hugging Face API client"""

import os
import requests
import base64
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import io

from ..config.settings import settings

logger = logging.getLogger(__name__)


class SAM3Client:
    """SAM3 modeli için Hugging Face API client"""
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Args:
            api_token: Hugging Face API token
            model_name: Model adı
        """
        self.api_token = api_token or settings.huggingface_api_token
        self.model_name = model_name or settings.huggingface_model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        if not self.api_token:
            raise ValueError("Hugging Face API token gerekli! .env dosyasında HUGGINGFACE_API_TOKEN ayarlayın.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"SAM3 client başlatıldı: {self.model_name}")
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        OpenCV image'ı base64 string'e çevir
        
        Args:
            image: OpenCV BGR image array
            
        Returns:
            Base64 encoded string
        """
        # BGR'dan RGB'ye çevir
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # PIL Image'a çevir
        pil_image = Image.fromarray(image_rgb)
        
        # Bytes buffer'a kaydet
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # Base64 encode
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return base64_string
    
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[List[Dict[str, Any]]] = None,
        return_mask: bool = True
    ) -> Dict[str, Any]:
        """
        Image üzerinde segmentasyon yap
        
        Args:
            image: OpenCV image array (BGR format)
            prompts: Segmentasyon prompt'ları (point, box, text vb.)
            return_mask: Mask döndürülsün mü
            
        Returns:
            Segmentasyon sonuçları
        """
        # Image'ı encode et
        image_base64 = self._encode_image(image)
        
        # Request payload
        payload = {
            "inputs": {
                "image": image_base64,
            }
        }
        
        if prompts:
            payload["inputs"]["prompts"] = prompts
        
        try:
            logger.info("SAM3 API'ye istek gönderiliyor...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info("SAM3 segmentasyon tamamlandı")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SAM3 API hatası: {e}")
            raise
    
    def segment_with_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Point prompt'ları ile segmentasyon
        
        Args:
            image: OpenCV image array
            points: (x, y) koordinatları listesi
            labels: Her point için label (1: foreground, 0: background)
            
        Returns:
            Segmentasyon sonuçları
        """
        if labels is None:
            labels = [1] * len(points)
        
        prompts = [
            {
                "type": "point",
                "coordinates": point,
                "label": label
            }
            for point, label in zip(points, labels)
        ]
        
        return self.segment_image(image, prompts=prompts)
    
    def segment_with_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    ) -> Dict[str, Any]:
        """
        Bounding box prompt'u ile segmentasyon
        
        Args:
            image: OpenCV image array
            box: Bounding box koordinatları (x1, y1, x2, y2)
            
        Returns:
            Segmentasyon sonuçları
        """
        prompts = [
            {
                "type": "box",
                "coordinates": box
            }
        ]
        
        return self.segment_image(image, prompts=prompts)
    
    def segment_with_text(
        self,
        image: np.ndarray,
        text: str
    ) -> Dict[str, Any]:
        """
        Text prompt ile segmentasyon (SAM3 text prompt desteği)
        
        Args:
            image: OpenCV image array
            text: Text prompt (e.g., "basketball player", "person", "basketball")
            
        Returns:
            Segmentasyon sonuçları
        """
        prompts = [
            {
                "type": "text",
                "text": text
            }
        ]
        
        return self.segment_image(image, prompts=prompts)
    
    def segment_everything(
        self,
        image: np.ndarray,
        min_area: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Segment everything (tüm nesneleri segment et)
        
        Args:
            image: OpenCV image array
            min_area: Minimum segment area (filter küçük segmentler)
            
        Returns:
            Segmentasyon sonuçları
        """
        # SAM3'ün segment everything özelliği
        # Prompt olmadan çağrılabilir veya özel bir prompt kullanılabilir
        payload = {
            "inputs": {
                "image": self._encode_image(image),
                "mode": "everything"
            }
        }
        
        if min_area:
            payload["inputs"]["min_area"] = min_area
        
        try:
            logger.info("SAM3 segment everything API'ye istek gönderiliyor...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info("SAM3 segment everything tamamlandı")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SAM3 API hatası: {e}")
            raise
    
    def check_api_status(self) -> bool:
        """API erişim durumunu kontrol et"""
        try:
            response = requests.get(
                f"https://api-inference.huggingface.co/models/{self.model_name}",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API status check hatası: {e}")
            return False

