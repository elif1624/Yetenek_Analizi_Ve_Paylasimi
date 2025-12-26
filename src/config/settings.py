"""Uygulama ayarları ve konfigürasyon"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# .env dosyasını yükle
load_dotenv()


class Settings(BaseSettings):
    """Uygulama ayarları"""
    
    # Hugging Face API
    huggingface_api_token: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    huggingface_model_name: str = os.getenv("HUGGINGFACE_MODEL_NAME", "facebook/sam3")
    
    # Video Processing
    max_video_duration: int = int(os.getenv("MAX_VIDEO_DURATION", "300"))
    frame_extraction_fps: float = float(os.getenv("FRAME_EXTRACTION_FPS", "5.0"))
    video_output_quality: str = os.getenv("VIDEO_OUTPUT_QUALITY", "high")
    
    # Analysis
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    min_event_duration: int = int(os.getenv("MIN_EVENT_DURATION", "2"))
    max_event_duration: int = int(os.getenv("MAX_EVENT_DURATION", "30"))
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    input_dir: Path = data_dir / "input"
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", str(data_dir / "output")))
    results_dir: Path = Path(os.getenv("RESULTS_DIR", str(data_dir / "results")))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Dizinleri oluştur
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

