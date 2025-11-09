from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration with environment variable support"""

    # Gemini Configuration (Primary LLM)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash-exp"  # Fast and powerful
    
    # ElevenLabs Configuration
    eleven_api_key: Optional[str] = None
    eleven_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    eleven_latency_ms: int = 300
    
    # Detection Weights
    text_weight: float = 0.6
    audio_weight: float = 0.4
    
    # Features
    enable_audio: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Session Configuration
    session_timeout_minutes: int = 30
    intervention_cooldown_seconds: int = 30
    
    # Safety Configuration
    max_response_words: int = 18
    escalation_timeout_seconds: int = 120
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

