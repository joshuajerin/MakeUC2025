import os
import tempfile
from typing import Optional
from app.config import settings

try:
    from elevenlabs import generate, set_api_key, Voice, VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        ELEVENLABS_AVAILABLE = True
    except ImportError:
        ELEVENLABS_AVAILABLE = False


class TTSClient:
    """
    Client for ElevenLabs text-to-speech.
    Converts coaching text into natural spoken audio.
    """
    
    def __init__(self):
        self.enabled = bool(settings.eleven_api_key) and ELEVENLABS_AVAILABLE
        self.voice_id = settings.eleven_voice_id
        self.client = None
        
        if self.enabled:
            try:
                from elevenlabs.client import ElevenLabs
                self.client = ElevenLabs(api_key=settings.eleven_api_key)
            except ImportError:
                try:
                    set_api_key(settings.eleven_api_key)
                except:
                    self.enabled = False
    
    async def synthesize(self, text: str) -> Optional[str]:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to generated audio file, or None if TTS unavailable
        """
        if not self.enabled:
            return None
        
        try:
            if self.client:
                audio_generator = self.client.generate(
                    text=text,
                    voice=self.voice_id,
                    model="eleven_monolingual_v1",
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True
                    )
                )
                
                audio_bytes = b"".join(audio_generator)
            else:
                from elevenlabs import generate, Voice
                audio_bytes = generate(
                    text=text,
                    voice=Voice(
                        voice_id=self.voice_id,
                        settings=VoiceSettings(
                            stability=0.5,
                            similarity_boost=0.75,
                            style=0.0,
                            use_speaker_boost=True
                        )
                    ),
                    model="eleven_monolingual_v1"
                )
            
            temp_dir = os.path.join(tempfile.gettempdir(), "empathlens_audio")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = os.path.join(temp_dir, f"response_{hash(text)}.mp3")
            
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
            
            return temp_file
            
        except Exception:
            return None
    
    def cleanup_old_files(self, max_age_hours: int = 1):
        """Clean up old audio files"""
        try:
            temp_dir = os.path.join(tempfile.gettempdir(), "empathlens_audio")
            if not os.path.exists(temp_dir):
                return
            
            import time
            now = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(temp_dir):
                filepath = os.path.join(temp_dir, filename)
                if os.path.isfile(filepath):
                    age = now - os.path.getmtime(filepath)
                    if age > max_age_seconds:
                        os.remove(filepath)
        except Exception:
            pass

