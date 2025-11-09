"""
Privacy-Preserving Preprocessor for ARM Edge Device

Ensures sensitive data (images, audio) can be processed locally on ARM
without sending raw data to cloud, preserving user privacy.

Privacy modes:
- STRICT: All processing local, only metadata sent to cloud
- BALANCED: Send raw data only when local confidence is low
- MINIMAL: Send all data to cloud (lowest privacy)
"""

from typing import Dict, Tuple, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class PrivacyPreprocessor:
    """
    Privacy-preserving filter for ARM edge processing.

    Determines what data can be sent to cloud based on privacy settings.
    """

    def __init__(self, privacy_mode: str = "balanced"):
        """
        Initialize privacy filter.

        Args:
            privacy_mode: "strict", "balanced", or "minimal"
        """
        self.privacy_mode = privacy_mode.lower()
        logger.info(f"Privacy mode: {self.privacy_mode}")

    def filter_facial_data(
        self,
        image_data: bytes,
        emotion_result: Dict,
        privacy_mode: Optional[str] = None
    ) -> Tuple[Optional[bytes], Dict, str]:
        """
        Filter facial image data based on privacy settings.

        Args:
            image_data: Raw image bytes
            emotion_result: Local emotion detection result
            privacy_mode: Override default privacy mode

        Returns:
            Tuple of (filtered_image_bytes, metadata, reason)
        """
        mode = privacy_mode or self.privacy_mode

        if mode == "strict":
            # Never send image to cloud
            metadata = {
                "emotion": emotion_result.get("primary_emotion"),
                "confidence": emotion_result.get("confidence"),
                "emotions": emotion_result.get("emotions"),
                "processed_locally": True,
                "image_hash": self._hash_data(image_data)
            }
            return None, metadata, "strict_privacy_mode"

        elif mode == "balanced":
            # Send image only if local confidence is low
            confidence = emotion_result.get("confidence", 0.0)

            if confidence >= 0.7:
                # High confidence - don't need cloud verification
                metadata = {
                    "emotion": emotion_result.get("primary_emotion"),
                    "confidence": confidence,
                    "emotions": emotion_result.get("emotions"),
                    "processed_locally": True,
                    "image_hash": self._hash_data(image_data)
                }
                return None, metadata, "high_local_confidence"
            else:
                # Low confidence - send for cloud processing
                metadata = {
                    "local_emotion": emotion_result.get("primary_emotion"),
                    "local_confidence": confidence,
                    "needs_cloud_verification": True
                }
                return image_data, metadata, "low_local_confidence"

        else:  # minimal privacy
            # Always send to cloud
            metadata = {
                "local_emotion": emotion_result.get("primary_emotion"),
                "local_confidence": emotion_result.get("confidence")
            }
            return image_data, metadata, "minimal_privacy_mode"

    def filter_audio_data(
        self,
        audio_data: bytes,
        audio_features: Dict,
        privacy_mode: Optional[str] = None
    ) -> Tuple[Optional[bytes], Dict, str]:
        """
        Filter audio data based on privacy settings.

        Args:
            audio_data: Raw audio bytes
            audio_features: Extracted audio features
            privacy_mode: Override default privacy mode

        Returns:
            Tuple of (filtered_audio_bytes, metadata, reason)
        """
        mode = privacy_mode or self.privacy_mode

        if mode == "strict":
            # Never send raw audio - only features
            metadata = {
                "features": audio_features.get("features"),
                "distress_probability": audio_features.get("distress_probability"),
                "processed_locally": True,
                "audio_hash": self._hash_data(audio_data)
            }
            return None, metadata, "strict_privacy_mode"

        elif mode == "balanced":
            # Send audio only if local distress detection is uncertain
            distress_prob = audio_features.get("distress_probability", 0.0)

            # If clearly distressed or clearly calm, don't need cloud
            if distress_prob < 0.3 or distress_prob > 0.7:
                metadata = {
                    "features": audio_features.get("features"),
                    "distress_probability": distress_prob,
                    "processed_locally": True,
                    "audio_hash": self._hash_data(audio_data)
                }
                return None, metadata, "clear_local_classification"
            else:
                # Uncertain - send for cloud processing
                metadata = {
                    "local_features": audio_features.get("features"),
                    "local_distress_prob": distress_prob,
                    "needs_cloud_verification": True
                }
                return audio_data, metadata, "uncertain_classification"

        else:  # minimal privacy
            # Always send to cloud
            metadata = {
                "local_features": audio_features.get("features"),
                "local_distress_prob": audio_features.get("distress_probability")
            }
            return audio_data, metadata, "minimal_privacy_mode"

    @staticmethod
    def _hash_data(data: bytes) -> str:
        """Create hash of data for verification without sending raw data"""
        return hashlib.sha256(data).hexdigest()[:16]

    def get_privacy_summary(self) -> Dict:
        """Get summary of privacy settings and guarantees"""
        summaries = {
            "strict": {
                "facial_images": "Never sent to cloud - processed locally only",
                "audio": "Never sent to cloud - only features transmitted",
                "guarantees": "Maximum privacy - all sensitive data stays on ARM device",
                "tradeoff": "May have lower accuracy when local models are uncertain"
            },
            "balanced": {
                "facial_images": "Sent to cloud only when local confidence < 70%",
                "audio": "Sent to cloud only when distress unclear (30-70% range)",
                "guarantees": "Privacy-first with cloud backup for edge cases",
                "tradeoff": "Best balance of privacy and accuracy"
            },
            "minimal": {
                "facial_images": "Always sent to cloud for processing",
                "audio": "Always sent to cloud for processing",
                "guarantees": "Standard cloud processing - no local privacy",
                "tradeoff": "Maximum accuracy, minimal privacy"
            }
        }

        return {
            "current_mode": self.privacy_mode,
            "settings": summaries.get(self.privacy_mode, summaries["balanced"])
        }
