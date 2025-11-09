"""
ARM Edge Server - Privacy-Preserving Local AI Processing

This service runs on ARM devices (Raspberry Pi, NVIDIA Jetson, etc.) and handles:
1. Local facial emotion detection using TensorFlow Lite with ARM NN acceleration
2. ARM NEON-optimized audio distress detection
3. Privacy-preserving preprocessing before cloud upload
4. Offline distress detection capability

Architecture:
Meta Glasses → ARM Edge Device (this service) → Cloud Backend (optional)
                     ↓
                Local AI Models:
                - Facial Emotion (TFLite + ARM NN)
                - Audio Distress (NEON SIMD)
                - Speech Features (optimized)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import httpx
import os
import time
from datetime import datetime
import asyncio
import logging

# ARM-specific imports
from arm_edge.facial_detector_arm import ARMFacialEmotionDetector
from arm_edge.audio_processor_arm import ARMAudioDistressProcessor
from arm_edge.privacy_filter import PrivacyPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EmpathLens ARM Edge Service",
    description="Privacy-preserving local AI processing on ARM devices",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ARM-optimized components
facial_detector = ARMFacialEmotionDetector()
audio_processor = ARMAudioDistressProcessor()
privacy_filter = PrivacyPreprocessor()

# Cloud backend configuration
CLOUD_BACKEND_URL = os.getenv("CLOUD_BACKEND_URL", "http://localhost:8000")
PRIVACY_MODE = os.getenv("PRIVACY_MODE", "strict")  # strict, balanced, minimal
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() == "true"


class EdgeProcessingRequest(BaseModel):
    """Request for ARM edge processing"""
    chat_id: str
    user_message: Optional[str] = None
    conversation_mode: bool = False  # True for conversation assist, False for distress
    conversation_context: Optional[str] = None


class EdgeProcessingResponse(BaseModel):
    """Response from ARM edge processing"""
    facial_emotion: Optional[Dict] = None
    audio_features: Optional[Dict] = None
    local_distress_detected: bool = False
    local_distress_level: float = 0.0
    processing_time_ms: float
    privacy_filters_applied: List[str]
    cloud_forwarding_needed: bool
    offline_response: Optional[str] = None


class PerformanceMetrics(BaseModel):
    """ARM performance metrics"""
    facial_inference_ms: float
    audio_processing_ms: float
    total_processing_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    arm_neon_enabled: bool
    arm_nn_enabled: bool


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "EmpathLens ARM Edge Service",
        "status": "running",
        "device_type": facial_detector.device_info["device_type"],
        "processor": facial_detector.device_info["processor"],
        "arm_optimizations": {
            "neon_simd": audio_processor.neon_enabled,
            "arm_nn_delegate": facial_detector.arm_nn_enabled,
            "quantized_models": True
        },
        "privacy_mode": PRIVACY_MODE,
        "offline_mode": OFFLINE_MODE,
        "cloud_backend": CLOUD_BACKEND_URL
    }


@app.post("/edge/process", response_model=EdgeProcessingResponse)
async def process_on_edge(
    chat_id: str = Form(...),
    user_message: Optional[str] = Form(None),
    conversation_mode: bool = Form(False),
    conversation_context: Optional[str] = Form(None),
    face_image: Optional[UploadFile] = File(None),
    audio_data: Optional[UploadFile] = File(None)
):
    """
    Main ARM edge processing endpoint.

    Performs local AI inference with ARM optimizations:
    1. Facial emotion detection (TFLite + ARM NN)
    2. Audio distress detection (NEON SIMD)
    3. Privacy filtering
    4. Decides whether cloud forwarding is needed

    Returns local results and handles cloud communication if needed.
    """
    start_time = time.time()

    try:
        facial_emotion = None
        audio_features = None
        privacy_filters = []

        # Process facial image locally with ARM NN acceleration
        if face_image:
            logger.info(f"Processing facial image on ARM device...")
            image_bytes = await face_image.read()

            facial_start = time.time()
            facial_emotion = await facial_detector.detect_emotion(image_bytes)
            facial_time = (time.time() - facial_start) * 1000

            logger.info(f"Facial detection completed in {facial_time:.2f}ms using ARM NN")

            # Apply privacy filtering
            if PRIVACY_MODE == "strict":
                # Don't send image to cloud, only emotion metadata
                privacy_filters.append("facial_image_local_only")
            elif PRIVACY_MODE == "balanced":
                # Only send if high uncertainty
                if facial_emotion["confidence"] < 0.7:
                    privacy_filters.append("facial_image_on_uncertainty")
                else:
                    privacy_filters.append("facial_image_local_only")

        # Process audio locally with ARM NEON SIMD
        if audio_data:
            logger.info(f"Processing audio with ARM NEON SIMD...")
            audio_bytes = await audio_data.read()

            audio_start = time.time()
            audio_features = await audio_processor.extract_features(audio_bytes)
            audio_time = (time.time() - audio_start) * 1000

            logger.info(f"Audio processing completed in {audio_time:.2f}ms using NEON")

            # Apply privacy filtering
            if PRIVACY_MODE == "strict":
                # Don't send raw audio to cloud
                privacy_filters.append("audio_features_only")

        # Local distress detection using ARM-processed features
        local_distress = False
        local_distress_level = 0.0

        if audio_features:
            local_distress_level = audio_features.get("distress_probability", 0.0)
            local_distress = local_distress_level > 0.6

        # Determine if cloud forwarding is needed
        cloud_forwarding_needed = True
        offline_response = None

        if OFFLINE_MODE or (PRIVACY_MODE == "strict" and local_distress):
            # Provide offline response for immediate support
            cloud_forwarding_needed = False

            if local_distress:
                offline_response = "I'm here with you. Take a slow breath in for four counts."
                logger.info("Offline distress response generated")

        # If cloud forwarding is needed and allowed, forward to cloud backend
        if cloud_forwarding_needed and not OFFLINE_MODE:
            try:
                await forward_to_cloud(
                    chat_id=chat_id,
                    user_message=user_message,
                    facial_emotion=facial_emotion,
                    audio_features=audio_features,
                    conversation_mode=conversation_mode,
                    conversation_context=conversation_context,
                    privacy_filters=privacy_filters
                )
            except Exception as e:
                logger.error(f"Cloud forwarding failed: {e}")
                # Fallback to offline mode
                if local_distress:
                    offline_response = "I'm here with you. Take a slow breath in for four counts."

        processing_time = (time.time() - start_time) * 1000

        return EdgeProcessingResponse(
            facial_emotion=facial_emotion,
            audio_features=audio_features,
            local_distress_detected=local_distress,
            local_distress_level=local_distress_level,
            processing_time_ms=processing_time,
            privacy_filters_applied=privacy_filters,
            cloud_forwarding_needed=cloud_forwarding_needed,
            offline_response=offline_response
        )

    except Exception as e:
        logger.error(f"Edge processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Edge processing failed: {str(e)}")


async def forward_to_cloud(
    chat_id: str,
    user_message: Optional[str],
    facial_emotion: Optional[Dict],
    audio_features: Optional[Dict],
    conversation_mode: bool,
    conversation_context: Optional[str],
    privacy_filters: List[str]
):
    """
    Forward preprocessed data to cloud backend.
    Only sends metadata and features, not raw images/audio (privacy-preserving).
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        if conversation_mode:
            # Forward to conversation assist endpoint
            payload = {
                "chat_id": chat_id,
                "other_person_said": user_message,
                "conversation_context": conversation_context,
                # Include emotion metadata from local ARM processing
                "edge_facial_emotion": facial_emotion,
                "privacy_filters": privacy_filters
            }
            endpoint = f"{CLOUD_BACKEND_URL}/conversation/assist"
        else:
            # Forward to distress detection endpoint
            payload = {
                "chat_id": chat_id,
                "message": user_message,
                # Include audio features from ARM NEON processing
                "edge_audio_features": audio_features,
                "edge_distress_level": audio_features.get("distress_probability", 0.0) if audio_features else 0.0,
                "privacy_filters": privacy_filters
            }
            endpoint = f"{CLOUD_BACKEND_URL}/distress/infer"

        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()


@app.get("/arm/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """
    Get ARM-specific performance metrics.
    Shows benefits of ARM optimization.
    """
    # Get metrics from ARM components
    facial_metrics = facial_detector.get_performance_metrics()
    audio_metrics = audio_processor.get_performance_metrics()

    import psutil

    return PerformanceMetrics(
        facial_inference_ms=facial_metrics["avg_inference_ms"],
        audio_processing_ms=audio_metrics["avg_processing_ms"],
        total_processing_ms=facial_metrics["avg_inference_ms"] + audio_metrics["avg_processing_ms"],
        cpu_usage_percent=psutil.cpu_percent(),
        memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
        arm_neon_enabled=audio_processor.neon_enabled,
        arm_nn_enabled=facial_detector.arm_nn_enabled
    )


@app.get("/arm/benchmark")
async def run_benchmark():
    """
    Run performance benchmarks comparing ARM-optimized vs standard processing.
    Demonstrates the value of ARM acceleration.
    """
    results = {
        "facial_detection": await facial_detector.run_benchmark(),
        "audio_processing": await audio_processor.run_benchmark()
    }

    return {
        "benchmark_results": results,
        "summary": {
            "facial_speedup": f"{results['facial_detection']['speedup_factor']:.2f}x faster with ARM NN",
            "audio_speedup": f"{results['audio_processing']['speedup_factor']:.2f}x faster with NEON SIMD",
            "total_speedup": f"{((results['facial_detection']['speedup_factor'] + results['audio_processing']['speedup_factor']) / 2):.2f}x average speedup"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "facial_detector": "operational" if facial_detector.is_ready() else "not_ready",
            "audio_processor": "operational" if audio_processor.is_ready() else "not_ready",
            "privacy_filter": "operational"
        },
        "arm_optimizations": {
            "neon_simd": audio_processor.neon_enabled,
            "arm_nn_delegate": facial_detector.arm_nn_enabled,
            "quantized_int8": True
        },
        "device_info": facial_detector.device_info,
        "privacy_mode": PRIVACY_MODE,
        "offline_capable": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
