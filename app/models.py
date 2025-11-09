from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DistressState(str, Enum):
    """States representing the user's distress level"""
    CALM = "calm"
    RISING = "rising"
    PANIC = "panic"
    OVERWHELMED = "overwhelmed"
    RECOVERY = "recovery"
    CRISIS_RISK = "crisis_risk"


class InterventionType(str, Enum):
    """Types of therapeutic interventions"""
    PACED_BREATHING = "paced_breathing"
    FOUR_SEVEN_EIGHT = "four_seven_eight"
    BOX_BREATHING = "box_breathing"
    GROUNDING_54321 = "grounding_54321"
    REINFORCEMENT = "reinforcement"
    ESCALATION = "escalation"
    CHECK_IN = "check_in"


class CheckInResponse(str, Enum):
    """User responses to check-in questions"""
    BETTER = "better"
    SAME = "same"
    WORSE = "worse"


class InferRequest(BaseModel):
    """Request payload for distress inference"""
    message: str = Field(..., description="User's text message")
    chat_id: str = Field(..., description="Unique conversation identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    frame_url: Optional[str] = Field(None, description="Optional image/video frame URL")


class CheckInRequest(BaseModel):
    """Request payload for follow-up check-ins"""
    chat_id: str = Field(..., description="Unique conversation identifier")
    response: CheckInResponse = Field(..., description="User's check-in response")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ResponseMeta(BaseModel):
    """Metadata about the inference"""
    state: DistressState
    confidence: float = Field(..., ge=0.0, le=1.0)
    intervention_type: Optional[InterventionType] = None
    session_duration_seconds: int = 0


class InferResponse(BaseModel):
    """Response payload from distress inference"""
    reply_text: str = Field(..., description="Text to send/speak to user")
    expect_followup: bool = Field(default=False, description="Whether to expect a follow-up")
    followup_after_sec: int = Field(default=0, description="Seconds until follow-up")
    buttons: Optional[List[str]] = Field(None, description="Quick reply buttons")
    audio_url: Optional[str] = Field(None, description="URL to generated audio (if available)")
    meta: ResponseMeta = Field(..., description="Metadata about the inference")


class SessionState(BaseModel):
    """Internal session state tracking"""
    chat_id: str
    current_state: DistressState = DistressState.CALM
    distress_prob: float = 0.0
    last_intervention: Optional[InterventionType] = None
    last_intervention_time: Optional[datetime] = None
    intervention_count: int = 0
    session_start: datetime = Field(default_factory=datetime.utcnow)
    last_update: datetime = Field(default_factory=datetime.utcnow)
    escalation_offered: bool = False
    stopped: bool = False


class ConversationAssistRequest(BaseModel):
    """Request for conversation assistance with facial analysis"""
    chat_id: str = Field(..., description="Unique conversation identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    other_person_said: str = Field(..., description="What the other person just said")
    frame_url: Optional[str] = Field(None, description="Image URL of the other person's face")
    frame_base64: Optional[str] = Field(None, description="Base64 encoded image of the other person's face")
    conversation_context: Optional[str] = Field(None, description="Brief context about the conversation")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class EmotionAnalysis(BaseModel):
    """Facial emotion analysis results"""
    primary_emotion: str = Field(..., description="Primary detected emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in emotion detection")
    secondary_emotions: Optional[List[str]] = Field(None, description="Other detected emotions")
    facial_cues: Optional[str] = Field(None, description="Notable facial cues (e.g., smiling, frowning)")


class ConversationAssistResponse(BaseModel):
    """Response with conversation suggestions"""
    suggested_response: str = Field(..., description="Suggested thing to say back")
    alternative_responses: Optional[List[str]] = Field(None, description="Alternative response options")
    emotion_analysis: Optional[EmotionAnalysis] = Field(None, description="Facial emotion analysis")
    tone_guidance: str = Field(..., description="Recommended tone (e.g., 'friendly', 'empathetic', 'casual')")
    social_cues: Optional[str] = Field(None, description="Interpretation of social cues")
    audio_url: Optional[str] = Field(None, description="Audio version of suggested response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in suggestion")

