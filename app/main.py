from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.models import (
    InferRequest,
    InferResponse,
    CheckInRequest,
    CheckInResponse,
    ResponseMeta,
    DistressState,
    ConversationAssistRequest,
    ConversationAssistResponse,
)
from app.config import settings
from app.core.detector import DistressDetector
from app.core.state_machine import StateMachine
from app.core.interventions import InterventionManager
from app.core.session_manager import SessionManager
from app.core.safety import SafetyFilter, sanitize_user_input
from app.services.gemini_client import GeminiClient
from app.services.tts_client import TTSClient
from app.services.conversation_assistant import FacialAnalyzer, ConversationCoach


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown"""
    yield
    tts_client.cleanup_old_files()


app = FastAPI(
    title="EmpathLens - Distress Helper",
    description="Local provider for Meta Glasses API distress assistance",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = DistressDetector(
    text_weight=settings.text_weight,
    audio_weight=settings.audio_weight
)
state_machine = StateMachine()
intervention_manager = InterventionManager()

# Initialize Gemini client
llm_client = GeminiClient()

# Initialize conversation assistant components
facial_analyzer = FacialAnalyzer()
conversation_coach = ConversationCoach()

tts_client = TTSClient()
session_manager = SessionManager()
safety_filter = SafetyFilter(max_words=settings.max_response_words)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "EmpathLens - Social Companion & Distress Helper",
        "status": "running",
        "llm_provider": "gemini",
        "llm_model": settings.gemini_model,
        "tts_enabled": tts_client.enabled,
        "audio_detection_enabled": settings.enable_audio,
    }


@app.post("/distress/infer", response_model=InferResponse)
async def infer_distress(request: InferRequest):
    """
    Main endpoint for distress inference and intervention.
    
    Processes user message, detects distress, determines state,
    generates coaching response, and optionally synthesizes speech.
    """
    try:
        chat_id = request.chat_id
        user_message = sanitize_user_input(request.message)
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        session = session_manager.get_or_create_session(chat_id)
        
        distress_prob, is_crisis, is_stop, detection_details = detector.detect(
            text=user_message,
            audio_features=None
        )
        
        if is_stop:
            session_manager.update_session(chat_id, stopped=True)
            return InferResponse(
                reply_text="Understood. I'm here if you need me.",
                expect_followup=False,
                followup_after_sec=0,
                buttons=None,
                audio_url=None,
                meta=ResponseMeta(
                    state=DistressState.CALM,
                    confidence=1.0,
                    intervention_type=None,
                    session_duration_seconds=session_manager.get_session_duration(chat_id)
                )
            )
        
        new_state, state_changed = state_machine.determine_state(
            session=session,
            distress_prob=distress_prob,
            is_crisis=is_crisis
        )
        
        session_manager.update_session(
            chat_id=chat_id,
            state=new_state,
            distress_prob=distress_prob
        )
        
        should_escalate = state_machine.should_escalate(session)
        
        if should_escalate and not session.escalation_offered:
            escalation_text = intervention_manager.get_escalation_prompt()
            session_manager.update_session(chat_id, escalation_offered=True)
            
            audio_url = await tts_client.synthesize(escalation_text)
            
            return InferResponse(
                reply_text=escalation_text,
                expect_followup=True,
                followup_after_sec=0,
                buttons=["contact_support", "continue_alone"],
                audio_url=audio_url,
                meta=ResponseMeta(
                    state=new_state,
                    confidence=distress_prob,
                    intervention_type=None,
                    session_duration_seconds=session_manager.get_session_duration(chat_id)
                )
            )
        
        intervention_type, duration, base_prompt = intervention_manager.get_intervention(
            state=new_state,
            chat_id=chat_id
        )
        
        coaching_text = await llm_client.generate_response(
            user_message=user_message,
            current_state=new_state,
            distress_prob=distress_prob,
            last_intervention=session.last_intervention,
            context=detection_details
        )
        
        filtered_text, is_safe = safety_filter.filter(coaching_text)
        
        if not is_safe:
            filtered_text = base_prompt
        
        session_manager.update_session(
            chat_id=chat_id,
            intervention=intervention_type
        )
        
        audio_url = await tts_client.synthesize(filtered_text)
        
        expect_followup = new_state not in [DistressState.CALM, DistressState.RECOVERY]
        followup_sec = duration if expect_followup else 0
        
        should_check_in = (
            new_state in [DistressState.PANIC, DistressState.OVERWHELMED, DistressState.RISING]
            and duration > 0
        )
        buttons = ["better", "same", "worse"] if should_check_in else None
        
        return InferResponse(
            reply_text=filtered_text,
            expect_followup=expect_followup,
            followup_after_sec=followup_sec,
            buttons=buttons,
            audio_url=audio_url,
            meta=ResponseMeta(
                state=new_state,
                confidence=distress_prob,
                intervention_type=intervention_type,
                session_duration_seconds=session_manager.get_session_duration(chat_id)
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/distress/checkin", response_model=InferResponse)
async def checkin_followup(request: CheckInRequest):
    """
    Handle follow-up check-ins after interventions.
    
    Processes user's better/same/worse response and adapts next steps.
    """
    try:
        chat_id = request.chat_id
        response_type = request.response
        
        session = session_manager.get_or_create_session(chat_id)
        
        if response_type == CheckInResponse.BETTER:
            new_state = DistressState.RECOVERY
            reply_text = "Your body is settling. Take two slow breaths."
            expect_followup = False
            followup_sec = 0
            buttons = None
            
        elif response_type == CheckInResponse.SAME:
            current = session.current_state
            
            if session.last_intervention in [
                None,
                intervention_manager.INTERVENTIONS[DistressState.PANIC][0]
            ]:
                new_state = DistressState.OVERWHELMED
                reply_text = "Let's try grounding. Name five things you can see."
            else:
                new_state = DistressState.PANIC
                reply_text = "In for four, hold seven, out for eight."
            
            expect_followup = True
            followup_sec = 45
            buttons = ["better", "same", "worse"]
            
        else:
            duration = session_manager.get_session_duration(chat_id)
            
            if duration >= settings.escalation_timeout_seconds:
                reply_text = intervention_manager.get_escalation_prompt()
                session_manager.update_session(chat_id, escalation_offered=True)
                expect_followup = True
                followup_sec = 0
                buttons = ["contact_support", "continue_alone"]
                new_state = session.current_state
            else:
                new_state = session.current_state
                intervention_type, duration, _ = intervention_manager.get_intervention(
                    state=new_state,
                    chat_id=chat_id
                )
                
                reply_text = await llm_client.generate_response(
                    user_message="feeling worse",
                    current_state=new_state,
                    distress_prob=session.distress_prob,
                    last_intervention=session.last_intervention
                )
                
                reply_text, _ = safety_filter.filter(reply_text)
                expect_followup = True
                followup_sec = duration
                buttons = ["better", "same", "worse"]
        
        session_manager.update_session(
            chat_id=chat_id,
            state=new_state
        )
        
        audio_url = await tts_client.synthesize(reply_text)
        
        return InferResponse(
            reply_text=reply_text,
            expect_followup=expect_followup,
            followup_after_sec=followup_sec,
            buttons=buttons,
            audio_url=audio_url,
            meta=ResponseMeta(
                state=new_state,
                confidence=session.distress_prob,
                intervention_type=session.last_intervention,
                session_duration_seconds=session_manager.get_session_duration(chat_id)
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/distress/stop")
async def stop_session(chat_id: str):
    """Stop intervention for a specific conversation"""
    session_manager.update_session(chat_id, stopped=True)
    return {"status": "stopped", "chat_id": chat_id}


@app.post("/conversation/assist", response_model=ConversationAssistResponse)
async def assist_conversation(request: ConversationAssistRequest):
    """
    Conversation assistant endpoint for social interaction help.

    Analyzes the other person's facial expressions (if image provided),
    processes what they said, and provides suggestions on how to respond.
    Helps users who struggle with social cues or conversation anxiety.
    """
    try:
        # Sanitize input
        what_they_said = sanitize_user_input(request.other_person_said)

        if not what_they_said:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Analyze facial expression if image provided
        emotion_analysis = None
        if request.frame_url or request.frame_base64:
            emotion_analysis = await facial_analyzer.analyze_face(
                image_url=request.frame_url,
                image_base64=request.frame_base64
            )

        # Get conversation suggestions
        suggested, alternatives, tone, social_cues, confidence = await conversation_coach.suggest_response(
            what_they_said=what_they_said,
            emotion_analysis=emotion_analysis,
            conversation_context=request.conversation_context
        )

        # Generate audio for the suggested response
        audio_url = await tts_client.synthesize(suggested)

        return ConversationAssistResponse(
            suggested_response=suggested,
            alternative_responses=alternatives if alternatives else None,
            emotion_analysis=emotion_analysis,
            tone_guidance=tone,
            social_cues=social_cues,
            audio_url=audio_url,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    llm_status = "operational" if llm_client.enabled else "disabled"
    facial_analyzer_status = "operational" if facial_analyzer.enabled else "disabled"
    conversation_coach_status = "operational" if conversation_coach.enabled else "disabled"
    return {
        "status": "healthy",
        "components": {
            "detector": "operational",
            "state_machine": "operational",
            "llm": llm_status,
            "llm_provider": "gemini",
            "facial_analyzer": facial_analyzer_status,
            "conversation_coach": conversation_coach_status,
            "tts": "operational" if tts_client.enabled else "disabled",
        },
        "active_sessions": len(session_manager.sessions),
    }

