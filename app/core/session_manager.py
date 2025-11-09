from datetime import datetime, timedelta
from typing import Dict, Optional
from app.models import SessionState, DistressState, InterventionType
from app.config import settings


class SessionManager:
    """
    Manages per-conversation session state.
    Tracks distress progression, interventions, and timing.
    """
    
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self.timeout_minutes = settings.session_timeout_minutes
    
    def get_or_create_session(self, chat_id: str) -> SessionState:
        """Get existing session or create new one"""
        
        # Clean up old sessions first
        self._cleanup_old_sessions()
        
        if chat_id not in self.sessions:
            self.sessions[chat_id] = SessionState(chat_id=chat_id)
        
        session = self.sessions[chat_id]
        session.last_update = datetime.utcnow()
        return session
    
    def update_session(
        self,
        chat_id: str,
        state: Optional[DistressState] = None,
        distress_prob: Optional[float] = None,
        intervention: Optional[InterventionType] = None,
        escalation_offered: bool = False,
        stopped: bool = False
    ) -> SessionState:
        """Update session with new information"""
        
        session = self.get_or_create_session(chat_id)
        
        if state is not None:
            session.current_state = state
        
        if distress_prob is not None:
            session.distress_prob = distress_prob
        
        if intervention is not None:
            session.last_intervention = intervention
            session.last_intervention_time = datetime.utcnow()
            session.intervention_count += 1
        
        if escalation_offered:
            session.escalation_offered = True
        
        if stopped:
            session.stopped = True
        
        session.last_update = datetime.utcnow()
        return session
    
    def should_check_in(self, chat_id: str) -> bool:
        """
        Determine if it's time for a check-in.
        Check-in after each intervention cycle.
        """
        session = self.get_or_create_session(chat_id)
        
        # Don't check in if stopped or in crisis
        if session.stopped or session.current_state == DistressState.CRISIS_RISK:
            return False
        
        # Check in after intervention completes
        if session.last_intervention_time:
            cooldown = settings.intervention_cooldown_seconds
            time_since = (datetime.utcnow() - session.last_intervention_time).total_seconds()
            
            if time_since >= cooldown:
                return True
        
        return False
    
    def can_intervene(self, chat_id: str) -> bool:
        """Check if we're in cooldown period"""
        session = self.get_or_create_session(chat_id)
        
        if not session.last_intervention_time:
            return True
        
        cooldown = settings.intervention_cooldown_seconds
        time_since = (datetime.utcnow() - session.last_intervention_time).total_seconds()
        
        return time_since >= cooldown
    
    def get_session_duration(self, chat_id: str) -> int:
        """Get session duration in seconds"""
        session = self.get_or_create_session(chat_id)
        return int((datetime.utcnow() - session.session_start).total_seconds())
    
    def end_session(self, chat_id: str):
        """End and remove a session"""
        self.sessions.pop(chat_id, None)
    
    def _cleanup_old_sessions(self):
        """Remove sessions that haven't been updated recently"""
        cutoff = datetime.utcnow() - timedelta(minutes=self.timeout_minutes)
        
        expired = [
            chat_id for chat_id, session in self.sessions.items()
            if session.last_update < cutoff
        ]
        
        for chat_id in expired:
            self.sessions.pop(chat_id, None)

