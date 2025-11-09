from app.models import DistressState, SessionState
from typing import Tuple
from datetime import datetime, timedelta


class StateMachine:
    """
    State machine for managing distress state transitions.
    Uses hysteresis to avoid rapid state flipping.
    """
    
    # State thresholds (distress_prob ranges)
    THRESHOLDS = {
        DistressState.CALM: (0.0, 0.2),
        DistressState.RISING: (0.2, 0.45),
        DistressState.PANIC: (0.6, 1.0),
        DistressState.OVERWHELMED: (0.5, 0.85),
        DistressState.RECOVERY: (0.1, 0.35),
    }
    
    # Hysteresis settings: require consecutive confirmations
    # Set to 1 for immediate response (good for demos)
    CONFIRMATION_WINDOW = 1
    
    def __init__(self):
        self.state_history = {}  # chat_id -> list of recent states
    
    def determine_state(
        self,
        session: SessionState,
        distress_prob: float,
        is_crisis: bool
    ) -> Tuple[DistressState, bool]:
        """
        Determine the next state based on distress probability and current context.
        
        Args:
            session: Current session state
            distress_prob: Calculated distress probability (0.0-1.0)
            is_crisis: Whether crisis keywords were detected
            
        Returns:
            Tuple of (new_state, state_changed)
        """
        if is_crisis:
            return DistressState.CRISIS_RISK, session.current_state != DistressState.CRISIS_RISK
        
        if session.stopped:
            return DistressState.CALM, False
        
        current_state = session.current_state
        proposed_state = self._map_prob_to_state(distress_prob, current_state)
        
        # Apply hysteresis
        chat_id = session.chat_id
        if chat_id not in self.state_history:
            self.state_history[chat_id] = []
        
        self.state_history[chat_id].append(proposed_state)
        
        # Keep only recent history
        if len(self.state_history[chat_id]) > 5:
            self.state_history[chat_id] = self.state_history[chat_id][-5:]
        
        # Require confirmation for state changes (smoothing)
        recent = self.state_history[chat_id][-self.CONFIRMATION_WINDOW:]
        if len(recent) >= self.CONFIRMATION_WINDOW and all(s == proposed_state for s in recent):
            confirmed_state = proposed_state
        else:
            confirmed_state = current_state
        
        state_changed = confirmed_state != current_state
        return confirmed_state, state_changed
    
    def _map_prob_to_state(self, prob: float, current_state: DistressState) -> DistressState:
        """Map distress probability to a state, considering current state"""
        
        # Special handling for recovery
        if current_state == DistressState.RECOVERY and prob < 0.25:
            return DistressState.CALM
        
        # Check for panic (highest priority)
        if prob >= 0.6:
            # Distinguish panic vs overwhelm based on keywords (simplified)
            return DistressState.PANIC
        
        # Check for overwhelmed
        if 0.5 <= prob < 0.6:
            return DistressState.OVERWHELMED
        
        # Rising anxiety
        if 0.2 <= prob < 0.5:
            return DistressState.RISING
        
        # Recovery state (after intervention)
        if current_state in [DistressState.PANIC, DistressState.OVERWHELMED, DistressState.RISING]:
            if 0.1 <= prob < 0.35:
                return DistressState.RECOVERY
        
        # Calm
        return DistressState.CALM
    
    def should_escalate(self, session: SessionState) -> bool:
        """
        Determine if we should offer escalation based on session history.
        
        Escalate if:
        - Crisis state detected, OR
        - User has been in high distress for >2 minutes
        """
        if session.current_state == DistressState.CRISIS_RISK:
            return True
        
        # Check if in distress for too long
        if session.current_state in [DistressState.PANIC, DistressState.OVERWHELMED]:
            duration = (datetime.utcnow() - session.session_start).total_seconds()
            if duration > 120:  # 2 minutes
                return True
        
        return False

