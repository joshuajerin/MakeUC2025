from app.models import DistressState, InterventionType
from typing import Tuple, Optional


class InterventionManager:
    """
    Maps distress states to appropriate micro-interventions.
    Each intervention is a brief, concrete, therapist-inspired technique.
    """
    
    # Intervention definitions: (type, duration_sec, initial_prompt)
    INTERVENTIONS = {
        DistressState.RISING: (
            InterventionType.PACED_BREATHING,
            30,
            "Let's slow your breath. Breathe in for four, out for four."
        ),
        DistressState.PANIC: (
            InterventionType.FOUR_SEVEN_EIGHT,
            60,
            "Your body needs slower breaths. In for four, hold seven, out for eight."
        ),
        DistressState.OVERWHELMED: (
            InterventionType.GROUNDING_54321,
            75,
            "Let's bring you back to now. Name five things you can see."
        ),
        DistressState.RECOVERY: (
            InterventionType.REINFORCEMENT,
            0,
            "Your body is settling. Take two slow breaths."
        ),
        DistressState.CRISIS_RISK: (
            InterventionType.ESCALATION,
            0,
            "I hear you're struggling. Would you like me to contact your support person?"
        ),
    }
    
    # Follow-up prompts for grounding sequence
    GROUNDING_SEQUENCE = [
        "Name five things you can see.",
        "Name four things you can touch.",
        "Name three things you can hear.",
        "Name two things you can smell.",
        "Name one thing you can taste.",
        "You're here. You're present. Take a slow breath."
    ]
    
    # Breathing cues
    BREATHING_CUES = {
        InterventionType.PACED_BREATHING: [
            "Breathe in for four, out for four.",
            "Keep that rhythm. In for four, out for four.",
            "You're doing well. Two more breaths.",
        ],
        InterventionType.FOUR_SEVEN_EIGHT: [
            "In for four, hold for seven, out for eight.",
            "Again. In four, hold seven, out eight.",
            "One more cycle. In four, hold seven, out eight.",
        ],
        InterventionType.BOX_BREATHING: [
            "In for four, hold four, out four, hold four.",
            "Continue the box. In, hold, out, hold.",
            "You've got this. One more round.",
        ]
    }
    
    def __init__(self):
        self.grounding_step = {}  # chat_id -> current step
        self.breathing_step = {}  # chat_id -> current step
    
    def get_intervention(
        self,
        state: DistressState,
        chat_id: str,
        step: int = 0
    ) -> Tuple[InterventionType, int, str]:
        """
        Get the appropriate intervention for a state.
        
        Args:
            state: Current distress state
            chat_id: Conversation identifier
            step: Current step in multi-step intervention
            
        Returns:
            Tuple of (intervention_type, duration_sec, prompt_text)
        """
        if state == DistressState.CALM:
            return InterventionType.CHECK_IN, 0, "How are you feeling right now?"
        
        if state not in self.INTERVENTIONS:
            return InterventionType.CHECK_IN, 0, "How are you doing?"
        
        intervention_type, duration, initial_prompt = self.INTERVENTIONS[state]
        
        # Handle multi-step interventions
        if intervention_type == InterventionType.GROUNDING_54321:
            if chat_id not in self.grounding_step:
                self.grounding_step[chat_id] = 0
            
            step_idx = self.grounding_step[chat_id]
            if step_idx < len(self.GROUNDING_SEQUENCE):
                prompt = self.GROUNDING_SEQUENCE[step_idx]
                self.grounding_step[chat_id] += 1
            else:
                prompt = "You're here. You're present. Take a slow breath."
                self.grounding_step[chat_id] = 0  # Reset
            
            return intervention_type, duration // len(self.GROUNDING_SEQUENCE), prompt
        
        elif intervention_type in self.BREATHING_CUES:
            if chat_id not in self.breathing_step:
                self.breathing_step[chat_id] = 0
            
            step_idx = self.breathing_step[chat_id]
            cues = self.BREATHING_CUES[intervention_type]
            
            if step_idx < len(cues):
                prompt = cues[step_idx]
                self.breathing_step[chat_id] += 1
            else:
                prompt = "You're doing well. One more breath."
                self.breathing_step[chat_id] = 0  # Reset
            
            return intervention_type, duration // len(cues), prompt
        
        # Single-step interventions
        return intervention_type, duration, initial_prompt
    
    def get_checkin_prompt(self) -> Tuple[str, list]:
        """Get check-in prompt and button options"""
        return (
            "How are you feeling now?",
            ["better", "same", "worse"]
        )
    
    def get_escalation_prompt(self) -> str:
        """Get escalation offer prompt"""
        return "I hear you're struggling. Would you like me to contact your support person?"
    
    def reset_steps(self, chat_id: str):
        """Reset intervention steps for a conversation"""
        self.grounding_step.pop(chat_id, None)
        self.breathing_step.pop(chat_id, None)

