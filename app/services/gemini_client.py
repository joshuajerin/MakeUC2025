import google.generativeai as genai
from typing import Optional, Dict
from app.models import DistressState, InterventionType
from app.config import settings


class GeminiClient:
    """
    Client for interacting with Google Gemini API.
    Generates short, empathetic coaching lines.
    """
    
    SYSTEM_PROMPT = """You are a calm, supportive crisis companion helping someone through a distress episode.

STRICT RULES:
1. ONE sentence per response, maximum 18 words
2. Use ONLY these techniques:
   - Paced breathing (4-4, 4-7-8, or box 4-4-4-4)
   - 5-4-3-2-1 sensory grounding
   - Present-moment orientation
   - Brief validation
3. Be warm, concrete, non-clinical
4. NO metaphors, NO "close your eyes"
5. When asked for breathing: give specific counts (e.g., "In for four, hold seven, out for eight")
6. When asked for grounding: give specific sensory targets (e.g., "Name five things you see")
7. For reinforcement: acknowledge progress in present tense
8. NEVER give medical advice, diagnoses, or suggest medication

Style: Direct, gentle, one action at a time."""
    
    # Fallback responses if Gemini is unavailable
    FALLBACK_RESPONSES = {
        DistressState.RISING: "Let's slow your breath. In for four, out for four.",
        DistressState.PANIC: "In for four, hold seven, out for eight.",
        DistressState.OVERWHELMED: "Name five things you can see right now.",
        DistressState.RECOVERY: "Your body is settling. Take two slow breaths.",
        DistressState.CRISIS_RISK: "I hear you're struggling. Can I contact your support person?",
        DistressState.CALM: "How are you feeling right now?",
    }
    
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 50,
                }
            )
    
    async def generate_response(
        self,
        user_message: str,
        current_state: DistressState,
        distress_prob: float,
        last_intervention: Optional[InterventionType] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate a coaching response using Gemini.
        
        Args:
            user_message: User's cleaned message text
            current_state: Current distress state
            distress_prob: Distress probability (0.0-1.0)
            last_intervention: Previous intervention type
            context: Additional context
            
        Returns:
            Generated coaching line (single sentence, ≤18 words)
        """
        if not self.enabled:
            return self._get_fallback(current_state)
        
        try:
            prompt = self._build_prompt(
                user_message,
                current_state,
                distress_prob,
                last_intervention
            )
            
            # Call Gemini API
            response = self.model.generate_content(
                f"{self.SYSTEM_PROMPT}\n\n{prompt}"
            )
            
            generated = response.text.strip()
            
            # Take only first sentence
            if "." in generated:
                generated = generated.split(".")[0] + "."
            
            return generated
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_fallback(current_state)
    
    def _build_prompt(
        self,
        user_message: str,
        current_state: DistressState,
        distress_prob: float,
        last_intervention: Optional[InterventionType]
    ) -> str:
        """Build prompt for Gemini with context"""
        
        state_guidance = {
            DistressState.RISING: "Guide them to paced breathing (4-4 or 4-7-8). Be specific with counts.",
            DistressState.PANIC: "Use 4-7-8 breathing. Say the exact counts: in four, hold seven, out eight.",
            DistressState.OVERWHELMED: "Use 5-4-3-2-1 grounding. Start with 'Name five things you see.'",
            DistressState.RECOVERY: "Validate their progress. Keep it brief and present-tense.",
            DistressState.CRISIS_RISK: "Offer to contact support. Don't attempt coaching.",
            DistressState.CALM: "Ask a simple check-in question.",
        }
        
        guidance = state_guidance.get(current_state, "Respond with empathy and one concrete action.")
        
        prompt = f"""User's state: {current_state.value} (distress level: {distress_prob:.2f})
User said: "{user_message}"

{guidance}

Respond with ONE sentence, maximum 18 words:"""
        
        return prompt
    
    def _get_fallback(self, state: DistressState) -> str:
        """Get fallback response when Gemini is unavailable"""
        return self.FALLBACK_RESPONSES.get(state, "Take a slow breath.")

