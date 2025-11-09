import google.generativeai as genai
import base64
import httpx
from typing import Optional, Tuple
from app.models import EmotionAnalysis
from app.config import settings


class FacialAnalyzer:
    """
    Analyzes facial expressions and emotions using Gemini's vision capabilities.
    """

    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.enabled = bool(self.api_key)

        if self.enabled:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent analysis
                    "top_p": 0.9,
                    "max_output_tokens": 200,
                }
            )

    async def analyze_face(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None
    ) -> Optional[EmotionAnalysis]:
        """
        Analyze facial expressions and emotions from an image.

        Args:
            image_url: URL to the image
            image_base64: Base64 encoded image

        Returns:
            EmotionAnalysis object or None if analysis fails
        """
        if not self.enabled:
            return None

        try:
            # Get image data
            if image_base64:
                image_data = base64.b64decode(image_base64)
            elif image_url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(image_url)
                    if response.status_code != 200:
                        return None
                    image_data = response.content
            else:
                return None

            # Prepare prompt for emotion analysis
            prompt = """Analyze this person's facial expression and emotions. Provide:
1. Primary emotion (happy, sad, angry, surprised, fearful, disgusted, neutral, confused, interested, bored)
2. Confidence level (0.0 to 1.0)
3. Any secondary emotions you detect
4. Notable facial cues (e.g., "slight smile", "furrowed brow", "wide eyes")

Keep your response brief and structured. Focus on what's most relevant for understanding their emotional state."""

            # Call Gemini Vision API
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_data}
            ])

            # Parse response
            analysis_text = response.text.strip()

            # Extract structured data (simplified parsing)
            primary_emotion = self._extract_primary_emotion(analysis_text)
            confidence = self._extract_confidence(analysis_text)
            secondary_emotions = self._extract_secondary_emotions(analysis_text)
            facial_cues = self._extract_facial_cues(analysis_text)

            return EmotionAnalysis(
                primary_emotion=primary_emotion,
                confidence=confidence,
                secondary_emotions=secondary_emotions,
                facial_cues=facial_cues
            )

        except Exception as e:
            print(f"Facial analysis error: {e}")
            return None

    def _extract_primary_emotion(self, text: str) -> str:
        """Extract primary emotion from analysis text"""
        text_lower = text.lower()
        emotions = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral", "confused", "interested", "bored"]

        for emotion in emotions:
            if emotion in text_lower:
                return emotion

        return "neutral"

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from analysis text"""
        # Look for confidence patterns
        import re

        # Try to find explicit confidence values
        confidence_pattern = r'confidence[:\s]+(\d+\.?\d*)[\%]?'
        match = re.search(confidence_pattern, text.lower())

        if match:
            value = float(match.group(1))
            # Normalize if it's a percentage
            if value > 1.0:
                value = value / 100.0
            return min(1.0, max(0.0, value))

        # Default to medium confidence
        return 0.7

    def _extract_secondary_emotions(self, text: str) -> Optional[list]:
        """Extract secondary emotions from analysis text"""
        text_lower = text.lower()
        emotions = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "confused", "interested", "bored"]

        found_emotions = [e for e in emotions if e in text_lower]

        # Remove primary emotion and return rest
        if len(found_emotions) > 1:
            return found_emotions[1:3]  # Return up to 2 secondary emotions

        return None

    def _extract_facial_cues(self, text: str) -> str:
        """Extract facial cues description from analysis text"""
        # Look for common facial cue patterns
        import re

        cue_patterns = [
            r'(slight|warm|big|small|faint)?\s*(smile|smiling)',
            r'(furrowed|raised|arched)?\s*(brow|eyebrows)',
            r'(wide|narrowed)?\s*(eyes)',
            r'(tight|relaxed)?\s*(mouth|lips)',
            r'(tense|relaxed)?\s*(jaw|face)'
        ]

        found_cues = []
        for pattern in cue_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                found_cues.extend([' '.join(filter(None, match)) for match in matches[:2]])

        if found_cues:
            return ', '.join(found_cues[:3])

        return "neutral expression"


class ConversationCoach:
    """
    Provides conversation suggestions based on context, what was said, and facial analysis.
    """

    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.enabled = bool(self.api_key)

        if self.enabled:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 150,
                }
            )

    async def suggest_response(
        self,
        what_they_said: str,
        emotion_analysis: Optional[EmotionAnalysis] = None,
        conversation_context: Optional[str] = None
    ) -> Tuple[str, list, str, str, float]:
        """
        Generate conversation suggestions.

        Args:
            what_they_said: What the other person just said
            emotion_analysis: Optional facial emotion analysis
            conversation_context: Optional conversation context

        Returns:
            Tuple of (suggested_response, alternatives, tone_guidance, social_cues, confidence)
        """
        if not self.enabled:
            return self._get_fallback_response(what_they_said)

        try:
            # Build prompt with all available context
            prompt = self._build_conversation_prompt(
                what_they_said,
                emotion_analysis,
                conversation_context
            )

            # Call Gemini
            response = self.model.generate_content(prompt)
            generated = response.text.strip()

            # Parse the response
            suggested, alternatives, tone, social_cues = self._parse_response(generated)

            # Calculate confidence based on emotion analysis availability
            confidence = 0.85 if emotion_analysis else 0.7

            return suggested, alternatives, tone, social_cues, confidence

        except Exception as e:
            print(f"Conversation coach error: {e}")
            return self._get_fallback_response(what_they_said)

    def _build_conversation_prompt(
        self,
        what_they_said: str,
        emotion_analysis: Optional[EmotionAnalysis],
        conversation_context: Optional[str]
    ) -> str:
        """Build prompt for conversation assistance"""

        prompt = f"""You are a helpful social companion assisting someone who may struggle with social cues or conversation anxiety.

The other person just said: "{what_they_said}"
"""

        if emotion_analysis:
            prompt += f"""
Their facial expression shows: {emotion_analysis.primary_emotion} emotion
Facial cues: {emotion_analysis.facial_cues}
"""

        if conversation_context:
            prompt += f"""
Conversation context: {conversation_context}
"""

        prompt += """
Provide a helpful response in this EXACT format:

SUGGESTED RESPONSE: [One natural, appropriate thing to say back - keep it conversational and authentic]

ALTERNATIVES:
1. [Alternative response option 1]
2. [Alternative response option 2]

TONE: [Recommended tone: friendly/empathetic/casual/professional/warm]

SOCIAL CUES: [Brief interpretation of what they might be feeling or wanting from this interaction]

Keep everything natural and socially appropriate. Make suggestions that help maintain genuine connection."""

        return prompt

    def _parse_response(self, generated: str) -> Tuple[str, list, str, str]:
        """Parse the structured response from Gemini"""
        import re

        # Extract suggested response
        suggested_match = re.search(r'SUGGESTED RESPONSE:\s*(.+?)(?=\n\n|ALTERNATIVES:|$)', generated, re.DOTALL)
        suggested = suggested_match.group(1).strip() if suggested_match else "That's interesting. Tell me more about that."

        # Extract alternatives
        alternatives_section = re.search(r'ALTERNATIVES:\s*(.+?)(?=\n\nTONE:|TONE:|$)', generated, re.DOTALL)
        alternatives = []
        if alternatives_section:
            alt_text = alternatives_section.group(1)
            alt_matches = re.findall(r'\d+\.\s*(.+)', alt_text)
            alternatives = [alt.strip() for alt in alt_matches[:2]]

        # Extract tone
        tone_match = re.search(r'TONE:\s*(.+?)(?=\n\n|SOCIAL CUES:|$)', generated, re.DOTALL)
        tone = tone_match.group(1).strip() if tone_match else "friendly"

        # Extract social cues
        social_match = re.search(r'SOCIAL CUES:\s*(.+?)$', generated, re.DOTALL)
        social_cues = social_match.group(1).strip() if social_match else "They seem to be sharing information with you."

        return suggested, alternatives, tone, social_cues

    def _get_fallback_response(self, what_they_said: str) -> Tuple[str, list, str, str, float]:
        """Fallback response when Gemini is unavailable"""
        suggested = "That's interesting. Could you tell me more about that?"
        alternatives = [
            "I'd love to hear more about that.",
            "That sounds important. How does that make you feel?"
        ]
        tone = "friendly"
        social_cues = "They're sharing something with you and likely want engagement."
        confidence = 0.5

        return suggested, alternatives, tone, social_cues, confidence
