import re
from typing import Tuple, Dict
from app.models import DistressState


class TextDistressDetector:
    """
    Detects distress levels and crisis keywords from text.
    Primary signal source (0.6 weight by default).
    """
    
    # Crisis keywords that trigger immediate escalation
    CRISIS_KEYWORDS = [
        r"\bhurt\s+myself\b",
        r"\bkill\s+myself\b",
        r"\bsuicide\b",
        r"\bend\s+it\s+all\b",
        r"\bdon'?t\s+want\s+to\s+live\b",
        r"\bwant\s+to\s+die\b",
        r"\bno\s+point\s+(in\s+)?living\b",
        r"\bcan'?t\s+go\s+on\b",
    ]
    
    # Panic indicators (high intensity, acute)
    PANIC_PATTERNS = [
        r"\bpanic\s+attack\b",
        r"\bcan'?t\s+breathe\b",
        r"\bheart\s+(is\s+)?racing\b",
        r"\blosing\s+control\b",
        r"\bgoing\s+to\s+die\b",
        r"\bchest\s+(is\s+)?tight\b",
        r"\bfreaking\s+out\b",
        r"\bterri(fied|ble)\b",
    ]
    
    # Overwhelmed indicators (high intensity, freeze/shutdown)
    OVERWHELMED_PATTERNS = [
        r"\btoo\s+much\b",
        r"\bover(whelm|load)(ed|ing)?\b",
        r"\bshut(ting)?\s+down\b",
        r"\bcan'?t\s+handle\b",
        r"\bcan'?t\s+think\b",
        r"\bparalyz(ed|ing)\b",
        r"\bfrozen\b",
        r"\bcollapsing\b",
    ]
    
    # Rising anxiety indicators (medium intensity)
    RISING_PATTERNS = [
        r"\banxious\b",
        r"\bworri(ed|es)\b",
        r"\bnervous\b",
        r"\bstress(ed|ful)\b",
        r"\bscared\b",
        r"\buneasy\b",
        r"\btense\b",
        r"\bon\s+edge\b",
    ]
    
    # Recovery/calming indicators
    RECOVERY_PATTERNS = [
        r"\bfeeling\s+better\b",
        r"\bcalm(er|ing)\b",
        r"\bsettl(ing|ed)\b",
        r"\brelax(ed|ing)\b",
        r"\bimproved\b",
        r"\beasier\b",
    ]
    
    # Stop words
    STOP_WORDS = [
        r"\bstop\b",
        r"\bend\s+(this|session)\b",
        r"\bno\s+more\b",
        r"\bleave\s+me\s+alone\b",
    ]
    
    def __init__(self):
        self.crisis_regex = re.compile("|".join(self.CRISIS_KEYWORDS), re.IGNORECASE)
        self.panic_regex = re.compile("|".join(self.PANIC_PATTERNS), re.IGNORECASE)
        self.overwhelmed_regex = re.compile("|".join(self.OVERWHELMED_PATTERNS), re.IGNORECASE)
        self.rising_regex = re.compile("|".join(self.RISING_PATTERNS), re.IGNORECASE)
        self.recovery_regex = re.compile("|".join(self.RECOVERY_PATTERNS), re.IGNORECASE)
        self.stop_regex = re.compile("|".join(self.STOP_WORDS), re.IGNORECASE)
    
    def detect(self, text: str) -> Tuple[float, bool, bool, Dict[str, int]]:
        """
        Analyze text for distress signals.
        
        Args:
            text: User's message text
            
        Returns:
            Tuple of:
            - distress_prob (0.0-1.0)
            - is_crisis (bool)
            - is_stop (bool)
            - match_counts (dict of pattern type -> count)
        """
        if not text or not text.strip():
            return 0.0, False, False, {}
        
        text = text.lower().strip()
        
        # Check for stop words
        is_stop = bool(self.stop_regex.search(text))
        
        # Check for crisis keywords (highest priority)
        is_crisis = bool(self.crisis_regex.search(text))
        
        # Count pattern matches
        match_counts = {
            "crisis": len(self.crisis_regex.findall(text)),
            "panic": len(self.panic_regex.findall(text)),
            "overwhelmed": len(self.overwhelmed_regex.findall(text)),
            "rising": len(self.rising_regex.findall(text)),
            "recovery": len(self.recovery_regex.findall(text)),
        }
        
        # Calculate distress probability
        distress_prob = self._calculate_probability(match_counts)
        
        return distress_prob, is_crisis, is_stop, match_counts
    
    def _calculate_probability(self, match_counts: Dict[str, int]) -> float:
        """Calculate distress probability from pattern matches"""
        
        # Recovery reduces distress
        if match_counts["recovery"] > 0:
            return max(0.0, 0.2 - (match_counts["recovery"] * 0.1))
        
        # Crisis is maximum distress (but handled separately)
        if match_counts["crisis"] > 0:
            return 1.0
        
        # Calculate weighted score
        score = 0.0
        
        # Panic patterns (high weight)
        score += match_counts["panic"] * 0.85
        
        # Overwhelmed patterns (high weight)
        score += match_counts["overwhelmed"] * 0.80
        
        # Rising anxiety patterns (medium weight)
        score += match_counts["rising"] * 0.40
        
        # Normalize to 0.0-1.0
        # Cap at 0.95 for non-crisis
        prob = min(0.95, score / 2.0)
        
        return prob


class AudioDistressDetector:
    """
    Optional audio prosody detector.
    Currently a placeholder - would analyze pitch, loudness, pauses.
    """
    
    def __init__(self):
        self.enabled = False
    
    def detect(self, audio_features: Dict) -> float:
        """
        Analyze audio prosody features for distress.
        
        Args:
            audio_features: Dict with pitch, loudness, pause metrics
            
        Returns:
            distress_prob (0.0-1.0)
        """
        if not self.enabled or not audio_features:
            return 0.0
        
        # Placeholder: would implement prosody analysis here
        # - High pitch variance
        # - Increased loudness
        # - Shorter pauses (rapid speech)
        # - Tremor/shakiness
        
        return 0.0


class DistressDetector:
    """
    Main detector that fuses text and optional audio signals.
    """
    
    def __init__(self, text_weight: float = 0.6, audio_weight: float = 0.4):
        self.text_detector = TextDistressDetector()
        self.audio_detector = AudioDistressDetector()
        self.text_weight = text_weight
        self.audio_weight = audio_weight
    
    def detect(
        self,
        text: str,
        audio_features: Dict = None
    ) -> Tuple[float, bool, bool, Dict]:
        """
        Detect distress from text and optional audio.
        
        Args:
            text: User's message text
            audio_features: Optional audio prosody features
            
        Returns:
            Tuple of (distress_prob, is_crisis, is_stop, details)
        """
        # Text detection (primary signal)
        text_prob, is_crisis, is_stop, match_counts = self.text_detector.detect(text)
        
        # Audio detection (optional, secondary signal)
        audio_prob = 0.0
        if audio_features:
            audio_prob = self.audio_detector.detect(audio_features)
        
        # Fusion: text has priority
        # If audio is disabled, audio_weight effectively becomes 0
        final_prob = (self.text_weight * text_prob) + (self.audio_weight * audio_prob)
        
        # Ensure within bounds
        final_prob = max(0.0, min(1.0, final_prob))
        
        details = {
            "text_prob": text_prob,
            "audio_prob": audio_prob,
            "final_prob": final_prob,
            "match_counts": match_counts,
            "is_crisis": is_crisis,
            "is_stop": is_stop,
        }
        
        return final_prob, is_crisis, is_stop, details

