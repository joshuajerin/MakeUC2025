import re
from typing import Tuple


class SafetyFilter:
    """
    Post-processing filter for LLM outputs.
    Ensures responses are safe, appropriate, and within guidelines.
    """
    
    # Medical/clinical terms to flag
    MEDICAL_TERMS = [
        r"\bdiagnos(is|e|ed)\b",
        r"\bmedication\b",
        r"\bprescri(be|ption)\b",
        r"\bdisorder\b",
        r"\btherapy\b",
        r"\bcounseling\b",
        r"\bpsychiatri(st|c)\b",
        r"\btreatment\b",
        r"\bdoctor\b",
    ]
    
    # Problematic phrases
    PROBLEMATIC_PHRASES = [
        (r"\bclose\s+your\s+eyes\b", "soften your gaze"),
        (r"\bshut\s+your\s+eyes\b", "soften your gaze"),
        (r"\brelax\s+completely\b", "let your shoulders drop"),
        (r"\bdon'?t\s+worry\b", "you're safe right now"),
    ]
    
    def __init__(self, max_words: int = 18):
        self.max_words = max_words
        self.medical_regex = re.compile("|".join(self.MEDICAL_TERMS), re.IGNORECASE)
    
    def filter(self, text: str) -> Tuple[str, bool]:
        """
        Filter and validate LLM output.
        
        Args:
            text: Raw LLM output
            
        Returns:
            Tuple of (filtered_text, is_safe)
        """
        if not text or not text.strip():
            return "Take a slow breath.", True
        
        filtered = text.strip()
        
        # Check for medical terms (flag as unsafe)
        if self.medical_regex.search(filtered):
            return "Let's focus on your breath right now.", False
        
        # Replace problematic phrases
        for pattern, replacement in self.PROBLEMATIC_PHRASES:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)
        
        # Enforce word limit
        words = filtered.split()
        if len(words) > self.max_words:
            filtered = " ".join(words[:self.max_words])
            # Ensure it ends with punctuation
            if not filtered.endswith((".", "?", "!")):
                filtered += "."
        
        # Ensure first letter is capitalized
        if filtered:
            filtered = filtered[0].upper() + filtered[1:]
        
        # Ensure ends with punctuation
        if not filtered.endswith((".", "?", "!")):
            filtered += "."
        
        return filtered, True
    
    def is_safe_length(self, text: str) -> bool:
        """Check if text is within safe word limit"""
        return len(text.split()) <= self.max_words
    
    def truncate(self, text: str) -> str:
        """Truncate text to max words"""
        words = text.split()
        if len(words) > self.max_words:
            truncated = " ".join(words[:self.max_words])
            if not truncated.endswith((".", "?", "!")):
                truncated += "."
            return truncated
        return text


def sanitize_user_input(text: str) -> str:
    """
    Clean user input before processing.
    Remove excessive whitespace, normalize punctuation.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Normalize quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Limit length (prevent abuse)
    max_length = 500
    if len(text) > max_length:
        text = text[:max_length]
    
    return text

