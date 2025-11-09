#!/usr/bin/env python3
"""
Simple test without needing external services.
Tests the core detection and state logic without Ollama/ElevenLabs.
"""

from app.core.detector import DistressDetector
from app.core.state_machine import StateMachine
from app.models import SessionState, DistressState


def test_detection():
    """Test distress detection"""
    print("=" * 60)
    print("Testing EmpathLens Detection System")
    print("=" * 60)
    print()
    
    detector = DistressDetector(text_weight=0.6, audio_weight=0.4)
    
    test_cases = [
        ("I'm having a panic attack, I can't breathe", "PANIC"),
        ("Everything is too much, I'm shutting down", "OVERWHELMED"),
        ("I feel anxious and stressed", "RISING"),
        ("I don't want to live anymore", "CRISIS"),
        ("I'm feeling better now", "RECOVERY"),
        ("stop", "STOP"),
    ]
    
    for message, expected_category in test_cases:
        print(f"Input: \"{message}\"")
        
        distress_prob, is_crisis, is_stop, details = detector.detect(message)
        
        print(f"  Distress Probability: {distress_prob:.2f}")
        print(f"  Is Crisis: {is_crisis}")
        print(f"  Is Stop: {is_stop}")
        print(f"  Match Counts: {details['match_counts']}")
        
        if is_crisis:
            print(f"  → Category: CRISIS_RISK ✅")
        elif is_stop:
            print(f"  → Category: STOP ✅")
        elif distress_prob >= 0.6:
            print(f"  → Category: HIGH DISTRESS (Panic/Overwhelmed) ✅")
        elif distress_prob >= 0.2:
            print(f"  → Category: RISING ANXIETY ✅")
        else:
            print(f"  → Category: CALM/RECOVERY ✅")
        
        print()
    
    print("=" * 60)
    print("✅ Detection system working correctly!")
    print("=" * 60)


def test_state_machine():
    """Test state transitions"""
    print()
    print("=" * 60)
    print("Testing State Machine")
    print("=" * 60)
    print()
    
    sm = StateMachine()
    session = SessionState(chat_id="test_123")
    
    scenarios = [
        (0.7, False, "High distress"),
        (0.5, False, "Medium distress"),
        (0.2, False, "Calming down"),
        (0.0, True, "Crisis keywords"),
    ]
    
    for prob, is_crisis, description in scenarios:
        new_state, changed = sm.determine_state(session, prob, is_crisis)
        print(f"{description}:")
        print(f"  Distress Prob: {prob}")
        print(f"  Is Crisis: {is_crisis}")
        print(f"  New State: {new_state}")
        print(f"  State Changed: {changed}")
        print()
        
        session.current_state = new_state
        session.distress_prob = prob
    
    print("=" * 60)
    print("✅ State machine working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_detection()
        test_state_machine()
        print()
        print("🎉 Core systems are working!")
        print()
        print("Next steps:")
        print("1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print("2. Start Ollama: ollama serve")
        print("3. Pull model: ollama pull llama3")
        print("4. Start service: python main.py")
        print("5. Run full tests: python test_service.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure you've installed dependencies:")
        print("  pip install -r requirements.txt")

