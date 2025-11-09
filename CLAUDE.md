# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EmpathLens** is an AI social companion that integrates with Meta Ray-Ban Smart Glasses to provide two key features:

1. **Panic Attack & Distress Management**: Real-time voice-based support during anxiety, panic, or overwhelming moments
2. **Social Conversation Assistant**: Helps people who struggle with social cues or conversation anxiety by analyzing facial expressions and suggesting appropriate responses

**CRITICAL DISCLAIMER**: This is NOT a medical device. This project provides brief distress management techniques and conversation assistance but is not a substitute for professional mental health care.

## Architecture

### System Flow
```
User (Meta Glasses) → Messenger Extension → FastAPI Server → Google Gemini AI → Response
                                                ↓
                                         ElevenLabs TTS → Voice Output
```

### Core Components

1. **main.py**: FastAPI server with four primary endpoints:
   - `POST /distress/infer`: Main distress detection and intervention endpoint
   - `POST /distress/checkin`: Follow-up check-ins (better/same/worse responses)
   - `POST /conversation/assist`: NEW - Social conversation assistance with facial analysis
   - `POST /distress/stop`: Stop intervention for a conversation

2. **Gemini Integration**:
   - **gemini_client.py**: Handles distress response generation with strict ≤18 word limit
   - **conversation_assistant.py**: NEW - Two classes:
     - `FacialAnalyzer`: Uses Gemini Vision API to analyze facial expressions and emotions
     - `ConversationCoach`: Provides conversation suggestions based on context and emotions

3. **State Machine** (state_machine.py): Manages transitions between distress states with hysteresis to prevent rapid flipping. Uses `CONFIRMATION_WINDOW = 1` for demos (immediate response).

4. **Detector** (detector.py): Fuses text and optional audio signals:
   - `TextDistressDetector`: Primary signal (0.6 weight) using regex pattern matching for panic, overwhelm, rising anxiety, recovery, and crisis keywords
   - `AudioDistressDetector`: Currently placeholder (0.4 weight)

5. **Interventions** (interventions.py): Maps states to therapeutic techniques:
   - RISING → Paced breathing (4-4)
   - PANIC → 4-7-8 breathing
   - OVERWHELMED → 5-4-3-2-1 grounding sequence
   - RECOVERY → Brief reinforcement
   - CRISIS_RISK → Escalation offer

6. **Safety** (safety.py): Post-LLM filtering to ensure responses are safe, brief, and appropriate

7. **Session Manager** (session_manager.py): Tracks ephemeral session state (no persistent storage)

### State Definitions (models.py)

Distress states: CALM, RISING, PANIC, OVERWHELMED, RECOVERY, CRISIS_RISK

Key thresholds in `state_machine.py`:
- CALM: 0.0-0.2
- RISING: 0.2-0.45
- PANIC: 0.6-1.0
- OVERWHELMED: 0.5-0.85
- Escalation timeout: 120 seconds

### NEW: Conversation Assistant Models (models.py)

- `ConversationAssistRequest`: Input with what person said + optional face image
- `ConversationAssistResponse`: Suggested response + alternatives + emotion analysis + social cues
- `EmotionAnalysis`: Structured facial emotion data (primary emotion, confidence, facial cues)

## Development Commands

### Setup and Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cat > .env << EOF
# Gemini Configuration (Required)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# ElevenLabs (Optional, for voice)
ELEVEN_API_KEY=your_elevenlabs_key_here
ELEVEN_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Detection weights
TEXT_WEIGHT=0.6
AUDIO_WEIGHT=0.4

# Features
ENABLE_AUDIO=false

# Server
HOST=0.0.0.0
PORT=8000
EOF
```

**Get Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### Running the Service

```bash
# Easy way: Use the start script
./scripts/start.sh

# Or manually with auto-reload for development
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The service runs on `http://localhost:8000` by default.

### Testing

```bash
# Run comprehensive test suite
python tests/test_service.py

# Run simple test
python tests/simple_test.py

# Test distress detection
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{"message": "I'\''m having a panic attack", "chat_id": "test_123"}'

# Test conversation assistant
curl -X POST http://localhost:8000/conversation/assist \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test_123", "other_person_said": "Hey, how are you?"}'

# Health check
curl http://localhost:8000/health
```

### Debugging

```bash
# Check Gemini API connectivity
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print('Connected!')"

# Test individual components
python -c "from detector import TextDistressDetector; d = TextDistressDetector(); print(d.detect('I am having a panic attack'))"
```

## Configuration

### Gemini API (Required)

The system uses Google Gemini for:
1. **Distress Response Generation** (gemini_client.py): Short, empathetic coaching responses
2. **Facial Emotion Analysis** (conversation_assistant.py): Vision API for analyzing facial expressions
3. **Conversation Coaching** (conversation_assistant.py): Context-aware conversation suggestions

All three use the same API key but different model configurations:
- Distress responses: Temperature 0.7, max 50 tokens
- Facial analysis: Temperature 0.3 (more consistent), max 200 tokens
- Conversation coaching: Temperature 0.7, max 150 tokens

### ElevenLabs TTS (Optional)

```bash
ELEVEN_API_KEY=your_key_here
ELEVEN_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

Service works without TTS (text-only mode).

## Key Implementation Details

### Distress Detection Pipeline

1. **Input sanitization** (safety.py): Clean user input
2. **Text analysis** (detector.py): Regex pattern matching for distress keywords
3. **State determination** (state_machine.py): Map probability to state with hysteresis
4. **Crisis detection**: Immediate escalation on crisis keywords (suicide, self-harm)
5. **Stop command**: "stop" keyword ends intervention immediately

### Conversation Assistant Pipeline (NEW)

1. **Input**: User provides what other person said + optional face image
2. **Facial analysis** (conversation_assistant.py:FacialAnalyzer):
   - Accepts image URL or base64 encoded image
   - Uses Gemini Vision API to analyze emotions
   - Extracts: primary emotion, confidence, secondary emotions, facial cues
3. **Conversation coaching** (conversation_assistant.py:ConversationCoach):
   - Takes what was said + emotion analysis + optional context
   - Generates suggested response + 2 alternatives
   - Provides tone guidance and social cue interpretation
4. **Response**: User gets multiple options + explanation of social dynamics
5. **TTS**: Suggested response is converted to audio via ElevenLabs

### Gemini Response Generation

gemini_client.py uses strict system prompts enforcing:
- Single sentence responses (≤18 words)
- Concrete techniques only (breathing counts, sensory grounding)
- No metaphors, no "close your eyes"
- State-specific guidance

conversation_assistant.py prompts for structured output:
- SUGGESTED RESPONSE: [main suggestion]
- ALTERNATIVES: [2 options]
- TONE: [recommended tone]
- SOCIAL CUES: [interpretation]

### Safety Guardrails

- Crisis keyword detection triggers escalation (detector.py:13-22)
- Post-LLM filtering: max 18 words for distress, no medical claims (safety.py)
- Session timeout: 30 minutes inactivity
- Escalation after 120 seconds persistent high distress
- No audio/video/image storage (processed then discarded)

### Multi-Step Interventions

Grounding sequences (interventions.py:41-48) track per-chat progress:
1. "Name five things you can see"
2. "Name four things you can touch"
3. ... (continues through all senses)

Breathing cues also use step tracking for progressive guidance.

## Modifying the System

### Add New Intervention

1. Add to `InterventionType` enum in models.py
2. Map state → intervention in interventions.py `INTERVENTIONS` dict
3. Add LLM guidance in gemini_client.py `state_guidance`

### Adjust Detection Sensitivity

Edit `state_machine.py` THRESHOLDS:
```python
THRESHOLDS = {
    DistressState.PANIC: (0.6, 1.0),  # Increase 0.6 to make less sensitive
    # ...
}
```

Or adjust TEXT_WEIGHT in .env (default 0.6).

### Add Crisis Keywords

Edit detector.py CRISIS_KEYWORDS list:
```python
CRISIS_KEYWORDS = [
    r"\bhurt\s+myself\b",
    # Add new patterns here
]
```

### Customize Conversation Suggestions

Edit conversation_assistant.py `ConversationCoach._build_conversation_prompt()`:
- Modify system instructions
- Add new context types
- Change response format

### Adjust Facial Analysis

Edit conversation_assistant.py `FacialAnalyzer`:
- Modify emotion categories
- Change confidence extraction logic
- Update facial cue patterns

### Change Response Length

Modify config.py:
```python
max_response_words: int = 18  # Change this value
```

## Meta Glasses API Integration

This service implements the Meta Glasses API provider interface:

**Provider Configuration**:
- Distress helper URL: `http://localhost:8000/distress/infer`
- Check-in URL: `http://localhost:8000/distress/checkin`
- Conversation assistant URL: `http://localhost:8000/conversation/assist`

The extension forwards messages and images from Messenger to these endpoints and plays back TTS audio.

## Privacy & Security

- Text and images sent to Google Gemini API for processing
- Generated responses sent to ElevenLabs for TTS (optional)
- No message history or image storage
- Sessions expire after 30 minutes
- Temporary audio files auto-cleaned after 1 hour (main.py:29)

## Use Cases

### 1. Panic Attack Support
User experiencing panic attack speaks to glasses → System detects distress → Provides breathing guidance via voice

### 2. Social Anxiety at Events
User at networking event → Person approaches → Glasses capture face and speech → System suggests appropriate response + explains their emotions

### 3. Autism Spectrum Support
User struggles with facial expressions → System analyzes other person's face → Explains emotions and suggests contextually appropriate responses

### 4. Interview Preparation
User nervous about interviews → Practice conversations → Get real-time feedback on appropriate responses

## Common Issues

**Gemini API key not set**: Add GEMINI_API_KEY to .env file
**No voice output**: ElevenLabs API key not set (service works without it)
**Facial analysis fails**: Ensure image provided as frame_url or frame_base64, verify image format (JPEG/PNG)
**State too sensitive**: Increase TEXT_WEIGHT in .env or adjust THRESHOLDS in state_machine.py
**Glasses not responding**: Verify extension provider URLs match service address

## Related Documentation

- README.md: User-facing documentation and full setup guide
- QUICKSTART.md: 5-minute setup guide
- DEMO_SCRIPT.md: Demo scenarios and testing
- TEST_INSTRUCTIONS.md: Testing procedures

## Technical Architecture Summary

**Distress Management Flow**:
1. User message → Detector → State Machine → Intervention Manager → Gemini Client → Safety Filter → TTS → Response

**Conversation Assistant Flow**:
1. User input (what they said + face image) → Facial Analyzer (Gemini Vision) → Conversation Coach (Gemini) → Response with suggestions + alternatives + emotion analysis → TTS → Response

**Key Design Decisions**:
- Gemini-only (removed Ollama to simplify deployment and leverage multimodal capabilities)
- Stateless sessions (no database, all in-memory)
- Dual-mode system (distress + conversation) sharing same infrastructure
- Safety-first with multiple layers of filtering and escalation
- Privacy-focused with no persistent storage
