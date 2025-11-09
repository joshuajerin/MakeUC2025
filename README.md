# EmpathLens – AI Social Companion for Meta Ray-Ban Glasses

**Your personal AI therapist and conversation coach, right in your glasses.**

⚠️ **DISCLAIMER**: This is NOT a medical device. This tool provides distress management techniques and social conversation assistance but is not a substitute for professional mental health care, emergency services, or medical advice.

## Overview

EmpathLens integrates with your Meta Ray-Ban Smart Glasses to provide two powerful features:

### 1. Panic Attack & Distress Management
Real-time, voice-based support during anxiety, panic, or overwhelming moments:
- **Detects distress** from your voice messages
- **Guides breathing exercises** with calming voice instructions
- **Provides grounding techniques** to help you regain control
- **Adapts in real-time** based on how you're feeling
- **Escalates to crisis support** when needed

### 2. Social Conversation Assistant (NEW!)
For people who struggle with social cues or conversation anxiety:
- **Analyzes facial expressions** of the person you're talking to
- **Understands emotional context** from their face and words
- **Suggests what to say back** in real-time
- **Provides alternative responses** so you can choose what feels right
- **Explains social cues** to help you understand what they might be feeling

Perfect for people with social anxiety, autism spectrum, or anyone who wants support navigating conversations.

## Core Features

### Distress Management States

| State | Intervention | Duration | Example |
|-------|-------------|----------|---------|
| **RISING** | Paced breathing (4-4) | ~30s | "Breathe in for four, out for four." |
| **PANIC** | 4-7-8 breathing | ~60s | "In for four, hold seven, out for eight." |
| **OVERWHELMED** | 5-4-3-2-1 grounding | ~75s | "Name five things you can see." |
| **RECOVERY** | Reinforcement | — | "Your body is settling. Two slow breaths." |
| **CRISIS_RISK** | Escalation offer | — | "Can I contact your support person?" |

### Conversation Assistant Features

- **Facial emotion detection**: Analyzes happiness, sadness, anger, surprise, confusion, interest, etc.
- **Context-aware suggestions**: Considers what they said AND their facial expression
- **Multiple response options**: Get 2-3 alternatives to choose from
- **Tone guidance**: Tells you the recommended tone (friendly, empathetic, casual, etc.)
- **Social cue interpretation**: Explains what they might be feeling or wanting
- **Voice playback**: Hear the suggested response via ElevenLabs TTS

### Safety Guardrails

- ✅ Crisis keyword detection (self-harm, suicide ideation) triggers immediate escalation
- ✅ Post-AI filtering: concise responses, no medical claims
- ✅ "Stop" command immediately ends intervention
- ✅ Escalation after 2 minutes of persistent distress
- ✅ No raw audio/video storage
- ✅ Runs locally with Gemini API

## Architecture

```
User (Meta Glasses)
    ↓ (voice/text + optional face image)
Messenger Browser Extension
    ↓ (HTTP POST)
EmpathLens Provider (FastAPI) ←→ Google Gemini AI
    ↓                              ↓
ElevenLabs TTS ←→ Virtual Audio Device → Messenger Call
```

## Requirements

### Hardware
- Meta Ray-Ban Smart Glasses (or standalone Messenger app for testing)
- Computer running macOS/Linux/Windows

### Software
1. **Python 3.9+**
2. **Google Gemini API key** (free tier available)
3. **Meta Glasses API** - Messenger browser extension
4. **ElevenLabs API key** (optional, for voice output)
5. **Virtual audio device** (optional, for routing TTS into calls)
   - macOS: [BlackHole](https://github.com/ExistentialAudio/BlackHole)
   - Windows: VB-Audio Cable
   - Linux: PulseAudio loopback

## Installation

### 1. Get API Keys

#### Google Gemini API (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

#### ElevenLabs API (Optional, for voice)
1. Sign up at [elevenlabs.io](https://elevenlabs.io)
2. Go to Profile → API Keys
3. Copy your API key

### 2. Set Up EmpathLens

```bash
# Clone or navigate to project directory
cd /path/to/MakeUC2025

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
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

### 3. Set Up Meta Glasses API Extension

Follow the setup instructions at [github.com/dcrebbin/meta-glasses-api](https://github.com/dcrebbin/meta-glasses-api):

1. Install the browser extension
2. Create a Messenger group chat with your alt account
3. Configure the extension to use EmpathLens as a custom provider:
   - **Distress helper URL**: `http://localhost:8000/distress/infer`
   - **Check-in URL**: `http://localhost:8000/distress/checkin`
   - **Conversation assistant URL**: `http://localhost:8000/conversation/assist`

### 4. (Optional) Set Up Virtual Audio Routing

To route TTS audio into Messenger calls:

#### macOS with BlackHole:
```bash
# Install BlackHole
brew install blackhole-2ch

# Configure Multi-Output Device in Audio MIDI Setup
# 1. Open Audio MIDI Setup (/Applications/Utilities)
# 2. Click '+' → Create Multi-Output Device
# 3. Check: Built-in Output + BlackHole 2ch
# 4. Set as system output
# 5. In Messenger call settings, select BlackHole as microphone input
```

## Usage

### Starting the Service

```bash
# Easy way: Use the start script
./scripts/start.sh

# Or manually:
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will start on `http://localhost:8000`.

### Testing the API

#### Test Distress Detection
```bash
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I'\''m having a panic attack",
    "chat_id": "test_123"
  }'
```

#### Test Conversation Assistant
```bash
curl -X POST http://localhost:8000/conversation/assist \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "test_123",
    "other_person_said": "Hey, how are you doing today?",
    "conversation_context": "Casual greeting from a friend"
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

### Using with Meta Glasses

#### Distress Management Mode
1. **Join a Messenger call** or send a message
2. **Say**: "I'm having a panic attack"
3. **Listen** to the breathing guidance (arrives in ~2 seconds)
4. **Follow the breathing cue** for ~45-60 seconds
5. **Respond** with "better", "same", or "worse" when prompted
6. **System adapts** based on your response

#### Conversation Assistant Mode
1. **Join a Messenger call** with someone
2. The glasses capture the other person's face and what they say
3. **You receive** real-time suggestions on what to say back
4. **Choose** from the suggested responses or alternatives
5. **Speak naturally** using the guidance

### Stop Command

Say or type **"stop"** at any time to end any intervention immediately.

## API Reference

### POST `/distress/infer`

Main endpoint for distress detection and intervention.

**Request:**
```json
{
  "message": "I'm having a panic attack",
  "chat_id": "unique_chat_id",
  "user_id": "optional_user_id",
  "timestamp": "2025-11-08T12:00:00Z",
  "frame_url": "optional_image_url"
}
```

**Response:**
```json
{
  "reply_text": "In for four, hold seven, out for eight.",
  "expect_followup": true,
  "followup_after_sec": 60,
  "buttons": ["better", "same", "worse"],
  "audio_url": "/tmp/empathlens_audio/response_123.mp3",
  "meta": {
    "state": "panic",
    "confidence": 0.85,
    "intervention_type": "four_seven_eight",
    "session_duration_seconds": 45
  }
}
```

### POST `/conversation/assist`

Conversation assistant endpoint for social interaction help.

**Request:**
```json
{
  "chat_id": "unique_chat_id",
  "other_person_said": "That's really interesting! Tell me more.",
  "frame_url": "optional_image_url",
  "frame_base64": "optional_base64_image",
  "conversation_context": "Discussing my hobby"
}
```

**Response:**
```json
{
  "suggested_response": "Well, I've been really into photography for about a year now, especially landscape shots.",
  "alternative_responses": [
    "I'd love to share more about it! What would you like to know?",
    "It's something I'm passionate about. I could show you some examples if you'd like."
  ],
  "emotion_analysis": {
    "primary_emotion": "interested",
    "confidence": 0.85,
    "secondary_emotions": ["happy", "engaged"],
    "facial_cues": "slight smile, raised eyebrows"
  },
  "tone_guidance": "enthusiastic but not overwhelming",
  "social_cues": "They seem genuinely interested and want to hear more details. Their raised eyebrows and smile indicate positive engagement.",
  "audio_url": "/tmp/empathlens_audio/conversation_123.mp3",
  "confidence": 0.85
}
```

### POST `/distress/checkin`

Handle follow-up check-ins.

**Request:**
```json
{
  "chat_id": "unique_chat_id",
  "response": "better",
  "timestamp": "2025-11-08T12:01:00Z"
}
```

### POST `/distress/stop`

Stop intervention for a conversation.

**Request:**
```bash
?chat_id=unique_chat_id
```

### GET `/health`

Service health check.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "detector": "operational",
    "state_machine": "operational",
    "llm": "operational",
    "llm_provider": "gemini",
    "facial_analyzer": "operational",
    "conversation_coach": "operational",
    "tts": "operational"
  },
  "active_sessions": 3
}
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | `None` | Google Gemini API key (required) |
| `GEMINI_MODEL` | `gemini-2.0-flash-exp` | Gemini model to use |
| `ELEVEN_API_KEY` | `None` | ElevenLabs API key (optional) |
| `ELEVEN_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | Voice ID for TTS |
| `TEXT_WEIGHT` | `0.6` | Weight for text-based detection |
| `AUDIO_WEIGHT` | `0.4` | Weight for audio prosody (if enabled) |
| `ENABLE_AUDIO` | `false` | Enable audio prosody analysis |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### Session Configuration

- **Session timeout**: 30 minutes of inactivity
- **Intervention cooldown**: 30 seconds between interventions
- **Escalation timeout**: 120 seconds of persistent distress
- **Max response words**: 18 words per response (distress mode)

## Troubleshooting

### "Gemini API key not set"
- Ensure you've added your API key to `.env`
- Get a free key at [Google AI Studio](https://makersuite.google.com/app/apikey)

### No voice output
- Check ElevenLabs API key is set correctly
- Verify API key is valid: [elevenlabs.io/app](https://elevenlabs.io/app)
- Service falls back to text-only if TTS unavailable

### Facial analysis not working
- Ensure image is provided as `frame_url` or `frame_base64`
- Check that Gemini API key has vision API access
- Verify image format is supported (JPEG, PNG)

### Glasses not responding
- Verify Messenger extension is monitoring the correct chat
- Check provider URLs in extension settings
- Test API directly with curl first

### State machine too sensitive
- Increase `TEXT_WEIGHT` in `.env` (e.g., `0.8`)
- Adjust thresholds in `state_machine.py`

## Privacy & Security

### What's Stored
- ✅ Ephemeral session state (current distress level, intervention history)
- ✅ No raw audio or video
- ✅ No message history (processed then discarded)
- ✅ Temporary audio files (auto-cleaned after 1 hour)
- ✅ No facial images stored (analyzed then deleted)

### What's Shared
- ❌ Text and images sent to Google Gemini API for processing
- ❌ Generated responses sent to ElevenLabs for TTS (optional)
- ❌ No analytics, telemetry, or third-party tracking
- ❌ All processing happens via API calls, no local data retention

### Consent
- User must explicitly enable the extension and join monitored chats
- "Stop" command immediately ends all processing
- Sessions expire after 30 minutes of inactivity

## Development

### Project Structure

```
MakeUC2025/
├── README.md                   # User documentation
├── QUICKSTART.md              # 5-minute setup guide
├── CLAUDE.md                  # Developer guide for AI assistants
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI server & endpoints
│   ├── config.py              # Configuration management
│   ├── models.py              # Pydantic data models
│   ├── core/                  # Core business logic
│   │   ├── detector.py        # Distress detection
│   │   ├── state_machine.py   # State transitions
│   │   ├── interventions.py   # Intervention strategies
│   │   ├── session_manager.py # Session tracking
│   │   └── safety.py          # Safety filters
│   └── services/              # External integrations
│       ├── gemini_client.py   # Gemini LLM
│       ├── conversation_assistant.py  # Facial analysis
│       └── tts_client.py      # ElevenLabs TTS
├── tests/                     # Test suite
│   ├── test_service.py
│   └── simple_test.py
├── scripts/                   # Utility scripts
│   └── start.sh
└── docs/                      # Additional documentation
    ├── CONTRIBUTING.md
    ├── DEMO_SCRIPT.md
    └── TEST_INSTRUCTIONS.md
```

### Running Tests

```bash
# Run comprehensive test suite
python tests/test_service.py

# Run simple test
python tests/simple_test.py

# Test with increased logging
LOG_LEVEL=DEBUG python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Extending the System

#### Add a new intervention:
1. Update `app/models.py`: Add `InterventionType`
2. Update `app/core/interventions.py`: Map state → intervention
3. Update `app/services/gemini_client.py`: Add prompt guidance

#### Add new crisis keywords:
1. Edit `app/core/detector.py`: Add patterns to `CRISIS_KEYWORDS`

#### Change detection thresholds:
1. Edit `app/core/state_machine.py`: Adjust `THRESHOLDS` dict

#### Customize conversation suggestions:
1. Edit `app/services/conversation_assistant.py`: Modify prompts in `ConversationCoach`

## Use Cases

### 1. Panic Attack Support
**Scenario**: You're in public and feel a panic attack coming
- Discreetly activate through your glasses
- Get immediate breathing guidance
- Follow along privately without drawing attention

### 2. Social Anxiety at Events
**Scenario**: You're at a networking event and someone approaches you
- Glasses capture their face and what they say
- Get real-time suggestions on how to respond
- Choose responses that match your style

### 3. Autism Spectrum Support
**Scenario**: You struggle to read facial expressions
- System analyzes their emotions for you
- Explains what their facial cues mean
- Suggests appropriate responses based on context

### 4. Interview Preparation
**Scenario**: You're nervous about conversations in interviews
- Practice with the system analyzing interactions
- Learn appropriate responses to questions
- Build confidence in social situations

## Limitations & Future Work

### Current Limitations
- Requires internet for Gemini API
- English language only
- No persistent user profiles or learning
- Single-user sessions (no group support)
- Facial analysis requires clear face images

### Future Enhancements
- [ ] Audio prosody analysis (pitch, pace, tremor)
- [ ] Multi-language support
- [ ] Adaptive learning from user preferences
- [ ] Integration with wearables (heart rate, HRV)
- [ ] Offline mode with local models
- [ ] Group conversation support
- [ ] Real-time conversation tracking
- [ ] Personalized response style learning

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- **Meta Glasses API** by [@dcrebbin](https://github.com/dcrebbin)
- **Google Gemini** - [ai.google.dev](https://ai.google.dev)
- **ElevenLabs** - [elevenlabs.io](https://elevenlabs.io)
- Therapeutic techniques inspired by CBT, DBT, and mindfulness practices

## Support & Contact

For issues, questions, or contributions:
- Open an issue on GitHub
- Built for MakeUC 2025 Hackathon

---

**Remember**: This tool supports but does not replace professional care. If you're in crisis:
- 🇺🇸 **988 Suicide & Crisis Lifeline**: Call/text 988
- 🇺🇸 **Crisis Text Line**: Text HOME to 741741
- 🌍 **International**: [findahelpline.com](https://findahelpline.com)
