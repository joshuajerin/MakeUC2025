# EmpathLens Quick Start

Get up and running in 5 minutes with your AI social companion.

## Prerequisites

- Python 3.9+
- Meta Glasses (or Messenger app for testing)
- Google Gemini API key (free)

## Installation

### 1. Get Your Gemini API Key

```bash
# Go to Google AI Studio and get your free API key
# https://makersuite.google.com/app/apikey
```

### 2. Set Up EmpathLens

```bash
cd /path/to/MakeUC2025

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create config file
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
ELEVEN_API_KEY=
TEXT_WEIGHT=0.6
AUDIO_WEIGHT=0.4
ENABLE_AUDIO=false
HOST=0.0.0.0
PORT=8000
EOF
```

### 3. (Optional) Add ElevenLabs API Key for Voice

Get key from [elevenlabs.io](https://elevenlabs.io), then:
```bash
# Edit .env and add:
ELEVEN_API_KEY=your_key_here
```

### 4. Start the Service

```bash
# Easy way
./scripts/start.sh

# Or manually
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Test It

### Terminal Test
```bash
# In a new terminal
python tests/test_service.py
```

### Test Distress Detection
```bash
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I'\''m having a panic attack",
    "chat_id": "test_123"
  }'
```

### Test Conversation Assistant (NEW!)
```bash
curl -X POST http://localhost:8000/conversation/assist \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "test_123",
    "other_person_said": "Hey, how are you doing today?",
    "conversation_context": "Casual greeting from a friend"
  }'
```

## Connect Meta Glasses Extension

1. Install [Meta Glasses API extension](https://github.com/dcrebbin/meta-glasses-api)
2. In extension settings, add custom provider:
   - Name: EmpathLens
   - **Distress Infer URL**: `http://localhost:8000/distress/infer`
   - **Check-in URL**: `http://localhost:8000/distress/checkin`
   - **Conversation URL**: `http://localhost:8000/conversation/assist`
3. Create Messenger group chat, monitor it
4. Start talking!

## Quick Commands

### Distress Management Mode

**Test panic detection:**
Say: "I'm having a panic attack"

**Test grounding:**
Say: "I feel overwhelmed and shut down"

**Test crisis handling:**
Say: "I don't want to live anymore"

**Stop intervention:**
Say: "stop"

### Conversation Assistant Mode (NEW!)

**Use during conversations:**
1. Talk to someone while wearing your glasses
2. Glasses capture their face and what they say
3. Get real-time suggestions on what to say back
4. Choose from multiple response options
5. Hear tone guidance and social cue explanations

**Perfect for:**
- Social anxiety at events
- Dating conversations
- Job interviews
- Making new friends
- Understanding emotions

## Features

### 🚨 Panic Attack Support
- Detects distress from your voice
- Guides breathing exercises
- Provides grounding techniques
- Escalates to crisis support if needed

### 💬 Social Conversation Assistant
- Analyzes facial expressions of the person you're talking to
- Suggests what to say back in real-time
- Provides multiple response options
- Explains social cues and emotions
- Helps with tone and context

## Troubleshooting

**Service won't start:**
- Check you've added your Gemini API key to `.env`
- Verify port 8000 is free: `lsof -i :8000`

**No voice output:**
- ElevenLabs API key not set (service works without it)
- Check `.env` file has correct key

**Glasses not responding:**
- Verify extension is monitoring correct chat
- Check provider URLs match `http://localhost:8000/...`
- Test API directly with curl first

**Facial analysis not working:**
- Ensure image is being sent from glasses
- Check Gemini API has vision access
- Verify image format (JPEG/PNG)

## Next Steps

- Read full [README.md](README.md) for details
- Review [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for demo prep
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to help improve EmpathLens

## Use Cases

1. **Panic attacks in public** - Get discreet breathing guidance
2. **Social anxiety at events** - Real-time conversation help
3. **Autism spectrum support** - Understand facial expressions and social cues
4. **Interview preparation** - Practice appropriate responses
5. **Dating conversations** - Navigate emotional dynamics

## Support

- 🇺🇸 Crisis: 988 Suicide & Crisis Lifeline
- 🇺🇸 Text: HOME to 741741
- 🌍 International: [findahelpline.com](https://findahelpline.com)

---

**Ready in 5 minutes. Privacy-focused. Powered by Google Gemini.**
