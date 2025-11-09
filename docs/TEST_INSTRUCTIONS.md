# Testing EmpathLens - Step by Step

## ⚡ Quick Test (Without Meta Glasses)

You can test the system entirely from your terminal without needing the Meta Glasses or browser extension.

### Step 1: Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com
```

### Step 2: Start Ollama & Pull Model

Open a terminal and run:
```bash
# Start Ollama server (keep this running)
ollama serve

# In a NEW terminal, pull the model
ollama pull llama3
```

This might take a few minutes (the model is ~4GB).

### Step 3: Install Python Dependencies

```bash
cd /Users/krrishts/Documents/MakeUC2025

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Create .env File

```bash
# Copy the example and edit
cat > .env << 'EOF'
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
ELEVEN_API_KEY=
TEXT_WEIGHT=0.6
AUDIO_WEIGHT=0.4
ENABLE_AUDIO=false
HOST=0.0.0.0
PORT=8000
EOF
```

**Note**: Leave `ELEVEN_API_KEY` empty for now - the system works in text-only mode!

### Step 5: Start EmpathLens

```bash
# With virtual environment activated
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Test It! 

Open a **NEW terminal** and run:

```bash
# Activate venv
cd /Users/krrishts/Documents/MakeUC2025
source venv/bin/activate

# Run automated tests
python test_service.py
```

This will test all scenarios:
- ✅ Health check
- ✅ Panic attack detection
- ✅ Overwhelmed detection
- ✅ Crisis keyword detection
- ✅ Stop command

### Step 7: Manual Testing (Fun Part!)

Try these curl commands to see the system in action:

#### Test 1: Panic Attack
```bash
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I am having a panic attack, I cannot breathe",
    "chat_id": "demo_panic"
  }' | jq
```

**Expected**: You'll get a 4-7-8 breathing instruction in ~2 seconds.

#### Test 2: Feeling Overwhelmed
```bash
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Everything is too much, I am shutting down",
    "chat_id": "demo_overwhelm"
  }' | jq
```

**Expected**: You'll get a grounding technique (5-4-3-2-1).

#### Test 3: Crisis Keywords
```bash
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I do not want to live anymore",
    "chat_id": "demo_crisis"
  }' | jq
```

**Expected**: Immediate escalation offer (no coaching, straight to support).

#### Test 4: Check-in (Better)
```bash
curl -X POST http://localhost:8000/distress/checkin \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "demo_panic",
    "response": "better"
  }' | jq
```

**Expected**: Recovery/reinforcement message.

#### Test 5: Stop Command
```bash
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "stop",
    "chat_id": "demo_panic"
  }' | jq
```

**Expected**: "Understood. I'm here if you need me."

---

## 🎙️ Testing with Meta Glasses (Advanced)

If you have the Meta Glasses and want to test the full integration:

### 1. Set Up Meta Glasses Extension

Follow the guide at: https://github.com/dcrebbin/meta-glasses-api

Key steps:
1. Install the browser extension
2. Create a Messenger group chat with an alt account
3. Configure the extension to monitor that chat

### 2. Add EmpathLens as a Custom Provider

In the extension settings:
- **Provider Name**: EmpathLens
- **Infer Endpoint**: `http://localhost:8000/distress/infer`
- **Check-in Endpoint**: `http://localhost:8000/distress/checkin`

### 3. (Optional) Add ElevenLabs Voice

Get an API key from https://elevenlabs.io (free tier available).

Edit `.env`:
```bash
ELEVEN_API_KEY=your_actual_api_key_here
```

Restart the service.

### 4. Test Voice Commands

In your Messenger chat, say or type:
- "I'm having a panic attack"
- Wait for response
- Say "better" when prompted

You should hear the coaching through the glasses!

---

## 🐛 Troubleshooting

### "Connection refused" to localhost:8000
→ Make sure `python main.py` is running in one terminal

### "Connection refused" to Ollama
→ Make sure `ollama serve` is running in another terminal
→ Test with: `curl http://localhost:11434/api/tags`

### `jq: command not found`
→ Remove `| jq` from curl commands, or install jq: `brew install jq`

### No voice output
→ This is expected if ELEVEN_API_KEY is not set
→ The system works fine in text-only mode
→ Add your ElevenLabs key to enable voice

### Tests are slow
→ First LLM call might take 5-10 seconds (loading model)
→ Subsequent calls should be 1-3 seconds

### "Model not found"
→ Run: `ollama pull llama3`
→ Make sure it completes (downloads ~4GB)

---

## 🎯 What to Look For

When testing, observe:

✅ **Speed**: Responses should arrive in 1-3 seconds  
✅ **State Detection**: Panic keywords → panic state  
✅ **Interventions**: Appropriate technique for each state  
✅ **Safety**: Crisis keywords trigger escalation immediately  
✅ **Control**: Stop command ends intervention  
✅ **Adaptation**: Check-ins adjust based on your response  
✅ **Word Limit**: All responses ≤18 words  

---

## 📊 Understanding the Response

Example response:
```json
{
  "reply_text": "In for four, hold seven, out for eight.",
  "expect_followup": true,
  "followup_after_sec": 60,
  "buttons": ["better", "same", "worse"],
  "audio_url": null,
  "meta": {
    "state": "panic",
    "confidence": 0.85,
    "intervention_type": "four_seven_eight",
    "session_duration_seconds": 5
  }
}
```

- `reply_text`: What to speak/display
- `expect_followup`: Should check in later?
- `followup_after_sec`: How long to wait
- `buttons`: Quick reply options
- `meta.state`: Current distress state
- `meta.confidence`: Detection confidence (0-1)

---

## 🎬 Ready for Demo?

Once everything works:
1. Review `DEMO_SCRIPT.md` for the 60-90 second demo
2. Practice the panic → check-in → better flow
3. Show the crisis detection
4. Demonstrate the stop command

**Good luck!** 🚀

