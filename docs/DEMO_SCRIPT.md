# EmpathLens Demo Script

**Duration**: 60-90 seconds  
**Audience**: Judges, stakeholders, potential users

## Setup (Before Demo)

1. ✅ Ollama running: `ollama serve`
2. ✅ EmpathLens running: `python main.py`
3. ✅ Meta Glasses paired to phone
4. ✅ Messenger extension monitoring the group chat
5. ✅ Audio routing configured (if demonstrating voice)

## Demo Flow

### Opening (10 seconds)

**Say**:
> "EmpathLens helps you through anxiety and panic using your Meta Glasses. Watch as it detects distress and guides me through a brief intervention."

**Show**: Service running at `http://localhost:8000/health`

---

### Scenario 1: Panic Attack (30 seconds)

**Do**: Put on glasses or open Messenger chat

**Say** (into glasses or type):
> "I'm having a panic attack, I can't breathe"

**Expected Response** (~2 seconds):
> "In for four, hold seven, out for eight."

**Do**: Follow the breathing for ~15 seconds

**System Says** (~20 seconds later):
> "How are you feeling now?"  
> *[Buttons: better | same | worse]*

**Click**: "better"

**System Says**:
> "Your body is settling. Take two slow breaths."

**Explain**:
> "The system detected panic, delivered 4-7-8 breathing, checked in, and adapted based on my response."

---

### Scenario 2: Crisis Detection (20 seconds)

**Say**:
> "Now watch what happens with crisis keywords."

**Type/Say**:
> "I don't want to live anymore"

**Expected Response** (immediate):
> "I hear you're struggling. Would you like me to contact your support person?"  
> *[Buttons: contact_support | continue_alone]*

**Explain**:
> "Crisis keywords trigger immediate escalation - no standard coaching, straight to human support."

---

### Scenario 3: Stop Command (15 seconds)

**Say**:
> "Users have full control. Watch."

**Type/Say**:
> "stop"

**Expected Response**:
> "Understood. I'm here if you need me."

**Explain**:
> "The stop command immediately ends intervention. User autonomy is paramount."

---

### Closing (15 seconds)

**Say**:
> "EmpathLens runs locally for privacy, uses Ollama for coaching, and delivers help in under 2 seconds. It's not a replacement for therapy - it's a companion for brief distress episodes."

**Show** (optional): Architecture diagram or health check output

**End with**:
> "Questions?"

---

## Backup Scenarios

If live demo fails, have these ready:

### Pre-recorded Video
- Show the full flow working
- Point to specific moments (detection, response, check-in)

### Curl Commands
```bash
# Panic scenario
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{"message": "I'\''m having a panic attack", "chat_id": "demo_123"}'

# Crisis scenario
curl -X POST http://localhost:8000/distress/infer \
  -H "Content-Type: application/json" \
  -d '{"message": "I don'\''t want to live", "chat_id": "demo_456"}'
```

### Static Slides
1. Problem: Distress episodes happen anywhere
2. Solution: AI-powered micro-interventions via glasses
3. Architecture: Local + private + fast
4. Safety: Crisis detection, guardrails, stop command
5. Results: Sub-2-second response, evidence-based techniques

---

## Key Points to Emphasize

✅ **Speed**: Responses in 1-3 seconds  
✅ **Safety**: Crisis detection, post-filters, escalation  
✅ **Privacy**: Runs locally, no data storage  
✅ **Adaptability**: Checks in and adjusts based on feedback  
✅ **Control**: Stop command, user autonomy  
✅ **Evidence-based**: 4-7-8 breathing, 5-4-3-2-1 grounding  

## Anticipated Questions

**Q: Is this a medical device?**  
A: No. It's a supportive tool, not a replacement for professional care. We display clear disclaimers.

**Q: What if someone is in real danger?**  
A: Crisis keywords trigger immediate escalation offers. We don't attempt coaching - we route to human support.

**Q: How accurate is the distress detection?**  
A: Text-based (0.6 weight) is primary. Users explicitly describe their state. Audio prosody (0.4 weight) is optional and secondary.

**Q: What if Ollama or ElevenLabs fail?**  
A: Fallback responses are built-in. Service degrades gracefully to text-only safe defaults.

**Q: Privacy concerns?**  
A: Everything runs locally except optional TTS. No message history, no analytics, no storage. Sessions expire after 30 minutes.

**Q: Can this handle group calls?**  
A: Currently designed for 1:1 sessions. Group support is future work.

## Technical Specifications (If Asked)

- **Latency**: 1-3 seconds (LLM + TTS)
- **Stack**: FastAPI, Ollama (llama3), ElevenLabs
- **Detection**: Regex + keyword matching (text), optional prosody (audio)
- **Interventions**: 5 states, 6 technique types
- **Safety**: Post-LLM filters, 18-word limit, crisis escalation
- **Privacy**: Ephemeral state only, no persistent storage

## Post-Demo

Provide:
- GitHub repo link
- README with setup instructions
- Contact info for follow-up

---

**Good luck! Remember: clarity, safety, and impact.**

