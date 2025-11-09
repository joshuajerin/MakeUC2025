# ARM Edge Quick Start Guide

Get EmpathLens ARM edge service running in under 15 minutes.

## Prerequisites

- ARM device: Raspberry Pi 4/5, NVIDIA Jetson, or ARM64 Linux device
- MicroSD card (for RPi) or storage
- Network connection
- Meta Ray-Ban Smart Glasses (or test client)

## Quick Setup

### Step 1: Flash OS (Raspberry Pi)

```bash
# Download Raspberry Pi OS Lite (64-bit)
# Flash to SD card using Raspberry Pi Imager
# Enable SSH in imager settings
# Boot Raspberry Pi
```

### Step 2: Initial System Setup

```bash
# SSH into device
ssh pi@raspberrypi.local

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install git
sudo apt-get install -y git

# Clone repository
git clone https://github.com/yourusername/empatlens.git
cd empatlens
```

### Step 3: Run ARM Deployment Script

```bash
# Make script executable
chmod +x arm_edge/deployment/deploy_arm.sh

# Run deployment (takes 5-10 minutes)
./arm_edge/deployment/deploy_arm.sh

# Follow prompts - script will:
# ✓ Detect ARM device type
# ✓ Install system dependencies
# ✓ Create Python virtual environment
# ✓ Install TensorFlow Lite + dependencies
# ✓ Create and quantize demo model
# ✓ Configure environment
# ✓ Run tests
```

### Step 4: Start ARM Edge Service

```bash
# Activate virtual environment
source venv_arm/bin/activate

# Start service
python -m uvicorn arm_edge.arm_server:app --host 0.0.0.0 --port 8001

# Service runs at http://raspberrypi.local:8001
```

### Step 5: Verify Installation

```bash
# In another terminal, test the service:

# Health check
curl http://localhost:8001/health

# Should show ARM optimizations enabled:
{
  "status": "healthy",
  "arm_optimizations": {
    "neon_simd": true,
    "arm_nn_delegate": true,
    "quantized_int8": true
  },
  "device_info": {
    "device_type": "Raspberry Pi"
  }
}

# Run benchmarks
curl http://localhost:8001/arm/benchmark

# Should show 4-6x speedup with ARM optimizations
```

## Performance Verification

### Test Facial Emotion Detection

```bash
# Create test image
python3 << EOF
import numpy as np
from PIL import Image

# Create random test face image
img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
Image.fromarray(img).save('/tmp/test_face.jpg')
print("Test image created: /tmp/test_face.jpg")
EOF

# Test facial detection endpoint
curl -X POST http://localhost:8001/edge/process \
  -F "chat_id=test_001" \
  -F "face_image=@/tmp/test_face.jpg" \
  -F "conversation_mode=true"

# Check response time (should be 50-100ms)
```

### Test Audio Processing

```bash
# Create test audio
python3 << EOF
import numpy as np
import wave

# Generate 3 seconds of test audio
sample_rate = 16000
duration = 3
t = np.linspace(0, duration, duration * sample_rate)
audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
audio = (audio * 32767).astype(np.int16)

# Save as WAV
with wave.open('/tmp/test_audio.wav', 'w') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(audio.tobytes())
print("Test audio created: /tmp/test_audio.wav")
EOF

# Test audio processing
curl -X POST http://localhost:8001/edge/process \
  -F "chat_id=test_002" \
  -F "user_message=test" \
  -F "audio_data=@/tmp/test_audio.wav"

# Check response time (should be 15-30ms)
```

## Integration with Cloud Backend

### Step 1: Start Cloud Backend

```bash
# In a separate terminal (can be same device or different server)
cd empatlens
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 2: Configure ARM Edge to Use Cloud

```bash
# Edit ARM edge configuration
nano arm_edge/.env

# Set cloud backend URL:
CLOUD_BACKEND_URL=http://localhost:8000  # or remote server IP

# Set privacy mode (controls what data is sent to cloud):
PRIVACY_MODE=balanced  # options: strict, balanced, minimal

# Restart ARM edge service
```

### Step 3: Test End-to-End Flow

```bash
# Send request to ARM edge (it will forward to cloud as needed)
curl -X POST http://localhost:8001/edge/process \
  -F "chat_id=test_e2e" \
  -F "user_message=I'm feeling anxious" \
  -F "face_image=@/tmp/test_face.jpg"

# ARM edge will:
# 1. Process face locally (50-80ms)
# 2. Process text locally
# 3. Decide whether to send to cloud (based on privacy mode)
# 4. If needed, forward metadata to cloud
# 5. Return response

# Check logs to see privacy decisions made
```

## Privacy Mode Configuration

### Strict Mode (Maximum Privacy)

```bash
# Edit .env
PRIVACY_MODE=strict
OFFLINE_MODE=false

# Behavior:
# - Face images: NEVER sent to cloud
# - Audio: NEVER sent to cloud
# - Only metadata (emotions, features) transmitted
# - Best for: Healthcare, therapy, privacy-critical use cases
```

### Balanced Mode (Recommended)

```bash
# Edit .env
PRIVACY_MODE=balanced
OFFLINE_MODE=false

# Behavior:
# - Face images: Sent only if local confidence < 70%
# - Audio: Sent only if distress level uncertain (30-70%)
# - 70-80% of data stays local
# - Best for: General use, social anxiety support
```

### Offline Mode

```bash
# Edit .env
PRIVACY_MODE=strict
OFFLINE_MODE=true

# Behavior:
# - NO cloud communication
# - All processing local on ARM
# - Basic distress support offline
# - Best for: No internet, maximum privacy
```

## Production Setup (Systemd Service)

For production deployment, run ARM edge service as systemd service:

```bash
# Create service file
sudo nano /etc/systemd/system/empatlens-arm.service

# Add configuration:
[Unit]
Description=EmpathLens ARM Edge Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/empatlens
Environment="PATH=/home/pi/empatlens/venv_arm/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/pi/empatlens/venv_arm/bin/uvicorn arm_edge.arm_server:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Save and exit (Ctrl+X, Y, Enter)

# Enable and start service
sudo systemctl enable empatlens-arm.service
sudo systemctl start empatlens-arm.service

# Check status
sudo systemctl status empatlens-arm.service

# View logs
sudo journalctl -u empatlens-arm.service -f
```

## Troubleshooting

### Issue: TensorFlow Lite not found

```bash
# Solution: Install tflite-runtime
pip install tflite-runtime

# Or install full TensorFlow (larger but more compatible)
pip install tensorflow
```

### Issue: ARM NN delegate not loading

```bash
# This is optional - service works without it (just slower)
# Check logs:
curl http://localhost:8001/health

# If arm_nn_enabled: false, service still works but without ARM NN acceleration
# To install ARM NN (advanced):
# Visit: https://github.com/ARM-software/armnn
```

### Issue: Low performance / high latency

```bash
# Check CPU governor (should be 'performance')
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance mode
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# On Jetson, use max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Issue: Model file not found

```bash
# Regenerate model
cd empatlens
source venv_arm/bin/activate
python arm_edge/optimizations/quantize_models.py

# This creates demo model and quantizes it
# Output: arm_edge/models/emotion_quantized_armnn.tflite
```

## Performance Tuning

### Optimize for Maximum Speed

```bash
# Edit arm_edge/.env
ARM_THREADS=4  # Use all CPU cores
ENABLE_ARM_NN=true
ENABLE_NEON=true

# Set CPU to performance mode
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    echo "performance" | sudo tee $cpu/cpufreq/scaling_governor
done
```

### Optimize for Power Efficiency

```bash
# Edit arm_edge/.env
ARM_THREADS=2  # Use fewer cores
ENABLE_ARM_NN=true  # Still enable for efficiency
ENABLE_NEON=true

# Set CPU to powersave mode
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    echo "powersave" | sudo tee $cpu/cpufreq/scaling_governor
done
```

## Next Steps

1. **Connect Meta Glasses**: Configure glasses to send requests to `http://raspberrypi.local:8001/edge/process`

2. **Test with Real Data**: Use glasses to capture real facial expressions and audio

3. **Monitor Performance**: Check `/arm/performance` endpoint to see ARM optimization impact

4. **Adjust Privacy Settings**: Tune `PRIVACY_MODE` based on your use case

5. **Scale**: Deploy multiple ARM edge devices for multiple users

## Resources

- Full ARM architecture documentation: `docs/ARM_ARCHITECTURE.md`
- Model quantization guide: `arm_edge/optimizations/quantize_models.py`
- Deployment script: `arm_edge/deployment/deploy_arm.sh`
- Main project README: `README.md`

## Support

- GitHub Issues: Report ARM-specific issues with `[ARM]` tag
- Check logs: `sudo journalctl -u empatlens-arm.service -f`
- Performance benchmarks: `curl http://localhost:8001/arm/benchmark`

## Summary

You now have:
- ✓ ARM edge service running on local device
- ✓ Privacy-preserving facial emotion detection (local processing)
- ✓ ARM NEON-optimized audio processing (4-5x faster)
- ✓ Integration with cloud backend (optional)
- ✓ Offline capability for anywhere access

**Total setup time**: ~15 minutes

**Performance improvement**: 4-6x faster than cloud-only approach

**Privacy**: Sensitive data stays on your local ARM device
