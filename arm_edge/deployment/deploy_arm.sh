#!/bin/bash
#
# EmpathLens ARM Edge Deployment Script
#
# Deploys the ARM-optimized edge service on ARM devices:
# - Raspberry Pi 4/5
# - NVIDIA Jetson Nano/Xavier/Orin
# - Other ARM64 devices
#
# This script:
# 1. Detects ARM device type
# 2. Installs ARM-optimized dependencies
# 3. Sets up TensorFlow Lite with ARM NN
# 4. Configures performance settings
# 5. Starts the edge service

set -e

echo "=========================================="
echo "EmpathLens ARM Edge Deployment"
echo "=========================================="

# Detect ARM device type
detect_device() {
    echo -n "Detecting ARM device... "

    if [ -f /proc/cpuinfo ]; then
        if grep -q "Raspberry Pi" /proc/cpuinfo; then
            echo "Raspberry Pi detected"
            DEVICE_TYPE="raspberry_pi"
        elif grep -q "Jetson" /proc/cpuinfo || grep -q "NVIDIA" /proc/cpuinfo; then
            echo "NVIDIA Jetson detected"
            DEVICE_TYPE="jetson"
        elif uname -m | grep -q "aarch64\|arm"; then
            echo "Generic ARM device detected"
            DEVICE_TYPE="arm_generic"
        else
            echo "WARNING: Not an ARM device"
            DEVICE_TYPE="unknown"
        fi
    else
        echo "WARNING: Cannot detect device type"
        DEVICE_TYPE="unknown"
    fi
}

# Install system dependencies
install_system_deps() {
    echo ""
    echo "Installing system dependencies..."

    if [ "$DEVICE_TYPE" = "raspberry_pi" ]; then
        echo "Installing Raspberry Pi dependencies..."
        sudo apt-get update
        sudo apt-get install -y \
            libatlas-base-dev \
            libopenblas-dev \
            libhdf5-dev \
            libhdf5-serial-dev \
            libharfbuzz0b \
            libwebp6 \
            libjasper1 \
            libilmbase23 \
            libopenexr23 \
            libgstreamer1.0-0 \
            libavcodec58 \
            libavformat58 \
            libswscale5 \
            libqtgui4 \
            libqt4-test \
            python3-pip \
            python3-dev

        echo "✓ Raspberry Pi dependencies installed"

    elif [ "$DEVICE_TYPE" = "jetson" ]; then
        echo "Installing Jetson dependencies..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-dev

        echo "✓ Jetson dependencies installed"
        echo "  Note: TensorFlow should be installed via NVIDIA JetPack"

    else
        echo "Installing generic ARM dependencies..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-dev \
            libopenblas-dev

        echo "✓ Generic ARM dependencies installed"
    fi
}

# Create Python virtual environment
setup_virtualenv() {
    echo ""
    echo "Setting up Python virtual environment..."

    cd "$(dirname "$0")/../.."

    if [ ! -d "venv_arm" ]; then
        python3 -m venv venv_arm
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi

    source venv_arm/bin/activate
}

# Install Python dependencies
install_python_deps() {
    echo ""
    echo "Installing Python dependencies..."

    # Upgrade pip
    pip install --upgrade pip

    # Install ARM-specific requirements
    if [ "$DEVICE_TYPE" = "raspberry_pi" ]; then
        echo "Installing TFLite Runtime for Raspberry Pi..."
        # TFLite Runtime is much smaller than full TensorFlow
        pip install tflite-runtime

    elif [ "$DEVICE_TYPE" = "jetson" ]; then
        echo "Using pre-installed TensorFlow on Jetson..."
        # Jetson comes with TensorFlow pre-installed via JetPack
        # Skip TFLite installation

    else
        echo "Installing TFLite Runtime..."
        pip install tflite-runtime || pip install tensorflow
    fi

    # Install other requirements
    pip install -r arm_edge/requirements_arm.txt

    echo "✓ Python dependencies installed"
}

# Setup ARM NN delegate (optional, for best performance)
setup_arm_nn() {
    echo ""
    echo "Setting up ARM NN delegate..."

    if [ "$DEVICE_TYPE" = "raspberry_pi" ]; then
        echo "ARM NN delegate available for Raspberry Pi"
        echo "To install: https://www.arm.com/technologies/arm-nn"
        # pip install pyarmnn (if available for your platform)

    elif [ "$DEVICE_TYPE" = "jetson" ]; then
        echo "ARM NN not needed on Jetson (using TensorRT instead)"

    else
        echo "ARM NN delegate may be available for your device"
    fi

    echo "✓ ARM NN setup checked"
}

# Download pre-quantized models
download_models() {
    echo ""
    echo "Setting up models..."

    mkdir -p arm_edge/models

    # Check if quantized model exists
    if [ ! -f "arm_edge/models/emotion_quantized_armnn.tflite" ]; then
        echo "Quantized model not found - creating demo model..."
        python arm_edge/optimizations/quantize_models.py
        echo "✓ Demo model created and quantized"
    else
        echo "✓ Quantized model already exists"
    fi
}

# Configure environment
configure_env() {
    echo ""
    echo "Configuring environment..."

    # Create .env file for ARM edge service
    cat > arm_edge/.env << EOF
# ARM Edge Service Configuration

# Cloud Backend
CLOUD_BACKEND_URL=http://localhost:8000

# Privacy Mode: strict, balanced, minimal
PRIVACY_MODE=balanced

# Offline Mode: true/false
OFFLINE_MODE=false

# Performance Settings
ARM_THREADS=4
ENABLE_ARM_NN=true
ENABLE_NEON=true

# Device Type
DEVICE_TYPE=$DEVICE_TYPE
EOF

    echo "✓ Environment configured"
    echo "  Edit arm_edge/.env to customize settings"
}

# Performance tuning for ARM
optimize_performance() {
    echo ""
    echo "Optimizing ARM performance..."

    if [ "$DEVICE_TYPE" = "raspberry_pi" ]; then
        # Raspberry Pi optimizations
        echo "Applying Raspberry Pi optimizations..."

        # Set CPU governor to performance mode
        echo "  Setting CPU governor to performance mode..."
        for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
            if [ -f "$cpu/cpufreq/scaling_governor" ]; then
                echo "performance" | sudo tee "$cpu/cpufreq/scaling_governor" > /dev/null 2>&1 || true
            fi
        done

        echo "✓ Raspberry Pi optimizations applied"

    elif [ "$DEVICE_TYPE" = "jetson" ]; then
        # Jetson optimizations
        echo "Applying Jetson optimizations..."

        # Set Jetson to max performance mode
        if command -v nvpmodel &> /dev/null; then
            echo "  Setting Jetson to max performance..."
            sudo nvpmodel -m 0 || true
            sudo jetson_clocks || true
        fi

        echo "✓ Jetson optimizations applied"
    fi
}

# Run tests
run_tests() {
    echo ""
    echo "Running ARM edge service tests..."

    # Basic import test
    python -c "from arm_edge.facial_detector_arm import ARMFacialEmotionDetector; print('✓ Facial detector imports OK')"
    python -c "from arm_edge.audio_processor_arm import ARMAudioDistressProcessor; print('✓ Audio processor imports OK')"

    echo "✓ All tests passed"
}

# Start service
start_service() {
    echo ""
    echo "=========================================="
    echo "ARM Edge Service Ready!"
    echo "=========================================="
    echo ""
    echo "Device: $DEVICE_TYPE"
    echo "ARM NEON: Enabled"
    echo "Privacy Mode: balanced"
    echo ""
    echo "To start the service:"
    echo "  source venv_arm/bin/activate"
    echo "  python -m uvicorn arm_edge.arm_server:app --host 0.0.0.0 --port 8001"
    echo ""
    echo "Service will be available at:"
    echo "  http://localhost:8001"
    echo ""
    echo "API endpoints:"
    echo "  POST /edge/process - Main processing endpoint"
    echo "  GET  /arm/performance - Performance metrics"
    echo "  GET  /arm/benchmark - Run benchmarks"
    echo "  GET  /health - Health check"
    echo ""
}

# Main deployment flow
main() {
    detect_device

    read -p "Continue with deployment on $DEVICE_TYPE? (y/n) " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 0
    fi

    install_system_deps
    setup_virtualenv
    install_python_deps
    setup_arm_nn
    download_models
    configure_env
    optimize_performance
    run_tests
    start_service
}

# Run deployment
main
