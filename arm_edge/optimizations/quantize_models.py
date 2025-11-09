"""
Model Quantization for ARM Devices

Converts TensorFlow/PyTorch models to TensorFlow Lite with INT8 quantization
optimized for ARM processors.

Benefits of INT8 quantization on ARM:
- 4x smaller model size (critical for edge devices)
- 2-4x faster inference with ARM NEON
- Lower memory bandwidth usage
- Lower power consumption

Optimization techniques:
1. Post-training quantization (PTQ) - no retraining needed
2. ARM NN delegate integration
3. NEON-optimized operations
4. Dynamic range quantization
"""

import tensorflow as tf
import numpy as np
import os
import logging
from typing import Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARMModelQuantizer:
    """
    Quantize models for optimal ARM performance.

    Converts models to TFLite INT8 format with ARM NN optimizations.
    """

    def __init__(self, output_dir: str = "arm_edge/models"):
        """
        Initialize quantizer.

        Args:
            output_dir: Directory to save quantized models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def quantize_emotion_model(
        self,
        model_path: str,
        representative_dataset: Optional[Callable] = None
    ) -> str:
        """
        Quantize facial emotion detection model to INT8 for ARM.

        Args:
            model_path: Path to source model (SavedModel, H5, or Keras)
            representative_dataset: Generator yielding sample inputs for calibration

        Returns:
            Path to quantized TFLite model
        """
        logger.info(f"Quantizing emotion model: {model_path}")

        try:
            # Load model
            if model_path.endswith('.h5'):
                model = tf.keras.models.load_model(model_path)
            else:
                model = tf.saved_model.load(model_path)

            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Enable INT8 quantization (key ARM optimization)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Set INT8 as target (ARM NEON works best with INT8)
            converter.target_spec.supported_types = [tf.int8]

            # Use ARM NN delegate-compatible operations
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]

            # Representative dataset for full integer quantization
            if representative_dataset:
                converter.representative_dataset = representative_dataset
            else:
                # Default: random samples matching input shape
                input_shape = model.input_shape
                def default_dataset():
                    for _ in range(100):
                        # Generate random face-like images
                        yield [np.random.uniform(0, 255, input_shape[1:]).astype(np.float32)]
                converter.representative_dataset = default_dataset

            # Enable experimental ARM NN optimizations
            converter.experimental_new_converter = True

            # Convert model
            logger.info("Converting model to INT8 TFLite...")
            tflite_model = converter.convert()

            # Save quantized model
            output_path = os.path.join(self.output_dir, "emotion_quantized_armnn.tflite")
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            # Get size reduction
            original_size = os.path.getsize(model_path) if os.path.isfile(model_path) else 0
            quantized_size = len(tflite_model)

            logger.info(f"✓ Quantized model saved: {output_path}")
            logger.info(f"  Original size: {original_size / 1024:.1f} KB")
            logger.info(f"  Quantized size: {quantized_size / 1024:.1f} KB")
            logger.info(f"  Size reduction: {((1 - quantized_size/max(original_size, 1)) * 100):.1f}%")

            return output_path

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise

    def benchmark_quantized_model(self, tflite_path: str, num_runs: int = 100):
        """
        Benchmark quantized model on ARM device.

        Args:
            tflite_path: Path to TFLite model
            num_runs: Number of inference runs

        Returns:
            Dict with benchmark results
        """
        logger.info(f"Benchmarking quantized model: {tflite_path}")

        try:
            # Load interpreter with ARM NN delegate if available
            try:
                delegates = [tf.lite.load_delegate('libarmnn_delegate.so')]
                interpreter = tf.lite.Interpreter(
                    model_path=tflite_path,
                    experimental_delegates=delegates,
                    num_threads=4
                )
                arm_nn_enabled = True
                logger.info("✓ ARM NN delegate loaded")
            except:
                interpreter = tf.lite.Interpreter(
                    model_path=tflite_path,
                    num_threads=4
                )
                arm_nn_enabled = False
                logger.info("✗ ARM NN delegate not available - using standard TFLite")

            interpreter.allocate_tensors()

            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']

            logger.info(f"Input shape: {input_shape}, dtype: {input_dtype}")

            # Run benchmark
            import time
            inference_times = []

            for i in range(num_runs):
                # Generate random input
                if input_dtype == np.uint8:
                    test_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
                else:
                    test_input = np.random.rand(*input_shape).astype(np.float32)

                # Run inference
                start = time.time()
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                inference_time = (time.time() - start) * 1000

                inference_times.append(inference_time)

                if (i + 1) % 20 == 0:
                    logger.info(f"  {i + 1}/{num_runs} runs completed")

            # Calculate statistics
            results = {
                "avg_inference_ms": np.mean(inference_times),
                "min_inference_ms": np.min(inference_times),
                "max_inference_ms": np.max(inference_times),
                "std_inference_ms": np.std(inference_times),
                "p50_inference_ms": np.percentile(inference_times, 50),
                "p95_inference_ms": np.percentile(inference_times, 95),
                "p99_inference_ms": np.percentile(inference_times, 99),
                "num_runs": num_runs,
                "arm_nn_enabled": arm_nn_enabled,
                "quantization": "INT8" if input_dtype == np.uint8 else "FLOAT32"
            }

            logger.info("\nBenchmark Results:")
            logger.info(f"  Average: {results['avg_inference_ms']:.2f} ms")
            logger.info(f"  Min: {results['min_inference_ms']:.2f} ms")
            logger.info(f"  Max: {results['max_inference_ms']:.2f} ms")
            logger.info(f"  P95: {results['p95_inference_ms']:.2f} ms")
            logger.info(f"  ARM NN: {'✓' if arm_nn_enabled else '✗'}")

            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def create_demo_model(self) -> str:
        """
        Create a demo emotion detection model for testing.

        Returns a simple CNN that can be quantized and tested on ARM.
        """
        logger.info("Creating demo emotion detection model...")

        # Simple CNN for 48x48 grayscale face images
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(48, 48, 1)),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save model
        model_path = os.path.join(self.output_dir, "emotion_demo.h5")
        model.save(model_path)

        logger.info(f"✓ Demo model saved: {model_path}")
        logger.info(f"  Parameters: {model.count_params():,}")

        return model_path


def main():
    """
    Main function to quantize models for ARM deployment.
    """
    quantizer = ARMModelQuantizer()

    # Option 1: Create and quantize demo model
    logger.info("=== Creating Demo Model ===")
    demo_model_path = quantizer.create_demo_model()

    logger.info("\n=== Quantizing Model for ARM ===")
    quantized_path = quantizer.quantize_emotion_model(demo_model_path)

    logger.info("\n=== Benchmarking Quantized Model ===")
    results = quantizer.benchmark_quantized_model(quantized_path)

    logger.info("\n=== Summary ===")
    logger.info("Model ready for ARM deployment!")
    logger.info(f"Quantized model: {quantized_path}")
    logger.info(f"Average inference: {results['avg_inference_ms']:.2f} ms")
    logger.info(f"Suitable for: Raspberry Pi 4, Jetson Nano, ARM Cortex devices")

    # Option 2: Quantize your own model
    # Uncomment and modify:
    # quantizer.quantize_emotion_model("path/to/your/model.h5")


if __name__ == "__main__":
    main()
