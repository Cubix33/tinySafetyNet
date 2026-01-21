# inference.py

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "sample_rate": 22050,
    "duration": 3.0,
    "n_mfcc": 40,
    # Point to your DS-CNN TFLite model
    "model_path": "women_safety_dscnn_f16.tflite",
    "classes_path": "classes.npy"
}

# 3.0 seconds * 22050 Hz = 66150 samples
TARGET_LENGTH = int(CONFIG["sample_rate"] * CONFIG["duration"])

# Expected MFCC time steps (approx 130 for 3.0s)
EXPECTED_FRAMES = 130


# ==========================================
# 2. SAFETY PREDICTOR CLASS
# ==========================================
class SafetyPredictor:
    def __init__(self):
        print(f"--> Loading {CONFIG['model_path']}...")

        # Load Classes
        if not os.path.exists(CONFIG["classes_path"]):
            sys.exit("Error: classes.npy not found. Run training first.")
        self.classes = np.load(CONFIG["classes_path"], allow_pickle=True)

        # Load TFLite Model
        if not os.path.exists(CONFIG["model_path"]):
            sys.exit(f"Error: {CONFIG['model_path']} not found")

        # Standard Interpreter (DS-CNN works universally)
        self.interpreter = tf.lite.Interpreter(model_path=CONFIG["model_path"])
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Verify shape
        self.input_shape = self.input_details[0]['shape']
        # Typically [1, 40, 130, 1] for our DS-CNN
        print(f"--> Model Input Shape: {self.input_shape}")
        print("--> System Ready.")

    def preprocess(self, audio_path):
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=CONFIG["sample_rate"])
        except Exception as e:
            print(f"Error reading audio: {e}")
            return None

        # Trim silence
        y, _ = librosa.effects.trim(y)

        # Pad/Truncate to target sample length
        if len(y) > TARGET_LENGTH:
            y = y[:TARGET_LENGTH]
        else:
            padding = TARGET_LENGTH - len(y)
            y = np.pad(y, (0, padding), mode="constant")

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=CONFIG["n_mfcc"]
        ).astype(np.float32)

        # Fix time steps to exactly EXPECTED_FRAMES (130)
        curr = mfcc.shape[1]
        if curr < EXPECTED_FRAMES:
            pad_width = EXPECTED_FRAMES - curr
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :EXPECTED_FRAMES]

        # Add batch & channel dimensions: (40, 130) -> (1, 40, 130, 1)
        input_tensor = np.expand_dims(mfcc, axis=0)
        input_tensor = np.expand_dims(input_tensor, axis=-1)

        return input_tensor

    def predict(self, audio_path):
        inp = self.preprocess(audio_path)
        if inp is None:
            return

        # Set input
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            inp
        )

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        probs = output_data[0]  # Probability distribution

        # Get max probability class
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        emotion = self.classes[pred_idx]

        # Safety Logic
        UNSAFE_EMOTIONS = ['fear', 'angry']
        is_unsafe = emotion in UNSAFE_EMOTIONS

        # Hard-coded ignore for "disgust" (common false positive)
        if emotion == 'disgust':
            is_unsafe = False
            status = "SAFE ✅ (Filtered Noise)"
        elif is_unsafe:
            status = "UNSAFE 🚨"
        else:
            status = "SAFE ✅"

        print("-" * 40)
        print(f"File:    {os.path.basename(audio_path)}")
        print(f"Emotion: {emotion.upper()} ({confidence:.1f}%)")
        print(f"Result:  {status}")
        print("-" * 40)


# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    predictor = SafetyPredictor()

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if os.path.exists(test_file):
            predictor.predict(test_file)
        else:
            print("File not found.")
    else:
        print("\nUsage: python inference.py <path_to_audio_file>")
        print("Example: python inference.py test_audio.wav")
