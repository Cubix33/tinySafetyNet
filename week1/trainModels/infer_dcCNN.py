# # inference.py
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
    "duration": 3.0,        # window size = 3 seconds
    "n_mfcc": 40,
    "model_path": "women_safety_dscnn_f16.tflite",
    "classes_path": "classes.npy"
}

# 3.0 seconds * 22050 Hz = 66150 samples
TARGET_LENGTH = int(CONFIG["sample_rate"] * CONFIG["duration"])

# Expected MFCC frames for 3 sec (~130)
EXPECTED_FRAMES = 130


# ==========================================
# 2. SAFETY PREDICTOR CLASS
# ==========================================
class SafetyPredictor:
    def __init__(self):
        print(f"--> Loading {CONFIG['model_path']}...")

        # Load classes
        if not os.path.exists(CONFIG["classes_path"]):
            sys.exit("Error: classes.npy not found. Run training first.")
        self.classes = np.load(CONFIG["classes_path"], allow_pickle=True)

        # Load TFLite model
        if not os.path.exists(CONFIG["model_path"]):
            sys.exit(f"Error: {CONFIG['model_path']} not found")

        self.interpreter = tf.lite.Interpreter(model_path=CONFIG["model_path"])
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        print(f"--> Model Input Shape: {self.input_shape}")
        print("--> System Ready.")

    # ---------------------------------------------------
    # Preprocess ONE 3-second audio chunk
    # ---------------------------------------------------
    def preprocess_chunk(self, y, sr):
        """
        y : raw audio chunk of length TARGET_LENGTH
        returns tensor shape (1, 40, 130, 1)
        """

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=CONFIG["n_mfcc"]
        ).astype(np.float32)

        # Fix MFCC frames to exactly 130
        curr = mfcc.shape[1]
        if curr < EXPECTED_FRAMES:
            mfcc = np.pad(mfcc, ((0, 0), (0, EXPECTED_FRAMES - curr)), mode="constant")
        else:
            mfcc = mfcc[:, :EXPECTED_FRAMES]

        # Add batch & channel dims → (1, 40, 130, 1)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)

        return mfcc

    # ---------------------------------------------------
    # Predict emotion for ONE chunk
    # ---------------------------------------------------
    def predict_chunk(self, mfcc_tensor):
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            mfcc_tensor
        )
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        probs = output_data[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        emotion = self.classes[pred_idx]

        return emotion, confidence

    # ---------------------------------------------------
    # Predict for FULL AUDIO using sliding 3-second windows
    # ---------------------------------------------------
    def predict(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=CONFIG["sample_rate"])
        except Exception as e:
            print(f"Error reading audio: {e}")
            return

        # Trim silence
        y, _ = librosa.effects.trim(y)

        total_len = len(y)
        step = TARGET_LENGTH  # 3 seconds step
        num_chunks = int(np.ceil(total_len / step))

        print(f"\n--> Total audio length: {total_len/sr:.2f} sec")
        print(f"--> Processing in {num_chunks} chunk(s) of 3 sec each\n")

        UNSAFE_EMOTIONS = ['fear', 'angry']
        overall_unsafe = False

        for i in range(num_chunks):
            start = i * step
            end = start + step
            chunk = y[start:end]

            # Pad last chunk if needed
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)), mode="constant")

            # Preprocess chunk
            mfcc_tensor = self.preprocess_chunk(chunk, sr)

            # Predict chunk
            emotion, confidence = self.predict_chunk(mfcc_tensor)

            # Safety logic
            is_unsafe = emotion in UNSAFE_EMOTIONS
            if emotion == 'disgust':
                is_unsafe = False
                status = "SAFE ✅ (Filtered Noise)"
            elif is_unsafe:
                status = "UNSAFE 🚨"
                overall_unsafe = True
            else:
                status = "SAFE ✅"

            print("-" * 50)
            print(f"Chunk {i+1}/{num_chunks}  [{start/sr:.1f}s → {min(end, total_len)/sr:.1f}s]")
            print(f"Emotion: {emotion.upper()} ({confidence:.1f}%)")
            print(f"Status:  {status}")
            print("-" * 50)

        # Final Decision
        print("\n" + "=" * 60)
        if overall_unsafe:
            print("FINAL RESULT:  UNSAFE 🚨  (Fear/Anger detected in at least one chunk)")
        else:
            print("FINAL RESULT:  SAFE ✅   (No unsafe emotion detected)")
        print("=" * 60)


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
        print("\nUsage: python inference.py <path_to_audio_or_video_file>")
        print("Example: python inference.py test_audio.wav")


# import os
# import sys
# import numpy as np
# import librosa
# import tensorflow as tf
# import warnings

# warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # ==========================================
# # 1. CONFIGURATION
# # ==========================================
# CONFIG = {
#     "sample_rate": 22050,
#     "duration": 3.0,
#     "n_mfcc": 40,
#     # Point to your DS-CNN TFLite model
#     "model_path": "women_safety_dscnn_f16.tflite",
#     "classes_path": "classes.npy"
# }

# # 3.0 seconds * 22050 Hz = 66150 samples
# TARGET_LENGTH = int(CONFIG["sample_rate"] * CONFIG["duration"])

# # Expected MFCC time steps (approx 130 for 3.0s)
# EXPECTED_FRAMES = 130


# # ==========================================
# # 2. SAFETY PREDICTOR CLASS
# # ==========================================
# class SafetyPredictor:
#     def __init__(self):
#         print(f"--> Loading {CONFIG['model_path']}...")

#         # Load Classes
#         if not os.path.exists(CONFIG["classes_path"]):
#             sys.exit("Error: classes.npy not found. Run training first.")
#         self.classes = np.load(CONFIG["classes_path"], allow_pickle=True)

#         # Load TFLite Model
#         if not os.path.exists(CONFIG["model_path"]):
#             sys.exit(f"Error: {CONFIG['model_path']} not found")

#         # Standard Interpreter (DS-CNN works universally)
#         self.interpreter = tf.lite.Interpreter(model_path=CONFIG["model_path"])
#         self.interpreter.allocate_tensors()

#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()

#         # Verify shape
#         self.input_shape = self.input_details[0]['shape']
#         # Typically [1, 40, 130, 1] for our DS-CNN
#         print(f"--> Model Input Shape: {self.input_shape}")
#         print("--> System Ready.")

#     def preprocess(self, audio_path):
#         try:
#             # Load audio
#             y, sr = librosa.load(audio_path, sr=CONFIG["sample_rate"])
#         except Exception as e:
#             print(f"Error reading audio: {e}")
#             return None

#         # Trim silence
#         y, _ = librosa.effects.trim(y)

#         # Pad/Truncate to target sample length
#         if len(y) > TARGET_LENGTH:
#             y = y[:TARGET_LENGTH]
#         else:
#             padding = TARGET_LENGTH - len(y)
#             y = np.pad(y, (0, padding), mode="constant")

#         # Extract MFCC
#         mfcc = librosa.feature.mfcc(
#             y=y,
#             sr=sr,
#             n_mfcc=CONFIG["n_mfcc"]
#         ).astype(np.float32)

#         # Fix time steps to exactly EXPECTED_FRAMES (130)
#         curr = mfcc.shape[1]
#         if curr < EXPECTED_FRAMES:
#             pad_width = EXPECTED_FRAMES - curr
#             mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
#         else:
#             mfcc = mfcc[:, :EXPECTED_FRAMES]

#         # Add batch & channel dimensions: (40, 130) -> (1, 40, 130, 1)
#         input_tensor = np.expand_dims(mfcc, axis=0)
#         input_tensor = np.expand_dims(input_tensor, axis=-1)

#         return input_tensor

#     def predict(self, audio_path):
#         inp = self.preprocess(audio_path)
#         if inp is None:
#             return

#         # Set input
#         self.interpreter.set_tensor(
#             self.input_details[0]['index'],
#             inp
#         )

#         # Run inference
#         self.interpreter.invoke()

#         # Get output
#         output_data = self.interpreter.get_tensor(
#             self.output_details[0]['index']
#         )
#         probs = output_data[0]  # Probability distribution

#         # Get max probability class
#         pred_idx = np.argmax(probs)
#         confidence = probs[pred_idx] * 100
#         emotion = self.classes[pred_idx]

#         # Safety Logic
#         UNSAFE_EMOTIONS = ['fear', 'angry']
#         is_unsafe = emotion in UNSAFE_EMOTIONS

#         # Hard-coded ignore for "disgust" (common false positive)
#         if emotion == 'disgust':
#             is_unsafe = False
#             status = "SAFE ✅ (Filtered Noise)"
#         elif is_unsafe:
#             status = "UNSAFE 🚨"
#         else:
#             status = "SAFE ✅"

#         print("-" * 40)
#         print(f"File:    {os.path.basename(audio_path)}")
#         print(f"Emotion: {emotion.upper()} ({confidence:.1f}%)")
#         print(f"Result:  {status}")
#         print("-" * 40)


# # ==========================================
# # 3. MAIN
# # ==========================================
# if __name__ == "__main__":
#     predictor = SafetyPredictor()

#     if len(sys.argv) > 1:
#         test_file = sys.argv[1]
#         if os.path.exists(test_file):
#             predictor.predict(test_file)
#         else:
#             print("File not found.")
#     else:
#         print("\nUsage: python inference.py <path_to_audio_file>")
#         print("Example: python inference.py test_audio.wav")
