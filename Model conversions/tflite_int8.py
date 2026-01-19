import tensorflow as tf
import numpy as np
import librosa
import glob

# ================================
# CONFIG (must match training!)
# ================================
TARGET_SR = 16000
N_MELS = 64
TIME_STEPS = 64

# ================================
# Load audio files
# ================================
audio_files = glob.glob("./tess_data/**/*.wav", recursive=True)

def representative_dataset():
    for path in audio_files[:100]:  # 100 samples is enough
        try:
            audio, _ = librosa.load(path, sr=TARGET_SR)

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=TARGET_SR,
                n_mels=N_MELS,
                n_fft=1024,
                hop_length=512
            )

            mel_db = librosa.power_to_db(mel)

            # Resize to 64x64
            mel_db = mel_db[:N_MELS, :TIME_STEPS]
            mel_db = np.pad(
                mel_db,
                ((0, max(0, N_MELS - mel_db.shape[0])),
                 (0, max(0, TIME_STEPS - mel_db.shape[1]))),
                mode='constant'
            )

            # Normalize (same as training)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

            mel_db = mel_db.astype(np.float32)
            mel_db = np.expand_dims(mel_db, axis=(0, 1))  # (1,1,64,64)

            yield [mel_db]

        except Exception:
            continue

# ================================
# Convert to INT8
# ================================
converter = tf.lite.TFLiteConverter.from_saved_model("tf_safety_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# FULL INT8 (ESP32 safe)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()

with open("tiny_safety_3class_int8.tflite", "wb") as f:
    f.write(tflite_int8)

print("✅ INT8 TFLite model created")
