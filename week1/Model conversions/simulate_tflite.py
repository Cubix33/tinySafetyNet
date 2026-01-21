import numpy as np
import librosa
import tensorflow as tf

# =========================
# CONFIG
# =========================
MODEL_PATH = "tiny_safety_3class.tflite"
AUDIO_PATH = "test.wav"
SAMPLE_RATE = 16000
N_MELS = 64
TARGET_FRAMES = 64

# =========================
# LOAD MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Model loaded")
print("Expected input shape:", input_details[0]['shape'])
print("Expected dtype:", input_details[0]['dtype'])

# =========================
# LOAD AUDIO
# =========================
audio, _ = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)

# =========================
# MEL SPECTROGRAM
# =========================
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=SAMPLE_RATE,
    n_fft=512,
    hop_length=256,
    n_mels=N_MELS
)

mel_db = librosa.power_to_db(mel, ref=np.max)

# =========================
# FIX SHAPE TO (64, 64)
# =========================
mel_db = mel_db[:N_MELS, :]

if mel_db.shape[1] < TARGET_FRAMES:
    mel_db = np.pad(
        mel_db,
        ((0, 0), (0, TARGET_FRAMES - mel_db.shape[1])),
        mode="constant"
    )
else:
    mel_db = mel_db[:, :TARGET_FRAMES]

# =========================
# NORMALIZE
# =========================
mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
mel_db = mel_db.astype(np.float32)

# =========================
# FORMAT INPUT  → (1, 1, 64, 64)
# =========================
input_data = np.expand_dims(mel_db, axis=0)   # batch
input_data = np.expand_dims(input_data, axis=1)  # channel FIRST

print("Final input shape:", input_data.shape)

# =========================
# INFERENCE
# =========================
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])

# =========================
# RESULT
# =========================
print("Output:", output)
print("Predicted class:", np.argmax(output))
