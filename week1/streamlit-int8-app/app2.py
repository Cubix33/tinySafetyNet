import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import paho.mqtt.client as mqtt
import pyaudio
import os
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "sample_rate": 22050,
    "duration": 3.0,       # Context window
    "chunk_duration": 0.5, # Updates every 0.5s
    "n_mfcc": 40,
    "model_path": "women_safety_dscnn_f16.tflite",
    "classes_path": "classes.npy",
    "mqtt_broker": "broker.hivemq.com",
    "mqtt_topic": "tinyml/anshika/badge"
}

TARGET_LENGTH = int(CONFIG["sample_rate"] * CONFIG["duration"]) # 66150 samples
CHUNK_SAMPLES = int(CONFIG["sample_rate"] * CONFIG["chunk_duration"]) # 11025 samples
EXPECTED_FRAMES = 130 

# ==========================================
# 2. MQTT FUNCTION (Non-Blocking)
# ==========================================
# We use a simplified sender for the loop so it doesn't freeze the UI
def send_signal(client, command):
    try:
        client.publish(CONFIG["mqtt_topic"], command)
    except Exception as e:
        print(f"MQTT Error: {e}")

# ==========================================
# 3. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_model():
    # Load Classes
    if os.path.exists(CONFIG["classes_path"]):
        classes = np.load(CONFIG["classes_path"], allow_pickle=True)
    else:
        classes = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
    
    # Load Model
    interpreter = tf.lite.Interpreter(model_path=CONFIG["model_path"])
    interpreter.allocate_tensors()
    
    return interpreter, classes

interpreter, classes = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# 4. PREPROCESS (Real-Time Buffer Logic)
# ==========================================
def preprocess_live(audio_buffer):
    # 1. Trim Silence (Critical for DS-CNN)
    y, _ = librosa.effects.trim(audio_buffer)

    # 2. Pad/Crop to 3.0s
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - len(y)
        y = np.pad(y, (0, padding), 'constant')

    # 3. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=CONFIG["sample_rate"], n_mfcc=CONFIG["n_mfcc"])
    mfcc = mfcc.astype(np.float32)

    # 4. Fix Frames to 130
    curr = mfcc.shape[1]
    if curr < EXPECTED_FRAMES:
        pad_width = EXPECTED_FRAMES - curr
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_FRAMES]
        
    # 5. Reshape (1, 40, 130, 1)
    input_tensor = np.expand_dims(mfcc, axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=-1)
    
    return input_tensor

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("🎙️ Real-Time Safety Guard")
st.markdown("This app listens to your microphone and controls the Wokwi Badge instantly.")

# Create placeholders for dynamic updates
status_header = st.empty()
emotion_text = st.empty()
confidence_bar = st.empty()

# Start Button
run_live = st.toggle("🔴 START LISTENING", value=False)

# ==========================================
# 6. MAIN REAL-TIME LOOP
# ==========================================
if run_live:
    # --- SETUP AUDIO ---
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=CONFIG["sample_rate"],
                    input=True,
                    frames_per_buffer=CHUNK_SAMPLES)

    # --- SETUP MQTT ---
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.connect(CONFIG["mqtt_broker"], 1883, 60)
    mqtt_client.loop_start()

    # --- INIT BUFFER (3 Seconds of Silence) ---
    # We keep a rolling buffer of the last 3 seconds
    audio_buffer = np.zeros(TARGET_LENGTH, dtype=np.float32)

    st.toast("Microphone Active! Scream to test.", icon="🎤")

    try:
        while run_live:
            # 1. Read new Audio Chunk (0.5s)
            data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            new_chunk = np.frombuffer(data, dtype=np.float32)

            # 2. Roll Buffer (Remove old 0.5s, Add new 0.5s)
            audio_buffer = np.roll(audio_buffer, -len(new_chunk))
            audio_buffer[-len(new_chunk):] = new_chunk

            # 3. Silence Check (Optimization)
            vol = np.sqrt(np.mean(new_chunk**2))
            if vol < 0.01:
                status_header.markdown("## 💤 Status: :grey[Silence]")
                emotion_text.text("Waiting for sound...")
                continue

            # 4. Inference
            input_data = preprocess_live(audio_buffer)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            probs = interpreter.get_tensor(output_details[0]['index'])[0]

            # 5. Results
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] * 100
            emotion = classes[pred_idx]

            # 6. Logic
            if emotion == 'fear' and confidence > 50:
                final_status = "🚨 DANGER"
                color = "red"
                send_signal(mqtt_client, "D")
            elif emotion == 'angry' and confidence > 50:
                final_status = "⚠️ CAUTION"
                color = "orange"
                send_signal(mqtt_client, "C")
            else:
                final_status = "✅ SAFE"
                color = "green"
                send_signal(mqtt_client, "S")

            # 7. Update UI (Instantly)
            status_header.markdown(f"## Status: :{color}[{final_status}]")
            emotion_text.markdown(f"**Detected:** {emotion.upper()} ({confidence:.1f}%)")
            
            # Show a cute progress bar for confidence
            confidence_bar.progress(int(confidence))
            
            # Small sleep to yield CPU
            time.sleep(0.05)

    except Exception as e:
        st.error(f"Error: {e}")
    
    finally:
        # Cleanup when toggle is switched off
        stream.stop_stream()
        stream.close()
        p.terminate()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        st.info("Stopped Listening.")