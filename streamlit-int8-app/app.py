import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import paho.mqtt.client as mqtt
import tempfile
import os
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "sample_rate": 22050,
    "duration": 3.0,
    "n_mfcc": 40,
    "model_path": "women_safety_dscnn_f16.tflite",
    "classes_path": "classes.npy",
    # MQTT (Wokwi)
    "mqtt_broker": "broker.hivemq.com",
    "mqtt_topic": "tinyml/anshika/badge"
}

TARGET_LENGTH = int(CONFIG["sample_rate"] * CONFIG["duration"])
EXPECTED_FRAMES = 130 

# ==========================================
# 2. MQTT SETUP (FIXED)
# ==========================================
# ==========================================
# 2. MQTT SETUP (ROBUST VERSION)
# ==========================================
def send_to_wokwi(command):
    try:
        # 1. Setup Client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        # 2. Connect
        client.connect(CONFIG["mqtt_broker"], 1883, 60)
        client.loop_start() # Start background network thread
        
        # 3. Publish with QoS=1 (Guarantees delivery)
        # We capture the 'message info' object to track status
        msg_info = client.publish(CONFIG["mqtt_topic"], command, qos=1)
        
        # 4. BLOCK until the broker confirms receipt (Max 2 seconds)
        msg_info.wait_for_publish(timeout=2.0)
        
        # 5. Cleanup
        client.loop_stop()
        client.disconnect()
        return True

    except Exception as e:
        st.error(f"MQTT Error: {e}")
        return False

# ==========================================
# 3. LOAD MODEL & CLASSES
# ==========================================
@st.cache_resource
def load_resources():
    # 1. Load Classes
    if os.path.exists(CONFIG["classes_path"]):
        classes = np.load(CONFIG["classes_path"], allow_pickle=True)
    else:
        classes = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
    
    # 2. Load Model
    if not os.path.exists(CONFIG["model_path"]):
        st.error(f"Model file {CONFIG['model_path']} not found!")
        return None, None

    interpreter = tf.lite.Interpreter(model_path=CONFIG["model_path"])
    interpreter.allocate_tensors()
    return interpreter, classes

interpreter, classes = load_resources()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# ==========================================
# 4. PREPROCESS
# ==========================================
def preprocess_audio(audio_path):
    try:
        # Load audio (22050 Hz)
        y, sr = librosa.load(audio_path, sr=CONFIG["sample_rate"])
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Pad/Truncate to target sample length (3.0s)
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - len(y)
        y = np.pad(y, (0, padding), 'constant')

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG["n_mfcc"])
    mfcc = mfcc.astype(np.float32)

    # Fix Time Steps to exactly 130
    curr = mfcc.shape[1]
    if curr < EXPECTED_FRAMES:
        pad_width = EXPECTED_FRAMES - curr
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_FRAMES]
        
    # Add Batch & Channel dims: (1, 40, 130, 1)
    input_tensor = np.expand_dims(mfcc, axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=-1)
    
    return input_tensor

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.title("🛡️ Women Safety Analytics (Wokwi Linked)")
st.markdown(f"**Connected to:** `{CONFIG['mqtt_topic']}` on `{CONFIG['mqtt_broker']}`")

uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(audio_path)

    if st.button("Analyze Audio"):
        if interpreter is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Processing..."):
                # 1. Preprocess
                input_data = preprocess_audio(audio_path)
                
                if input_data is not None:
                    # 2. Inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    probs = interpreter.get_tensor(output_details[0]['index'])[0]

                    # 3. Get Result
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx] * 100
                    emotion = classes[pred_idx]

                    # 4. Safety Logic & MQTT
                    if emotion == 'fear':
                        status = "🚨 DANGER"
                        color = "red"
                        mqtt_cmd = "D"
                    elif emotion == 'angry':
                        status = "⚠️ CAUTION"
                        color = "orange"
                        mqtt_cmd = "C"
                    else:
                        status = "✅ SAFE"
                        color = "green"
                        mqtt_cmd = "S"

                    # 5. Send to Wokwi
                    sent = send_to_wokwi(mqtt_cmd)
                    
                    # 6. Display Result
                    st.markdown(f"## Result: :{color}[{status}]")
                    st.markdown(f"**Detected Emotion:** {emotion.upper()}")
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    
                    if sent:
                        st.success(f"📡 Signal '{mqtt_cmd}' sent to Wokwi Badge.")
                    else:
                        st.error("❌ Failed to send signal to Wokwi.")

                    # Show Bar Chart
                    st.bar_chart(probs)

    # Cleanup
    os.remove(audio_path)