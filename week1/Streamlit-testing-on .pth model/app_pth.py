import streamlit as st
import torch
import torch.nn as nn
import torchaudio.transforms as T
import librosa
import numpy as np
import tempfile
import os

# ===============================
# CONFIG
# ===============================
TARGET_SR = 16000
N_MELS = 64
THREAT_THRESHOLD = 0.50

ID_TO_LABEL = {
    0: "Safe/Neutral",
    1: "DANGER (Fear)",
    2: "Caution (Angry)"
}

# ===============================
# MODEL ARCH (UNCHANGED)
# ===============================
class DC_AC_Block(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.branch_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )
        self.branch_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return x * (self.branch_a(x) * self.branch_b(x))


class TinySafetyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = DC_AC_Block(32)
        self.conv1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.layer2 = DC_AC_Block(64)
        self.dropout2 = nn.Dropout(0.25)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.dropout1(self.conv1(x))
        x = self.dropout2(self.layer2(x))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


# ===============================
# LOAD MODEL (CACHED)
# ===============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySafetyNet().to(device)
    model.load_state_dict(torch.load("tiny_safety_3class.pth", map_location=device))
    model.eval()
    return model, device


# ===============================
# PREPROCESS (UNCHANGED)
# ===============================
def preprocess_chunk(chunk):
    max_val = np.abs(chunk).max()
    if max_val > 0:
        chunk = chunk / max_val

    waveform = torch.from_numpy(chunk).float().unsqueeze(0)
    spec = T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_mels=N_MELS
    )(waveform)
    spec = T.AmplitudeToDB()(spec)
    spec = torch.nn.functional.interpolate(
        spec.unsqueeze(0),
        size=(N_MELS, N_MELS)
    ).squeeze(0)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
    return spec.unsqueeze(0)


# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Tiny Safety Audio Monitor", layout="centered")

st.title("🎧 TinyML Audio Safety Monitor")
st.write("Detects **Fear / Scream (Angry)** from short audio clips.")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    if st.button("🔍 Analyze Audio"):
        model, device = load_model()

        audio, _ = librosa.load(audio_path, sr=TARGET_SR)

        window_samples = TARGET_SR
        step = int(window_samples * 0.5)

        max_threat = 0.0
        dominant = "Safe"

        for i in range(0, len(audio) - window_samples, step):
            chunk = audio[i:i + window_samples]
            if np.sqrt(np.mean(chunk**2)) < 0.01:
                continue

            input_tensor = preprocess_chunk(chunk).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)[0]
                p_safe, p_fear, p_angry = probs.tolist()

                threat = p_fear + p_angry
                if threat > max_threat:
                    max_threat = threat
                    dominant = "Fear" if p_fear > p_angry else "Scream (Angry)"

        st.subheader("🧠 Result")

        if max_threat > THREAT_THRESHOLD:
            st.error(f"🚨 DANGER detected ({dominant})")
        else:
            st.success("✅ Safe / Neutral")

        st.metric("Threat Score", f"{max_threat:.0%}")

    os.remove(audio_path)
