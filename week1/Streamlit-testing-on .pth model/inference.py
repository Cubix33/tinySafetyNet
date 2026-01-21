inference.py:import torch
import torch.nn as nn
import torchaudio.transforms as T
import librosa
import numpy as np
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    TARGET_SR = 16000
    N_MELS = 64
    # ID Mapping
    ID_TO_LABEL = { 0: 'Safe/Neutral', 1: 'DANGER (Fear)', 2: 'Caution (Angry)' }
    
    # Logic Settings
    ANALYSIS_WINDOW = 4.0   # Context Window
    REPORT_INTERVAL = 3.0   # How often to print
    SUB_WINDOW_SIZE = 1.0   # Model input size
    
    # THRESHOLDS (Lower = More Sensitive)
    FEAR_THRESHOLD = 0.45   # If ANY sub-chunk exceeds 45% Fear -> TRIGGER
    ANGRY_THRESHOLD = 0.60  # If ANY sub-chunk exceeds 60% Angry -> TRIGGER

# ==========================================
# 2. MODEL ARCHITECTURE (Unchanged)
# ==========================================
class DC_AC_Block(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.branch_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, reduced, 1), nn.ReLU(),
            nn.Conv2d(reduced, channels, 1), nn.Sigmoid()
        )
        self.branch_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, reduced, 1), nn.ReLU(),
            nn.Conv2d(reduced, channels, 1), nn.Tanh()
        )
    def forward(self, x): return x * (self.branch_a(x) * self.branch_b(x))

class TinySafetyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
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

# ==========================================
# 3. PREPROCESS (Unchanged)
# ==========================================
def preprocess_chunk(chunk):
    max_val = np.abs(chunk).max()
    if max_val > 0: chunk = chunk / max_val

    waveform = torch.from_numpy(chunk).float()
    if waveform.ndim == 1: waveform = waveform.unsqueeze(0)

    spec = T.MelSpectrogram(sample_rate=16000, n_mels=64)(waveform)
    spec = T.AmplitudeToDB()(spec)
    spec = torch.nn.functional.interpolate(spec.unsqueeze(0), size=(64,64)).squeeze(0)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
    return spec.unsqueeze(0)

# ==========================================
# 4. SCANNER (Peak Detection Logic)
# ==========================================
# ==========================================
# 4. SCANNER (Updated: Combined Logic)
# ==========================================
def scan_audio(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySafetyNet().to(device)
    
    if not os.path.exists('tiny_safety_3class.pth'):
        print("Model not found! Run train.py"); return
        
    model.load_state_dict(torch.load('tiny_safety_3class.pth', map_location=device))
    model.eval()

    print(f"\nScanning: {file_path}")
    print(f"Mode: Combined Threat (Fear + Angry)")
    
    try:
        audio, _ = librosa.load(file_path, sr=Config.TARGET_SR)
    except:
        print("Error reading file! Install ffmpeg."); return

    window_samples = int(Config.ANALYSIS_WINDOW * Config.TARGET_SR)
    step_samples = int(Config.REPORT_INTERVAL * Config.TARGET_SR)
    sub_window_samples = int(Config.SUB_WINDOW_SIZE * Config.TARGET_SR)
    
    print("-" * 75)
    print(f"{'Time':<12} | {'Prediction':<20} | {'Threat Score':<12} | {'Status'}")
    print("-" * 75)

    distress_count = 0
    
    for current_idx in range(0, len(audio), step_samples):
        
        # 1. Get the 4-second context
        start_idx = max(0, current_idx - window_samples + step_samples) 
        end_idx = min(len(audio), start_idx + window_samples)
        buffer_audio = audio[start_idx:end_idx]
        
        if len(buffer_audio) < sub_window_samples: continue

        # 2. Silence Check
        rms = np.sqrt(np.mean(buffer_audio**2))
        if rms < 0.01: 
            print(f"{current_idx/16000:.1f}s        | Safe (Silence)       | 0%           | 🟢")
            continue

        # 3. INTERNAL SCANNING (Overlap)
        max_threat_score = 0.0
        dominant_emotion = "Safe"
        
        sub_step = int(sub_window_samples * 0.5) 
        
        for sub_i in range(0, len(buffer_audio) - sub_window_samples + 1, sub_step):
            sub_chunk = buffer_audio[sub_i : sub_i + sub_window_samples]
            input_tensor = preprocess_chunk(sub_chunk).to(device)
            
            with torch.no_grad():
                probs = torch.nn.functional.softmax(model(input_tensor), dim=1)
                p_safe = probs[0][0].item()
                p_fear = probs[0][1].item()
                p_angry = probs[0][2].item()
                
                # === THE FIX: Sum Fear and Angry ===
                current_threat = p_fear + p_angry
                
                # Track the highest threat found in this window
                if current_threat > max_threat_score:
                    max_threat_score = current_threat
                    # Determine which of the two is driving the threat
                    if p_fear > p_angry:
                        dominant_emotion = "Fear"
                    else:
                        dominant_emotion = "Scream (Angry)"

        # 4. FINAL VERDICT
        # Threshold: If Fear+Angry is greater than 50%, it's dangerous.
        if max_threat_score > 0.50:
            status_icon = "🔴 DANGER"
            label = f"DANGER ({dominant_emotion})"
            distress_count += 1
        else:
            status_icon = "🟢"
            label = "Safe/Neutral"

        print(f"{current_idx/16000:.1f}s        | {label:<20} | {max_threat_score:.0%}          | {status_icon}")

    print("-" * 75)
    
    if distress_count >= 1: 
        print("🚨 ALARM TRIGGERED! Distress signal detected.")
    else: 
        print("✅ STATUS: SAFE.")

if __name__ == "__main__":
    scan_audio("fear.mp3")
