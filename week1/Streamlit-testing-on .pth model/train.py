import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
import zipfile
import random

# Try importing librosa
try:
    import librosa
except ImportError:
    print("ERROR: librosa not found. Please run: pip install librosa soundfile")
    exit()

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    ZIP_FILE = "archive.zip"
    DATA_PATH = "./tess_data"
    SAMPLE_RATE = 24414
    TARGET_SR = 16000
    N_MELS = 64
    TIME_STEPS = 64
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 0.0005 # Slower, more careful learning
    
    # === THE SAFETY STRATEGY ===
    # Map the 7 folders into 3 Logical Safety Classes
    # 0 = Safe (Ignore)
    # 1 = Danger (ALARM)
    # 2 = Caution (Monitor)
    EMOTION_MAP = {
        'neutral': 0, 
        'happy': 0, 
        'sad': 0, 
        'disgust': 0, # Merged into Safe so low-pitch voices don't trigger alarms
        'surprise': 0,
        'fear': 1,    # The priority
        'angry': 2    # Separate to prevent confusion
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ==========================================
# 2. DATA AUGMENTATION (From Notebook)
# ==========================================
class AudioAugmenter:
    """
    Motivation from audio-emotion-part-5-data-augmentation.ipynb
    Helps model handle real-world noise and different voice pitches.
    """
    @staticmethod
    def add_noise(data):
        # Add random white noise
        noise_amp = 0.005 * np.random.uniform() * np.amax(data)
        data = data + noise_amp * np.random.normal(size=data.shape)
        return data

    @staticmethod
    def stretch(data, rate=0.8):
        # Speed up or slow down
        return librosa.effects.time_stretch(y=data, rate=rate)

    @staticmethod
    def shift(data):
        # Shift audio left/right
        shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
        return np.roll(data, shift_range)

    @staticmethod
    def pitch(data, sampling_rate, pitch_factor=0.7):
        # Change pitch (make voice deeper or higher)
        return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

# ==========================================
# 3. DATASET
# ==========================================
class TESSDataset(Dataset):
    def __init__(self, files, labels, train_mode=False):
        self.files = files
        self.labels = labels
        self.train_mode = train_mode # Only augment training data
        self.mel_transform = T.MelSpectrogram(
            sample_rate=Config.TARGET_SR,
            n_mels=Config.N_MELS,
            n_fft=1024,
            hop_length=512
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Load with Librosa
            waveform_np, _ = librosa.load(path, sr=Config.TARGET_SR)
            
            # === APPLY AUGMENTATION (Randomly) ===
            if self.train_mode:
                # 30% Chance to Add Noise
                if random.random() < 0.3:
                    waveform_np = AudioAugmenter.add_noise(waveform_np)
                
                # 30% Chance to Pitch Shift (Helps with the "Disgust" confusion)
                if random.random() < 0.3:
                    step = random.uniform(-2, 2) # Shift pitch slightly up or down
                    waveform_np = AudioAugmenter.pitch(waveform_np, Config.TARGET_SR, step)
            # =====================================

            # Convert to Tensor
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)

            # Spectrogram
            spec = self.mel_transform(waveform)
            spec = T.AmplitudeToDB()(spec)
            
            # Resize
            spec = torch.nn.functional.interpolate(
                spec.unsqueeze(0), size=(Config.N_MELS, Config.TIME_STEPS), mode='bilinear'
            ).squeeze(0)
            
            # Normalize
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
            
            return spec, torch.tensor(label, dtype=torch.long)
        except Exception:
            return torch.zeros(1, Config.N_MELS, Config.TIME_STEPS), torch.tensor(label, dtype=torch.long)

# ==========================================
# 4. DATA SETUP & SPLIT
# ==========================================
def setup_data():
    if not os.path.exists(Config.DATA_PATH):
        if os.path.exists(Config.ZIP_FILE):
            print(f"Unzipping {Config.ZIP_FILE}...")
            with zipfile.ZipFile(Config.ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(Config.DATA_PATH)
        else:
            print("Please upload archive.zip!")
            exit()

def get_file_paths():
    all_files = glob.glob(f"{Config.DATA_PATH}/**/*.wav", recursive=True)
    file_list = []
    labels = []
    
    print("\n[Step 2] Categorizing Audio Files...")
    counts = {0: 0, 1: 0, 2: 0} # Safe, Danger, Caution

    for f in all_files:
        filename = os.path.basename(f).lower()
        parent = os.path.basename(os.path.dirname(f)).lower()
        
        # Check mapping
        for emotion, class_id in Config.EMOTION_MAP.items():
            if emotion in parent or emotion in filename:
                file_list.append(f)
                labels.append(class_id)
                counts[class_id] += 1
                break
    
    print(f"Total files: {len(file_list)}")
    print(f"Safe (0): {counts[0]} | Danger (1): {counts[1]} | Caution (2): {counts[2]}")
    return file_list, labels

# ==========================================
# 5. DC-AC MODEL (Your TinyML Arch)
# ==========================================
class DC_AC_Block(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.branch_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1), nn.ReLU(),
            nn.Conv2d(reduced, channels, 1), nn.Sigmoid()
        )
        self.branch_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1), nn.ReLU(),
            nn.Conv2d(reduced, channels, 1), nn.Tanh()
        )
    def forward(self, x):
        return x * (self.branch_a(x) * self.branch_b(x))

class TinySafetyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = DC_AC_Block(32)
        self.conv1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.dropout1 = nn.Dropout(0.2) # Added Dropout for robustness
        
        self.layer2 = DC_AC_Block(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # OUTPUT: 3 Classes (Safe, Danger, Caution)
        self.fc = nn.Linear(64, 3) 

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.dropout1(self.conv1(x))
        x = self.dropout2(self.layer2(x))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)

# ==========================================
# 6. MAIN LOOP
# ==========================================
def main():
    setup_data()
    files, labels = get_file_paths()
    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2, random_state=42)
    
    # Enable augmentation ONLY for training
    train_ds = TESSDataset(X_train, y_train, train_mode=True)
    test_ds = TESSDataset(X_test, y_test, train_mode=False)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print("\n[Step 3] Initializing Safety Model...")
    model = TinySafetyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    # Weight the loss because "Safe" class has 5x more data than "Fear"
    # [Safe weight, Fear weight, Caution weight]
    class_weights = torch.tensor([1.0, 4.0, 2.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print("Starting Training...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0
        for specs, targets in train_loader:
            specs, targets = specs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for specs, targets in test_loader:
                specs, targets = specs.to(device), targets.to(device)
                outputs = model(specs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()
        
        acc = 100 * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1:02d} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

    print("\n[Step 4] Saving Safety Model...")
    torch.save(model.state_dict(), 'tiny_safety_3class.pth')
    print("Saved as 'tiny_safety_3class.pth'")

if __name__ == "__main__":
    main()
