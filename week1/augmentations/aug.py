import numpy as np
import librosa
import soundfile as sf  # Use soundfile for saving
import matplotlib.pyplot as plt
import os

# CONFIG
SAMPLE_RATE = 16000 # Standard for TinyML
NOISE_FACTOR = 0.005 # Adjust this: 0.005 is subtle, 0.02 is loud
OUTPUT_DIR = "debug_audio_samples" # Directory to save the files

def add_white_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same type to avoid errors
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def save_and_plot(file_path):
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Audio
    data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Augment
    noisy_data = add_white_noise(data, NOISE_FACTOR)
    
    # Visualization (Still useful to see if the signal is "drowned" visually)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.title("Original Audio Waveform")
    plt.plot(data)
    
    plt.subplot(2, 1, 2)
    plt.title(f"Augmented Audio (Noise Factor: {NOISE_FACTOR})")
    plt.plot(noisy_data, color='orange')
    
    plt.tight_layout()
    # Save the plot image too, so you can view it without a GUI
    plot_path = os.path.join(OUTPUT_DIR, "augmentation_plot.png")
    plt.savefig(plot_path)
    print(f"📊 Visualization saved to: {plot_path}")
    plt.close() # Close plot to free memory

    # --- SAVE FILES INSTEAD OF PLAYING ---
    # 1. Save Original (resampled)
    orig_name = "original_check.wav"
    orig_path = os.path.join(OUTPUT_DIR, orig_name)
    sf.write(orig_path, data, sr)
    
    # 2. Save Augmented
    aug_name = f"augmented_noise_{NOISE_FACTOR}.wav"
    aug_path = os.path.join(OUTPUT_DIR, aug_name)
    sf.write(aug_path, noisy_data, sr)

    print(f"✅ Audio Saved!") 
    print(f"   Original:  {orig_path}")
    print(f"   Augmented: {aug_path}")
    print("   (Download these files to your local machine to listen)")

# --- TEST IT HERE ---
# Replace with a path to one of your actual WAV files
# e.g. "tess toronto emotional speech set data/OAF_Fear/OAF_beg_fear.wav"
test_file = "safety_data/tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/OAF_Fear/OAF_beg_fear.wav" 

try:
    if os.path.exists(test_file):
        save_and_plot(test_file)
    else:
        print(f"⚠️ File not found: {test_file}")
        print("Please set 'test_file' to a valid path in your dataset.")
except Exception as e:
    print(f"Error: {e}")