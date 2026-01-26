import os
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, SeparableConv1D, BatchNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import sys

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'root_dir': 'downloads',
    'sample_rate': 22050,
    'duration': 2.5,
    'offset': 0.6,
    'batch_size': 32,
    'epochs': 50,
    'target_len': 2376 
}

# ==========================================
# 2. DATA AUGMENTATION & UTILS
# ==========================================
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    result = np.squeeze(zcr)
    
    # RMSE
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    result = np.hstack((result, np.squeeze(rmse)))
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=frame_length, hop_length=hop_length)
    result = np.hstack((result, np.ravel(mfcc.T)))
    
    return result

# ==========================================
# 3. DATA LOADING (PATHS ONLY)
# ==========================================
def load_file_paths(root_dir):
    paths = []
    labels = []
    
    if not os.path.exists(root_dir):
        sys.exit(f"ERROR: Directory '{root_dir}' not found.")

    # --- RAVDESS ---
    ravdess_map = {'01': 0, '02': 0, '03': 0, '04': -1, '05': 1, '06': 1, '07': -1, '08': -1}
    for f in glob.glob(os.path.join(root_dir, 'ravdess', '**', '*.wav'), recursive=True):
        parts = os.path.basename(f).split('-')
        if len(parts) >= 3:
            code = parts[2]
            if code in ravdess_map and ravdess_map[code] != -1:
                paths.append(f); labels.append(ravdess_map[code])

    # --- TESS ---
    for f in glob.glob(os.path.join(root_dir, 'tess', '**', '*.wav'), recursive=True):
        fname = os.path.basename(f).lower()
        if 'angry' in fname or 'fear' in fname:
            paths.append(f); labels.append(1)
        elif 'neutral' in fname or 'happy' in fname:
            paths.append(f); labels.append(0)

    # --- SAVEE ---
    savee_map = {'a': 1, 'f': 1, 'n': 0, 'h': 0, 'd': -1, 'sa': -1, 'su': -1}
    for f in glob.glob(os.path.join(root_dir, 'savee', '*.wav')):
        fname = os.path.basename(f)
        code = fname[:2] if fname[1] in ['a','u'] else fname[0]
        if code in savee_map and savee_map[code] != -1:
            paths.append(f); labels.append(savee_map[code])
            
    # --- CREMA-D ---
    crema_map = {'ANG':1, 'FEA':1, 'NEU':0, 'HAP':0, 'SAD':-1, 'DIS':-1}
    for f in glob.glob(os.path.join(root_dir, 'cremad', '*.wav')):
        try:
            part = os.path.basename(f).split('_')[2]
            if part in crema_map and crema_map[part] != -1:
                paths.append(f); labels.append(crema_map[part])
        except: pass

    return pd.DataFrame({'path': paths, 'label': labels})

print("Scanning Data...")
df = load_file_paths(CONFIG['root_dir'])
print(f"Total Files Found: {len(df)}")

# ==========================================
# 4. SPLIT FIRST -> THEN AUGMENT (The Fix)
# ==========================================
# 1. Perform Split on the Dataframe (Files), NOT the features
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print(f"Training Files: {len(train_df)} | Testing Files: {len(test_df)}")

def process_dataset(dataframe, augment=False):
    X_local, Y_local = [], []
    
    for idx, row in dataframe.iterrows():
        try:
            # Load Audio
            data, sr = librosa.load(row['path'], duration=CONFIG['duration'], offset=CONFIG['offset'], sr=CONFIG['sample_rate'])
            if len(data) < 0.1 * CONFIG['sample_rate']: continue

            # Helper to extract and fix shape
            def get_feat(audio_data):
                feat = extract_features(audio_data, sr=sr)
                # Pad/Truncate
                if len(feat) > CONFIG['target_len']:
                    feat = feat[:CONFIG['target_len']]
                else:
                    pad = CONFIG['target_len'] - len(feat)
                    feat = np.pad(feat, (0, pad), 'constant')
                return feat

            # 1. Always add original
            X_local.append(get_feat(data))
            Y_local.append(row['label'])

            # 2. Add Augmentation ONLY if requested (Training Set)
            if augment and row['label'] == 1: # Balance the 'Unsafe' class
                # Noise
                X_local.append(get_feat(noise(data)))
                Y_local.append(row['label'])
                # Pitch
                X_local.append(get_feat(pitch(data, sr)))
                Y_local.append(row['label'])

        except Exception as e:
            print(f"Error: {e}")
            
    return np.array(X_local), np.array(Y_local)

print("Processing Training Set (With Augmentation)...")
x_train, y_train = process_dataset(train_df, augment=True)

print("Processing Test Set (NO Augmentation)...")
x_test, y_test = process_dataset(test_df, augment=False)

# Check Shapes
print(f"Train Shape: {x_train.shape}")
print(f"Test Shape:  {x_test.shape}")

# Scale Data
# CRITICAL: Fit scaler ONLY on training data, then transform test
# This prevents "Scaler Leakage"
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape for CNN (Batch, Steps, Channels)
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# ==========================================
# 5. MODEL (DS-CNN)
# ==========================================
def build_dscnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, 10, strides=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    
    model.add(SeparableConv1D(64, 3, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    
    model.add(SeparableConv1D(128, 3, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(SeparableConv1D(128, 3, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = build_dscnn_model((CONFIG['target_len'], 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

# ==========================================
# 6. TRAINING
# ==========================================
checkpoint = ModelCheckpoint("best_tiny_model.keras", monitor='val_recall', mode='max', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = model.fit(
    x_train, y_train, 
    validation_data=(x_test, y_test),
    batch_size=CONFIG['batch_size'], 
    epochs=CONFIG['epochs'], 
    callbacks=[checkpoint, reduce_lr]
)

# ==========================================
# 7. EXPORT
# ==========================================
# Save Scaler
joblib.dump(scaler, 'scaler.gz')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("safety_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"--> Saved 'safety_model.tflite' ({len(tflite_model)/1024:.2f} KB)")