import os
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'root_dir': 'downloads',
    'sample_rate': 22050,
    'duration': 3.0,
    'n_mfcc': 40,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50
}

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"--> Using device: GPU ({len(gpus)} found)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("--> Using device: CPU")

# ==========================================
# 2. DATA LOADING & LABEL CORRECTION
# ==========================================
def parse_datasets(root_dir):
    file_paths = []
    labels = []
    
    # --- A. RAVDESS ---
    ravdess_map = {
        '01': 'neutral', '02': 'neutral', 
        '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fear', 
        '07': 'disgust', '08': 'surprise'
    }
    ravdess_files = glob.glob(os.path.join(root_dir, 'ravdess', '**', '*.wav'), recursive=True)
    for f in ravdess_files:
        parts = os.path.basename(f).split('-')
        if len(parts) >= 3:
            code = parts[2]
            if code in ravdess_map:
                file_paths.append(f)
                labels.append(ravdess_map[code])
                
    # --- B. CREMA-D ---
    crema_map = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}
    crema_files = glob.glob(os.path.join(root_dir, 'cremad', '*.wav'))
    for f in crema_files:
        try:
            part = os.path.basename(f).split('_')[2]
            if part in crema_map:
                file_paths.append(f)
                labels.append(crema_map[part])
        except IndexError: pass

    # --- C. TESS ---
    tess_files = glob.glob(os.path.join(root_dir, 'tess', '**', '*.wav'), recursive=True)
    for f in tess_files:
        filename = os.path.basename(f)
        emotion = filename.split('_')[-1].split('.')[0].lower()
        if emotion == 'ps': emotion = 'surprise'
        if emotion == 'fearful': emotion = 'fear'
        file_paths.append(f)
        labels.append(emotion)

    # --- D. SAVEE ---
    savee_map = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise'}
    savee_files = glob.glob(os.path.join(root_dir, 'savee', '*.wav'))
    for f in savee_files:
        filename = os.path.basename(f)
        code = filename[0]
        if filename[1] == 'a' or filename[1] == 'u': code = filename[:2]
        if code in savee_map:
            file_paths.append(f)
            labels.append(savee_map[code])
            
    return pd.DataFrame({'path': file_paths, 'label': labels})

print("--> Scanning datasets...")
df = parse_datasets(CONFIG['root_dir'])

# Final Label Cleanup
df['label'] = df['label'].replace({'fearful': 'fear', 'calm': 'neutral'})
print(f"--> Total files: {len(df)}")

# Encode Labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])
classes = list(le.classes_)
CONFIG['num_classes'] = len(classes)
print(f"--> Classes found: {classes}")

# Split Data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# ==========================================
# 3. TF DATA PIPELINE
# ==========================================
TARGET_LENGTH = int(CONFIG['sample_rate'] * CONFIG['duration'])

def preprocess_audio(file_path, label):
    path_str = file_path.numpy().decode('utf-8')
    try:
        y, sr = librosa.load(path_str, sr=CONFIG['sample_rate'])
    except Exception:
        y = np.zeros(TARGET_LENGTH)
        sr = CONFIG['sample_rate']

    y, _ = librosa.effects.trim(y)
    if len(y) > TARGET_LENGTH: y = y[:TARGET_LENGTH]
    else: y = np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG['n_mfcc'])
    mfcc = mfcc.astype(np.float32)
    return mfcc, np.int32(label)

def create_dataset(dataframe, is_training=True):
    paths = dataframe['path'].values
    labels = dataframe['label_id'].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def _py_mapper(x, y):
        mfcc, label = tf.py_function(preprocess_audio, [x, y], [tf.float32, tf.int32])
        return mfcc, label

    ds = ds.map(_py_mapper, num_parallel_calls=tf.data.AUTOTUNE)

    def _fix_shape(mfcc, label):
        mfcc.set_shape([CONFIG['n_mfcc'], None]) 
        label.set_shape([]) 
        return mfcc, label

    ds = ds.map(_fix_shape, num_parallel_calls=tf.data.AUTOTUNE)
    # Add Channel Dim (For CNN): (40, 130) -> (40, 130, 1)
    ds = ds.map(lambda x, y: (tf.expand_dims(x, -1), y), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training: ds = ds.shuffle(1000)
    ds = ds.batch(CONFIG['batch_size'])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_df, is_training=True)
val_ds = create_dataset(val_df, is_training=False)

for x, y in train_ds.take(1):
    input_shape = x.shape[1:] 
    print(f"--> Model Input Shape: {input_shape}")

# ==========================================
# 4. MODEL: DS-CNN (Depthwise Separable CNN)
# ==========================================
# This replaces the LSTM/Attention architecture
def create_dscnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # 1. Stem (Standard Conv) - Downsample frequency/time initially
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 2. DS-CNN Block 1
    # Depthwise: spatial filtering per channel
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Pointwise: mixing channels (1x1 Conv)
    x = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3. DS-CNN Block 2 (Strided for reduction)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 4. Global Pooling (The "TinyML" way to handle time)
    # Reduces (Height, Width, Channels) -> (Channels)
    # This acts as our "Attention" - finding if the feature exists anywhere
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # 5. Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DS-CNN_Tiny")
    return model

model = create_dscnn_model(input_shape, CONFIG['num_classes'])
model.compile(
    optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ==========================================
# 5. TRAINING LOOP
# ==========================================
checkpoint = callbacks.ModelCheckpoint(
    'women_safety_dscnn.keras', 
    monitor='val_accuracy', 
    save_best_only=True,
    verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

print("\n--> Starting Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG['epochs'],
    callbacks=[checkpoint, reduce_lr, early_stop]
)

np.save('classes.npy', classes)
print("\n--> Training Complete.")

# ==========================================
# 6. TFLITE CONVERSION (Optimized)
# ==========================================
print("\n--> Converting to TFLite (DS-CNN)...")

# Load best model
model = models.load_model('women_safety_dscnn.keras')

# 1. Standard TFLite (Float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Note: No 'SELECT_TF_OPS' needed because DS-CNN uses standard ops!
tflite_model = converter.convert()
with open('women_safety_dscnn.tflite', 'wb') as f:
    f.write(tflite_model)
print("    Saved: women_safety_dscnn.tflite")

# 2. Quantized TFLite (Float16) - Recommended
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
tflite_model_f16 = converter_f16.convert()

with open('women_safety_dscnn_f16.tflite', 'wb') as f:
    f.write(tflite_model_f16)
print("    Saved: women_safety_dscnn_f16.tflite")

# ==========================================
# 7. INFERENCE EXAMPLE
# ==========================================
def check_audio_safety_tf(audio_path):
    classes = np.load('classes.npy')
    
    # Load Interpreter
    interpreter = tf.lite.Interpreter(model_path="women_safety_dscnn_f16.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess
    y, sr = librosa.load(audio_path, sr=CONFIG['sample_rate'])
    if len(y) > TARGET_LENGTH: y = y[:TARGET_LENGTH]
    else: y = np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG['n_mfcc'])
    mfcc = mfcc.astype(np.float32)
    
    # Ensure correct width (130 frames)
    # The DS-CNN expects fixed size input because of Flatten/Dense layers logic
    target_frames = input_details[0]['shape'][2] 
    if mfcc.shape[1] > target_frames: mfcc = mfcc[:, :target_frames]
    else: mfcc = np.pad(mfcc, ((0,0), (0, target_frames - mfcc.shape[1])))

    # Add Batch/Channel dims
    inp = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)
    
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    pred_idx = np.argmax(output_data)
    emotion = classes[pred_idx]
    
    UNSAFE_EMOTIONS = ['fear', 'angry'] 
    status = "UNSAFE 🚨" if emotion in UNSAFE_EMOTIONS else "SAFE ✅"
    
    print(f"Input: {audio_path}")
    print(f"Emotion: {emotion}")
    print(f"Status: {status}")

# check_audio_safety_tf("test_file.wav")