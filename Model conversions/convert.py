
import torch
import torch.nn as nn
import numpy as np
import onnx
import tensorflow as tf
import os
import onnx2tf 

# Import your model definition
from inference import TinySafetyNet, Config 

# ==========================================
# 1. LOAD PYTORCH MODEL
# ==========================================
print("Loading PyTorch model...")
model = TinySafetyNet()
model.load_state_dict(torch.load('tiny_safety_3class.pth', map_location='cpu'))
model.eval()

# ==========================================
# 2. EXPORT TO ONNX (SAFE MODE: OPSET 11)
# ==========================================
print("Exporting to ONNX (Opset 11)...")
dummy_input = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    verbose=False, 
    input_names=['input'], 
    output_names=['output'],
    
    # OPSET 11 is the most stable for TFLite conversion
    opset_version=11
)

# ==========================================
# 3. CONVERT TO TF (PYTHON API)
# ==========================================
print("Converting ONNX to TensorFlow...")

# We use the direct Python API to avoid subprocess crashes
# and explicitly disable the missing 'onnxsim'
onnx2tf.convert(
    input_onnx_file_path="model.onnx",
    output_folder_path="saved_model_tf",
    not_use_onnxsim=True,   # <--- FIX: Stop looking for onnxsim
    verbosity="info"
)

# ==========================================
# 4. CONVERT TO TFLITE & QUANTIZE
# ==========================================
print("\nConverting to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_tf")

# A. Standard Float32
tflite_model = converter.convert()
with open("safety_model.tflite", 'wb') as f:
    f.write(tflite_model)

# B. Int8 Quantization
print("Quantizing...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for _ in range(100):
        # Generate random data (1, 64, 64, 1)
        data = np.random.rand(1, 64, 64, 1).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32 
converter.inference_output_type = tf.float32

quant_model = converter.convert()

with open("safety_model_quant.tflite", 'wb') as f:
    f.write(quant_model)

# ==========================================
# 5. GENERATE C++ HEADER
# ==========================================
print("\nGenerating C Header for Arduino...")
os.system("xxd -i safety_model_quant.tflite > model_data.cc")

print("-" * 40)
print(f"PyTorch Size:   {os.path.getsize('tiny_safety_3class.pth')/1024:.2f} KB")
print(f"Quantized Size: {os.path.getsize('safety_model_quant.tflite')/1024:.2f} KB")
print("✅ DONE! Use 'model_data.cc' for your TinyML board.")
print("-" * 40)
