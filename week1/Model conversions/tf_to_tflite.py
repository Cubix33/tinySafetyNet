import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("tf_safety_model")

# Keep float first (safer)
converter.optimizations = []

tflite_model = converter.convert()

with open("tiny_safety_3class.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model created")
