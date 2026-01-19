import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="tiny_safety_3class.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

dummy_input = np.random.rand(1, 1, 64, 64).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print("Output shape:", output.shape)
print("Output:", output)
