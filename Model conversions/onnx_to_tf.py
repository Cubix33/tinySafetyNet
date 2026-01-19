import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("tiny_safety_3class_fixed.onnx")

tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_safety_model")

print("✅ TensorFlow SavedModel created")
