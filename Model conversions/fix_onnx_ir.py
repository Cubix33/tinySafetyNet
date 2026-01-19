import onnx

model = onnx.load("tiny_safety_3class.onnx")

# Force older, compatible IR
model.ir_version = 7

onnx.save(model, "tiny_safety_3class_fixed.onnx")

print("✅ ONNX IR version fixed")
