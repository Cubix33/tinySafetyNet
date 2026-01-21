with open("tiny_safety_3class_int8.tflite", "rb") as f:
    data = f.read()

with open("model.cc", "w") as f:
    f.write("unsigned char tiny_safety_3class_int8_tflite[] = {\n")

    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}, ")
        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n")
    f.write(f"unsigned int tiny_safety_int8_tflite_len = {len(data)};\n")
