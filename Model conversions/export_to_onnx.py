import torch
import torch.nn as nn

# -----------------------------
# COPY MODEL DEFINITIONS EXACTLY
# -----------------------------

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
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = DC_AC_Block(32)
        self.conv1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.dropout1 = nn.Dropout(0.2)

        self.layer2 = DC_AC_Block(64)
        self.dropout2 = nn.Dropout(0.2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.dropout1(self.conv1(x))
        x = self.dropout2(self.layer2(x))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


# -----------------------------
# LOAD WEIGHTS PROPERLY
# -----------------------------

model = TinySafetyNet()
state_dict = torch.load("tiny_safety_3class.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# EXPORT TO ONNX
# -----------------------------

dummy_input = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model,
    dummy_input,
    "tiny_safety_3class.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("✅ ONNX export successful: tiny_safety_3class.onnx")
