"""CNN សម្រាប់រូបភាព + តាមដាន shape គ្រប់ជាន់។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU — ប៉ុន្តែយឺតជាងលើ GPU ច្រើន)

រូបមន្ត Conv2d output size:
    out = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleCNN(nn.Module):
    """CNN តូចសម្រាប់រូបភាព 32x32 ពណ៌ (3 channels)។"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Conv block 1: 32x32 -> 16x16
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # (32+2-3)/1+1 = 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32 -> 16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16 -> 8
        )
        # Adaptive pooling ធ្វើឱ្យម៉ូដែលអាចទទួល input គ្រប់ទំហំ
        self.pool = nn.AdaptiveAvgPool2d(1)               # (8,8) -> (1,1)
        self.flatten = nn.Flatten()                       # (N,32,1,1) -> (N,32)
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


model = SimpleCNN(num_classes=10)
model.eval()

# ---------------------------------------------------------------------------
# ១. តាមដាន shape គ្រប់ជាន់ដោយប្រើ forward hooks
# ---------------------------------------------------------------------------
images = torch.randn(4, 3, 32, 32)     # (batch=4, channels=3, H=32, W=32)
print("input:", tuple(images.shape))
print("-" * 66)

shapes: dict[str, tuple] = {}


def make_hook(name: str):
    def hook(module, inputs, output):
        if torch.is_tensor(output):
            shapes[name] = tuple(output.shape)
            print(f"{name:<42} -> {tuple(output.shape)}")
    return hook


for name, module in model.named_modules():
    if name and not list(module.children()):      # leaf modules តែប៉ុណ្ណោះ
        module.register_forward_hook(make_hook(name))

with torch.no_grad():
    logits = model(images)

print("-" * 66)
print("logits សរុប:", tuple(logits.shape))
assert logits.shape == (4, 10)

# ---------------------------------------------------------------------------
# ២. ផ្ទៀងផ្ទាត់រូបមន្ត Conv2d
# ---------------------------------------------------------------------------
print("-" * 66)
print("ផ្ទៀងផ្ទាត់រូបមន្ត conv output size:")
cases = [
    # (in_size, kernel, stride, padding, រំពឹង)
    (32, 3, 1, 1, 32),
    (32, 3, 1, 0, 30),
    (32, 3, 2, 1, 16),
    (32, 5, 1, 2, 32),
    (16, 3, 1, 1, 16),
    (8, 3, 2, 1, 4),
]
for in_size, k, s, p, expect in cases:
    formula = (in_size + 2 * p - (k - 1) - 1) // s + 1
    conv = nn.Conv2d(1, 1, kernel_size=k, stride=s, padding=p, bias=False)
    real = conv(torch.zeros(1, 1, in_size, in_size)).shape[-1]
    status = "OK" if (formula == real == expect) else "ខុស"
    print(f"  in={in_size:>2} k={k} s={s} p={p} -> រូបមន្ត={formula:>2} ពិត={real:>2} {status}")
    assert formula == real == expect

# ---------------------------------------------------------------------------
# ៣. ចំនួន parameters
# ---------------------------------------------------------------------------
print("-" * 66)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters សរុប : {total:,}")
print(f"trainable      : {trainable:,}")
assert total == trainable

print("\nការបំបែកតាមជាន់:")
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name:<14} {n:>8,}")

# ---------------------------------------------------------------------------
# ៤. ទទួល input ទំហំផ្សេង (ដោយសារ AdaptiveAvgPool2d)
# ---------------------------------------------------------------------------
print("-" * 66)
for size in [(4, 3, 32, 32), (4, 3, 64, 64), (2, 3, 48, 48)]:
    with torch.no_grad():
        out = model(torch.randn(*size))
    print(f"  input {size} -> logits {tuple(out.shape)}")
    assert out.shape[1] == 10

print("-" * 66)
print("OK — CNN និងការគណនា shape ទាំងអស់ត្រឹមត្រូវ")
