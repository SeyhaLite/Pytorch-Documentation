"""ជ្រើសរើស device (CPU / CUDA / MPS) និងផ្លាស់ទី tensor ឬ model។

ដំណើរការលើ៖ CPU — ផ្នែក GPU/MPS នឹងត្រូវរំលងដោយស្វ័យប្រវត្តិបើមិនមាន។
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# ១. រក device ដែលល្អបំផុត (រូបមន្តស្តង់ដារ)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("device ដែលបានជ្រើសរើស :", device)
print("torch.cuda.is_available() :", torch.cuda.is_available())
if hasattr(torch.backends, "mps"):
    print("torch.backends.mps.is_available() :", torch.backends.mps.is_available())

# វិធីខ្លី (PyTorch 2.4+)៖ torch.get_device_module()
dev_mod = torch.get_device_module(device)
print("torch.get_device_module(device) :", dev_mod.__name__)

# ---------------------------------------------------------------------------
# ២. បង្កើត tensor លើ device ជាក់លាក់
# ---------------------------------------------------------------------------
print("-" * 55)
x = torch.randn(3, 4, device=device)
print("torch.randn(3, 4, device=device) -> device:", x.device)

y = torch.zeros(2, 2)                 # កើតលើ CPU ដោយស្រាប់
print("torch.zeros(2,2) ដោយមិនបញ្ជាក់   -> device:", y.device)

y_moved = y.to(device)
print("បន្ទាប់ y.to(device)              -> device:", y_moved.device)
assert y_moved.device.type == device.type

# ---------------------------------------------------------------------------
# ៣. ផ្លាស់ទី model ទាំងមូល
# ---------------------------------------------------------------------------
print("-" * 55)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
print("មុន .to(device) :", next(model.parameters()).device)
model = model.to(device)
print("បន្ទាប់ .to(device):", next(model.parameters()).device)

# forward pass ត្រូវតែប្រើ input នៅលើ device ដូចគ្នា
out = model(torch.randn(5, 4, device=device))
print("output shape:", tuple(out.shape), "| device:", out.device)
assert out.shape == (5, 2)
assert out.device.type == device.type

# ---------------------------------------------------------------------------
# ៤. កំហុសធម្មតា៖ device មិនត្រូវគ្នា
# ---------------------------------------------------------------------------
print("-" * 55)
cpu_tensor = torch.ones(2, device="cpu")
if device.type != "cpu":
    try:
        _ = cpu_tensor + torch.ones(2, device=device)
    except RuntimeError as e:
        print("RuntimeError (ត្រឹមត្រូវ):", str(e).splitlines()[0][:80])
else:
    print("(កំពុងដំណើរការលើ CPU — មិនអាចបង្ហាញកំហុស device mismatch បាន)")
    print("ពេលមាន GPU កំហុសនឹងថា: Expected all tensors to be on the same device")

# ---------------------------------------------------------------------------
# ៥. ព័ត៌មានអំពី GPU (បើមាន)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    print("-" * 55)
    print("GPU count        :", torch.cuda.device_count())
    print("GPU name         :", torch.cuda.get_device_name(0))
    print("CUDA capability  :", torch.cuda.get_device_capability(0))
    print("allocated memory :", torch.cuda.memory_allocated(0) / 1e6, "MB")
    print("reserved memory  :", torch.cuda.memory_reserved(0) / 1e6, "MB")

    # ផ្លាស់ទីត្រឡប់មក CPU (ចាំបាច់មុន save ឬ .numpy())
    back = x.cpu()
    print("x.cpu() -> device:", back.device)

    # យកទិន្នន័យចេញពី GPU មក Python
    print("x.cpu().tolist()[:1] :", back.tolist()[0][:2])
else:
    print("-" * 55)
    print("គ្មាន CUDA GPU — ខ្លឹមសារខាងលើនឹងដំណើរការដូចគ្នានៅពេលអ្នកមាន GPU")

# ---------------------------------------------------------------------------
# ៦. ចំណាំសំខាន់អំពី .numpy()
# ---------------------------------------------------------------------------
print("-" * 55)
t = torch.arange(4, dtype=torch.float32)
arr = t.numpy()
print("torch tensor -> numpy:", arr, type(arr).__name__)
arr[0] = 999.0
print("កែ numpy[0]=999 -> tensor =", t.tolist(), "(ចែក memory រួមគ្នា!)")
assert t[0].item() == 999.0

# tensor នៅលើ GPU មិនអាច .numpy() ដោយផ្ទាល់ទេ — ត្រូវ .cpu() ជាមុន
print("ដូច្នេះសម្រាប់ GPU:  tensor.detach().cpu().numpy()")

print("-" * 55)
print("OK — device handling ដំណើរការត្រឹមត្រូវ")
