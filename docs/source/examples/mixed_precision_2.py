# REQUIRES: gpu
"""Mixed precision លើ GPU៖ float16 + GradScaler និង bfloat16។

ត្រូវការ៖ NVIDIA GPU (CUDA)។ ឯកសារនេះនឹងត្រូវរំលងដោយស្វ័យប្រវត្តិ
បើ torch.cuda.is_available() ត្រឡប់ False។

⚠ មិនទាន់បានសាកល្បងដោយអ្នកនិពន្ធ — គ្មាន GPU ក្នុង environment បង្កើតឯកសារ។
   កូដនេះផ្អែកលើ API ដែលបានផ្ទៀងផ្ទាត់ថាមានក្នុង PyTorch 2.14។
"""

import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

assert torch.cuda.is_available(), "ឯកសារនេះត្រូវការ CUDA GPU"

torch.manual_seed(0)
DEVICE = torch.device("cuda")


class MLP(nn.Module):
    def __init__(self, d: int = 1024, depth: int = 6) -> None:
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, d), nn.ReLU()]
        layers += [nn.Linear(d, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run(dtype: torch.Tensor, use_scaler: bool, steps: int = 30) -> dict:
    model = MLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(device="cuda", enabled=use_scaler)

    x = torch.randn(512, 1024, device=DEVICE)
    y = torch.randint(0, 10, (512,), device=DEVICE)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dtypes = set()

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
            loss = loss_fn(logits, y)
        dtypes.add(str(logits.dtype))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "loss": loss.item(),
        "seconds": elapsed,
        "logits_dtype": sorted(dtypes),
        "scale": scaler.get_scale(),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


print("=" * 72)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0),
      "(>= 8.0 គាំទ្រ bfloat16 ដោយ hardware)")
print("=" * 72)

torch.cuda.reset_peak_memory_stats()
results = {}
for label, dtype, use_scaler in [
    ("float32 (គ្មាន AMP)", torch.float32, False),
    ("float16 + GradScaler", torch.float16, True),
    ("bfloat16", torch.bfloat16, False),
]:
    torch.cuda.reset_peak_memory_stats()
    r = run(dtype, use_scaler)
    results[label] = r
    print(f"\n{label}")
    print(f"  loss ចុងក្រោយ     : {r['loss']:.4f}")
    print(f"  រយៈពេល 30 steps  : {r['seconds']:.2f}s")
    print(f"  logits dtype     : {r['logits_dtype']}")
    print(f"  peak GPU memory  : {r['peak_mem_mb']:.1f} MB")
    if use_scaler:
        print(f"  GradScaler scale : {r['scale']}")

print("\n" + "=" * 72)
print("⚠ លេខខាងលើវាស់លើ GPU របស់អ្នកនៅពេលនេះ។ វាមិនមែនជា benchmark")
print("  ជាសកលទេ — លទ្ធផលប្រែប្រួលតាម architecture, batch size និង driver។")
print("=" * 72)

# GradScaler គួរតែបង្កើន scale ដោយស្វ័យប្រវត្តិបើគ្មាន inf
assert results["float16 + GradScaler"]["scale"] > 0
print("OK — mixed precision លើ GPU ដំណើរការ")
