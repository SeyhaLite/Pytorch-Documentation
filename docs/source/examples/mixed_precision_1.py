"""Mixed precision training ដោយប្រើ torch.amp។

ដំណើរការលើ៖ CPU — ប្រើ device_type="cpu" ជាមួយ bfloat16។
ផ្នែក CUDA (float16 + GradScaler) មាននៅក្នុងកូដ ប៉ុន្តែនឹងត្រូវរំលងបើគ្មាន GPU។
"""

import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# ១. ជ្រើសរើស dtype និង device_type ត្រឹមត្រូវ
# ---------------------------------------------------------------------------
print("=" * 70)
print("ការកំណត់ mixed precision")
print("=" * 70)
if DEVICE.type == "cuda":
    device_type = "cuda"
    amp_dtype = torch.float16        # ឬ torch.bfloat16 សម្រាប់ GPU Ampere+
    use_scaler = True
else:
    device_type = "cpu"
    amp_dtype = torch.bfloat16       # CPU គាំទ្រ bfloat16 ល្អជាង float16
    use_scaler = False

print(f"  device      : {DEVICE}")
print(f"  device_type : {device_type}")
print(f"  amp dtype   : {amp_dtype}")
print(f"  GradScaler  : {use_scaler}")
print(f"  CPU គាំទ្រ bfloat16: {torch.cpu._is_avx512_bf16_supported() if hasattr(torch.cpu, '_is_avx512_bf16_supported') else 'unknown'}")

# ---------------------------------------------------------------------------
# ២. ហេតុអ្វី fp16 ត្រូវការ GradScaler ប៉ុន្តែ bf16 មិនត្រូវការ?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ដែនកំណត់នៃ dtype")
print("=" * 70)
print(f"{'dtype':<12}{'ធាតុតូចបំផុត > 0':<22}{'ធាតុធំបំផុត':<18}{'ខ្ទង់សំខាន់'}")
for dt in [torch.float32, torch.float16, torch.bfloat16]:
    info = torch.finfo(dt)
    print(f"{str(dt).replace('torch.', ''):<12}{info.tiny:<22.3e}{info.max:<18.3e}{info.bits} bits")

print("""
  float16 : ដែនតូច (max ~65504) ប៉ុន្តែភាពជាក់លាក់ល្អក្នុងដែននោះ
            -> gradient តូចៗ (ឧ. 1e-8) ក្លាយជា 0 (underflow)
            -> ត្រូវការ GradScaler ដើម្បីគុណ loss ឡើងជាមុន
  bfloat16: ដែនដូច float32 ប៉ុន្តែភាពជាក់លាក់ទាប (ខ្ទង់សំខាន់តិច)
            -> មិនត្រូវការ GradScaler ទេ ព្រោះគ្មាន underflow
""")

# បង្ហាញ underflow ពិតប្រាកដ
tiny = torch.tensor(1e-8, dtype=torch.float32)
print(f"  1e-8 ជា float32 : {tiny.item():.3e}")
print(f"  1e-8 ជា float16 : {tiny.to(torch.float16).item():.3e}   <- underflow ទៅ 0!")
print(f"  1e-8 ជា bfloat16: {tiny.to(torch.bfloat16).item():.3e}   <- នៅរក្សាបាន")
assert tiny.to(torch.float16).item() == 0.0
assert tiny.to(torch.bfloat16).item() != 0.0

big = torch.tensor(70000.0, dtype=torch.float32)
print(f"  70000 ជា float16 : {big.to(torch.float16).item()}   <- overflow ទៅ inf!")
print(f"  70000 ជា bfloat16: {big.to(torch.bfloat16).item():.1f}   <- នៅត្រឹមត្រូវ")
assert torch.isinf(big.to(torch.float16))

# ---------------------------------------------------------------------------
# ៣. វដ្ដបណ្ដុះបណ្ដាលជាមួយ autocast
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("វដ្ដបណ្ដុះបណ្ដាលជាមួយ torch.amp.autocast")
print("=" * 70)

model = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 10),
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
# GradScaler ត្រូវបានបង្កើតជាមួយ device ជាក់លាក់ (PyTorch 2.4+)
scaler = GradScaler(device=DEVICE.type, enabled=use_scaler)

x = torch.randn(256, 128, device=DEVICE)
y = torch.randint(0, 10, (256,), device=DEVICE)

dtypes_seen = set()
for step in range(3):
    optimizer.zero_grad(set_to_none=True)

    # autocast ប្ដូរ op មួយចំនួនទៅ dtype ទាបដោយស្វ័យប្រវត្តិ
    with autocast(device_type=device_type, dtype=amp_dtype):
        logits = model(x)
        loss = loss_fn(logits, y)
    dtypes_seen.add(str(logits.dtype))

    # scaler.scale(loss) គុណ loss ឡើង; scaler.step() ធ្វើ unscale + optimizer.step()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"  step {step + 1}: loss={loss.item():.4f}  logits.dtype={logits.dtype}  "
          f"weight.dtype={model[0].weight.dtype}")

print(f"\n  dtypes ដែលបានឃើញក្នុង autocast region: {sorted(dtypes_seen)}")
print("  ⚠ weight (master copy) នៅតែ float32 — នេះគឺជាចេតនា។")
print("     autocast ប្ដូរតែ *ប្រតិបត្តិការ* មិនមែន parameter ទេ។")
assert model[0].weight.dtype == torch.float32

# ---------------------------------------------------------------------------
# ។. អ្វីដែល autocast ប្ដូរ និងអ្វីដែលវាមិនប្ដូរ
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("អ្វីដែល autocast រក្សាជា float32")
print("=" * 70)
with autocast(device_type=device_type, dtype=amp_dtype):
    matmul_out = torch.matmul(torch.randn(8, 64, device=DEVICE), torch.randn(64, 32, device=DEVICE))
    sum_out = torch.randn(64, device=DEVICE).sum()
    softmax_out = torch.softmax(torch.randn(8, 10, device=DEVICE), dim=1)
    log_out = torch.log(torch.rand(8, device=DEVICE) + 0.5)
    mse_out = nn.functional.mse_loss(torch.randn(8, device=DEVICE), torch.randn(8, device=DEVICE))

print(f"{'ប្រតិបត្តិការ':<22}{'dtype លទ្ធផល':<16}")
print("-" * 44)
for name, out in [("matmul", matmul_out), ("sum (reduction)", sum_out),
                  ("softmax", softmax_out), ("log", log_out), ("mse_loss", mse_out)]:
    print(f"{name:<22}{str(out.dtype).replace('torch.', ''):<16}")
print("""
  ទូទៅ៖ matmul/conv ដំណើរការក្នុង dtype ទាប (លឿន)
         reduction, loss, softmax, log រក្សា float32 (ស្ថិរភាពលេខ)
""")

# ---------------------------------------------------------------------------
# ៕. ប្រៀបធ្វេបល្បឿន (លើម៉ាស៊ីននេះតែប៉ុណ្ណោះ)
# ---------------------------------------------------------------------------
print("=" * 70)
print("ការវាស់ល្បឿនលើម៉ាស៊ីននេះ (មិនមែន benchmark ជាសកលទេ)")
print("=" * 70)


def train_step(use_amp: bool) -> None:
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with autocast(device_type=device_type, dtype=amp_dtype):
            loss = loss_fn(model(x), y)
    else:
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward() if use_amp else loss.backward()
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def bench(fn, reps: int = 20) -> float:
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


t_fp32 = bench(lambda: train_step(False))
t_amp = bench(lambda: train_step(True))
print(f"  float32 train step : {t_fp32:>7.2f} ms")
print(f"  {str(amp_dtype).replace('torch.', ''):<8} train step : {t_amp:>7.2f} ms")
ratio = t_fp32 / t_amp
print(f"  អនុបាត              : {ratio:.2f}x")
if DEVICE.type == "cpu":
    print("  ⚠ លើ CPU ល្បឿនអាចមិនប្រសើរឡើង ឬអាចយឺតជាង អាស្រ័យលើ AVX512-BF16.")
    print("    អត្ថប្រយោជន៍ធំរបស់ mixed precision គឺនៅលើ GPU។")
else:
    print("  ⚠ លទ្ធផលប្រែប្រួលតាម GPU, batch size និង model architecture។")

# ---------------------------------------------------------------------------
# ៖. កំហុសធម្មតា
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("កំហុសធម្មតា")
print("=" * 70)

# ក) បង្កើត GradScaler បើមិនចាំបាច់ (CPU)
print("  ក) GradScaler(enabled=False) លើ CPU/bf16 — scaler ក្លាយជា pass-through")
s_disabled = GradScaler(device=DEVICE.type, enabled=False)
print(f"     is_enabled() = {s_disabled.is_enabled()}")
assert not s_disabled.is_enabled()

# ខ) ភ្លេច scaler.update()
print("  ខ) ភ្លេច scaler.update() -> scale factor មិនដែលកែតម្រូវ អាចបណ្ដាលឱ្យ NaN")

# គ) ផ្លាស់ទី tensor ចូល/ចេញ autocast ដោយដៃ
print("  គ) កុំ .half() parameter ដោយដៃ — ប្រើ autocast ឱ្យវាគ្រប់គ្រង")
try:
    with autocast(device_type=device_type, dtype=amp_dtype):
        _ = model[0].weight.half().sum()
    print("     .half() ដោយដៃដំណើរការ ប៉ុន្តែវាបំផ្លាញ master weight បើអ្នក assign វាវិញ")
except Exception as e:
    print(f"     {type(e).__name__}: {str(e)[:50]}")

# ឃ) NaN/Inf detection
print("  ឃ) ពិនិត្យ NaN ជាទម្លាប់:")
probe = torch.tensor([1.0, float("nan"), float("inf")])
print(f"     torch.isnan(x).any() = {bool(torch.isnan(probe).any())}")
print(f"     torch.isfinite(x).all() = {bool(torch.isfinite(probe).all())}")
print(f"     GradScaler រកឃើញ inf ដោយខ្លួនឯង ហើយរំលង step នោះ")

print("\nOK — mixed precision ដំណើរការត្រឹមត្រូវលើ", DEVICE)
