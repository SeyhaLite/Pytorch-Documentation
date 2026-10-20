# REQUIRES: gpu
"""ការវាស់ល្បឿនលើ GPU៖ data transfer, memory formats និង torch.compile។

ត្រូវការ៖ NVIDIA GPU (CUDA)។ នឹងត្រូវរំលងបើគ្មាន។

⚠ មិនទាន់បានសាកល្បងដោយអ្នកនិពន្ធ — គ្មាន GPU ក្នុង environment បង្កើតឯកសារ។
"""

import time

import torch
import torch.nn as nn

assert torch.cuda.is_available(), "ឯកសារនេះត្រូវការ CUDA GPU"
DEVICE = torch.device("cuda")


def bench(fn, reps: int = 50, warmup: int = 10) -> float:
    """វាស់ដោយត្រឹមត្រូវលើ GPU — ត្រូវ synchronize មុន និងក្រោយ។"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


print("=" * 72)
print("GPU:", torch.cuda.get_device_name(0))
print("=" * 72)

# ---------------------------------------------------------------------------
# ១. ⚠ កំហុសដែលជួបញឹកញាប់បំផុត: .item() / print() ក្នុង hot loop
# ---------------------------------------------------------------------------
print("\n១. Host-device synchronization overhead")
x = torch.randn(1024, 1024, device=DEVICE)
y = torch.randn(1024, 1024, device=DEVICE)


def no_sync():
    for _ in range(20):
        z = x @ y


def with_sync():
    for _ in range(20):
        z = x @ y
        _ = z.sum().item()          # បង្ខំឱ្យ CPU រង់ចាំ GPU


t_no = bench(no_sync, reps=10)
t_sync = bench(with_sync, reps=10)
print(f"  គ្មាន .item() ក្នុង loop : {t_no:>8.2f} ms")
print(f"  មាន .item() រាល់ iteration: {t_sync:>8.2f} ms")
print(f"  => {t_sync / t_no:.2f}x យឺតជាង ដោយសារការរង់ចាំ synchronize")

# ---------------------------------------------------------------------------
# ២. CPU <-> GPU transfer
# ---------------------------------------------------------------------------
print("\n២. Host-device transfer")
cpu_t = torch.randn(1024, 1024)
gpu_t = torch.randn(1024, 1024, device=DEVICE)

t_to_gpu = bench(lambda: cpu_t.to(DEVICE, non_blocking=True), reps=30)
t_to_cpu = bench(lambda: gpu_t.cpu(), reps=30)
mb = cpu_t.numel() * 4 / 1e6
print(f"  CPU -> GPU ({mb:.0f} MB) : {t_to_gpu:>7.2f} ms  (~{mb / (t_to_gpu / 1000):.0f} MB/s)")
print(f"  GPU -> CPU ({mb:.0f} MB) : {t_to_cpu:>7.2f} ms")
print("  យុទ្ធសាស្ត្រ: ផ្លាស់ទីទិន្នន័យម្ដងគត់ ហើយធ្វើការទាំងអស់លើ GPU។")
print("  pin_memory=True + non_blocking=True ជួយឱ្យការផ្លាស់ទីត្រួតគ្នាជាមួយការគណនា។")

# ---------------------------------------------------------------------------
# ៣. channels_last សម្រាប់ CNN
# ---------------------------------------------------------------------------
print("\n៣. Memory format សម្រាប់ CNN")
conv = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
).to(DEVICE)

img_nchw = torch.randn(64, 64, 56, 56, device=DEVICE)
img_nhwc = img_nchw.to(memory_format=torch.channels_last)
conv_cl = nn.Sequential(*conv).to(memory_format=torch.channels_last)


def run_nchw():
    with torch.no_grad():
        return conv(img_nchw)


def run_nhwc():
    with torch.no_grad():
        return conv_cl(img_nhwc)


t_nchw = bench(run_nchw, reps=20)
t_nhwc = bench(run_nhwc, reps=20)
print(f"  NCHW (contiguous)  : {t_nchw:>7.2f} ms")
print(f"  channels_last      : {t_nhwc:>7.2f} ms   ({t_nchw / t_nhwc:.2f}x)")
print("  ⚠ អត្ថប្រយោជន៍ប្រែប្រួលតាម GPU និង cuDNN version។ វាស់ដោយខ្លួនឯង។")

# ---------------------------------------------------------------------------
# ៤. TF32 និង float32 matmul precision
# ---------------------------------------------------------------------------
print("\n៥. TF32 matmul")
print(f"  allow_tf32 (matmul) បច្ចុប្បន្ន: {torch.backends.cuda.matmul.allow_tf32}")
print(f"  allow_tf32 (cudnn)  បច្ចុប្បន្ន: {torch.backends.cudnn.allow_tf32}")
print(f"  float32_matmul_precision: {torch.get_float32_matmul_precision()}")

a = torch.randn(2048, 2048, device=DEVICE)
b = torch.randn(2048, 2048, device=DEVICE)

orig = torch.get_float32_matmul_precision()
for mode in ["highest", "high", "medium"]:
    torch.set_float32_matmul_precision(mode)
    t = bench(lambda: a @ b, reps=20)
    print(f"  precision='{mode}': {t:>7.2f} ms")
torch.set_float32_matmul_precision(orig)
print("  'highest' = float32 ពេញលេញ | 'high' = TF32 | 'medium' = bfloat16")

# ---------------------------------------------------------------------------
# ៕. torch.compile
# ---------------------------------------------------------------------------
print("\n៕. torch.compile")
model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)).to(DEVICE)
inp = torch.randn(256, 512, device=DEVICE)

t_eager = bench(lambda: model(inp), reps=50)
print(f"  eager       : {t_eager:>7.3f} ms")

for mode in ["default", "reduce-overhead", "max-autotune"]:
    try:
        compiled = torch.compile(model, mode=mode)
        with torch.no_grad():
            compiled(inp)                        # កំដៅ (compile កើតឡើងនៅទីនេះ)
            compiled(inp)
        t = bench(lambda: compiled(inp), reps=50)
        print(f"  compile[{mode:<15}]: {t:>7.3f} ms  ({t_eager / t:.2f}x)")
    except Exception as e:
        print(f"  compile[{mode:<15}]: បរាជ័យ {type(e).__name__}: {str(e)[:44]}")

print("\n⚠ លទ្ធផលទាំងអស់ជាក់លាក់ចំពោះ GPU នេះ និង PyTorch", torch.__version__)
print("OK — GPU performance probes ដំណើរការ")
