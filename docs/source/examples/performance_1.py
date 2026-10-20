"""ការបង្កើនល្បឿន៖ vectorization, dtype, contiguity និងការវែកញែក bottleneck។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

⚠ ចំណាំអំពីលេខនៅក្នុង output៖ រាល់រយៈពេលទាំងអស់ត្រូវបាន **វាស់លើម៉ាស៊ីន
ដែលកំពុងរត់ឯកសារនេះ**។ វាមិនមែនជា benchmark ជាសកលទេ — hardware, ចំនួន
thread និងកំណែ PyTorch នឹងផ្លាស់ប្ដូរលទ្ធផល។ សូមវាស់ដោយខ្លួនឯងជានិច្ច។
"""

import time

import torch
import torch.nn as nn

torch.manual_seed(0)
REPS = 5


def measure(fn, reps: int = REPS) -> float:
    """វាស់រយៈពេលមធ្យមគិតជាវិនាទី ដោយមាន warmup។"""
    fn()                                  # warmup (JIT / cache / thread pool)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# ១. Vectorization ជាង Python for-loop — ភាពខុសគ្នាធំបំផុត
# ---------------------------------------------------------------------------
print("=" * 74)
print("១. Vectorization vs Python loop")
print("=" * 74)
big = torch.randn(2000, 2000)


def loop_sum() -> float:
    total = 0.0
    for i in range(big.shape[0]):
        total += big[i].sum().item()      # .item() រាល់ជួរ = ប្ដូរ GPU/CPU sync
    return total


def vector_sum() -> float:
    return big.sum().item()


t_loop = measure(loop_sum, reps=2)
t_vec = measure(vector_sum, reps=5)
print(f"  Python loop + .item() រាល់ជួរ : {t_loop * 1000:>9.1f} ms")
print(f"  tensor.sum() (vectorized)   : {t_vec * 1000:>9.3f} ms")
print(f"  លឿនជាងប្រហែល {t_loop / max(t_vec, 1e-9):,.0f}x")
assert abs(loop_sum() - vector_sum()) < 1e-2

# per-row operations ក៏គួរប្រើ dim= ជំនួស loop — ប៉ុន្តែប្រយ័ត្ន dim!
print("\n  ⚠ ចំណុចដែលអ្នកចាប់ផ្ដើមច្រឡំញឹកញាប់: dim=N មានន័យថា 'វិមាត្រទី N ត្រូវបាត់បង់'")
print(f"    big.shape = {tuple(big.shape)}")
print(f"    big.mean(dim=0).shape = {tuple(big.mean(dim=0).shape)}  <- មធ្យមភាគតាមជួរ (ជួរបាត់បង់)")
print(f"    big.mean(dim=1).shape = {tuple(big.mean(dim=1).shape)}  <- មធ្យមភាគតាមឈរ (ឈរបាត់បង់)")

# big[i] គឺជួរទី i (shape (2000,)) ដូច្នេះ big[i].mean() ត្រូវនឹង mean(dim=1)
row_means_loop = torch.stack([big[i].mean() for i in range(big.shape[0])])
row_means = big.mean(dim=1)
assert torch.allclose(row_means, row_means_loop, rtol=1e-4, atol=1e-6), \
    f"max diff = {(row_means - row_means_loop).abs().max().item():.2e}"

# ផ្ទៀងផ្ទាត់ dim=0 ផងដែរ (មធ្យមភាគឈរនីមួយៗលើជួរទាំងអស់)
col_means_loop = torch.stack([big[:, j].mean() for j in range(big.shape[1])])
assert torch.allclose(big.mean(dim=0), col_means_loop, rtol=1e-4, atol=1e-6)

print(f"    ផ្ទៀងផ្ទាត់ loop vs vectorized ដូចគ្នា: True "
      f"(ភាពខុសគ្នាអតិបរមា {(row_means - row_means_loop).abs().max().item():.2e})")
print("    ភាពខុសគ្នាតូចនេះកើតឡើងព្រោះ float32 មិន associative។")
print("    បើអ្នកច្រឡំ dim អ្នកនឹងប្រៀបធៀបលេខដែលមានន័យខុសគ្នាទាំងស្រុង!")

t_loop_rows = measure(lambda: [big[i].mean().item() for i in range(big.shape[0])], reps=2)
t_vec_rows = measure(lambda: big.mean(dim=1), reps=5)
print(f"    loop រាល់ជួរ + .item() : {t_loop_rows * 1000:>8.1f} ms")
print(f"    big.mean(dim=1)      : {t_vec_rows * 1000:>8.3f} ms")

# ---------------------------------------------------------------------------
# ២. ប្រើ boolean indexing ជំនួស nonzero + loop
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("២. Filtering")
print("=" * 74)
v = torch.randn(100000)


def loop_filter() -> torch.Tensor:
    out = []
    for i in range(v.numel()):
        if v[i].item() > 0:
            out.append(v[i])
    return torch.stack(out)


def vector_filter() -> torch.Tensor:
    return v[v > 0]


t1 = measure(loop_filter, reps=1)
t2 = measure(vector_filter, reps=20)
print(f"  loop + .item() រាល់ធាតុ : {t1 * 1000:>9.1f} ms  ({loop_filter().numel()} ធាតុ)")
print(f"  v[v > 0] (masked)      : {t2 * 1000:>9.3f} ms")
print(f"  លឿនជាងប្រហែល {t1 / max(t2, 1e-9):,.0f}x")
assert loop_filter().numel() == vector_filter().numel()

# ---------------------------------------------------------------------------
# ៣. dtype និងទំហំ memory
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("៣. dtype និងការប្រើប្រាស់ memory")
print("=" * 74)
n = 4_000_000
print(f"{'dtype':<14}{'bytes/ធាតុ':>12}{'MB':>10}{'ល្បឿន sum':>14}")
for dt in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
    t = torch.ones(n, dtype=dt)
    mb = t.numel() * t.element_size() / 1e6
    speed = measure(lambda: t.sum(), reps=3) * 1000
    print(f"{str(dt).replace('torch.', ''):<14}{t.element_size():>12}{mb:>22.1f}{speed:>12.2f} ms")

print("\n  ការណែនាំ:")
print("   - float32 គឺលំនាំដើម សម្រាប់ការបណ្ដុះបណ្ដាលភាគច្រើន")
print("   - bfloat16/float16 សម្រាប់ mixed precision (ត្រូវការ GradScaler ជាមួយ fp16)")
print("   - float64 តែពេលត្រូវការភាពជាក់លាក់ខ្ពស់ (scientific computing, gradcheck)")
print("   - int8 សម្រាប់ quantized inference")

# ---------------------------------------------------------------------------
# ៤. Contiguity និង memory layout
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("៤. Contiguity និង memory layout")
print("=" * 74)
m = torch.randn(1000, 1000)
mt = m.T                                   # មិន contiguous
t_cont = measure(lambda: mt.sum(dim=0), reps=5)
t_noncont = measure(lambda: m.sum(dim=1), reps=5)
print(f"  m.T.sum(dim=0)  (non-contiguous access): {t_cont * 1000:>8.2f} ms")
print(f"  m.sum(dim=1)    (contiguous access)    : {t_noncont * 1000:>8.2f} ms")
print(f"  m.is_contiguous()   = {m.is_contiguous()}")
print(f"  m.T.is_contiguous() = {mt.is_contiguous()}")

# .contiguous() ចម្លងទិន្នន័យ — ធ្វើម្ដងគត់ មិនមែនរាល់ iteration
t_make = measure(lambda: mt.contiguous(), reps=5)
print(f"  m.T.contiguous() (ចម្លងទិន្នន័យ)        : {t_make * 1000:>8.2f} ms")

# channels_last សម្រាប់ CNN
print("\n  channels_last (សម្រាប់ CNN លើ GPU):")
img = torch.randn(32, 3, 224, 224)
img_cl = img.to(memory_format=torch.channels_last)
print(f"    NCHW         is_contiguous = {img.is_contiguous()}")
print(f"    channels_last is_contiguous(memory_format=torch.channels_last) = "
      f"{img_cl.is_contiguous(memory_format=torch.channels_last)}")
print(f"    shape នៅដដែល: {tuple(img_cl.shape)} ប៉ុន្តែ stride ខុសគ្នា: {img_cl.stride()}")
assert img_cl.shape == img.shape
assert torch.allclose(img, img_cl)

# ---------------------------------------------------------------------------
# ៕. torch.no_grad() / inference_mode() កាត់បន្ថយ overhead
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("៥. Autograd overhead")
print("=" * 74)
net = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
inp = torch.randn(256, 512)

t_grad = measure(lambda: net(inp), reps=10)


def run_nograd():
    with torch.no_grad():
        return net(inp)


t_nograd = measure(run_nograd, reps=10)
print(f"  forward ជាមួយ grad tracking : {t_grad * 1000:>8.2f} ms")
print(f"  forward ក្នុង no_grad()      : {t_nograd * 1000:>8.2f} ms")
print(f"  សន្សំបាន {(1 - t_nograd / t_grad) * 100:.1f}%")

# ---------------------------------------------------------------------------
# ៖. ប្រមាណវិធី in-place និងការជៀសវាងការបង្កើត tensor ថ្មី
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("៦. In-place operations")
print("=" * 74)
t = torch.randn(1000000)
t_out = measure(lambda: t + 1, reps=20)


def in_place():
    t.add_(1)


t_inplace = measure(in_place, reps=20)
print(f"  t + 1   (បង្កើត tensor ថ្មី): {t_out * 1000:>8.2f} ms")
print(f"  t.add_(1) (in-place)      : {t_inplace * 1000:>8.2f} ms")
print("  ⚠ ប្រុងប្រយ័ត្ន: in-place អាចខូច autograd graph។ កុំប្រើលើ tensor")
print("     ដែលត្រូវការសម្រាប់ backward()។")

# ---------------------------------------------------------------------------
# ៗ. ការវែកញែក bottleneck ដោយ torch.profiler
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("៧. torch.profiler — រក bottleneck ពិតប្រាកដ")
print("=" * 74)
from torch.profiler import ProfilerActivity, profile

model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
data = torch.randn(512, 256)
target = torch.randint(0, 10, (512,))
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def step():
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(data), target)
    loss.backward()
    opt.step()


for _ in range(3):
    step()                                  # warmup

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(10):
        step()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=6))
print("\nការប្រើប្រាស់៖")
print("  from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler")
print("  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],")
print("               schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),")
print("               on_trace_ready=tensorboard_trace_handler('./log/profiler'),")
print("               record_shapes=True, with_stack=True) as prof:")
print("      for step, batch in enumerate(loader):")
print("          train_step(batch)")
print("          prof.step()          # <- ចាំបាច់ ដើម្បីបញ្ចប់ step នីមួយៗ")

print("\n" + "=" * 74)
print("របាយការណ៍អំពីការវាស់លើម៉ាស៊ីននេះ:")
print(f"  device        : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"  CPU threads   : {torch.get_num_threads()}")
print(f"  PyTorch       : {torch.__version__}")
print("  ⚠ លេខទាំងអស់ខាងលើជាក់លាក់ចំពោះម៉ាស៊ីននេះ ហើយនឹងខុសគ្នានៅកន្លែងផ្សេង។")
print("OK — បច្ចេកទេសបង្កើនល្បឿនទាំងអស់ដំណើរការត្រឹមត្រូវ")
