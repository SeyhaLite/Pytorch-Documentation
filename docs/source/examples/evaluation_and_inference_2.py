"""Inference៖ no_grad vs inference_mode, batch prediction និង torch.compile។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import time

import torch
import torch.nn as nn

torch.manual_seed(0)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = Net()
model.eval()                      # <- ចាំបាច់ជានិច្ចមុន inference
x = torch.randn(64, 32)

# ---------------------------------------------------------------------------
# ១. torch.no_grad() vs torch.inference_mode()
# ---------------------------------------------------------------------------
print("=" * 64)
print("no_grad vs inference_mode")
print("=" * 64)

with torch.no_grad():
    out_ng = model(x)
print("no_grad       : requires_grad =", out_ng.requires_grad, "| grad_fn =", out_ng.grad_fn)

with torch.inference_mode():
    out_im = model(x)
print("inference_mode: requires_grad =", out_im.requires_grad, "| grad_fn =", out_im.grad_fn)

assert torch.allclose(out_ng, out_im)
print("លទ្ធផលដូចគ្នា:", torch.allclose(out_ng, out_im))

# ភាពខុសគ្នាសំខាន់៖ tensor ពី inference_mode មិនអាចប្រើក្នុង autograd បានទេ
print("\nភាពខុសគ្នាសំខាន់:")
try:
    out_im.requires_grad_(True)
    print("  inference tensor + requires_grad_ -> អនុញ្ញាត (?)")
except RuntimeError as e:
    print("  inference tensor + requires_grad_ -> RuntimeError:")
    print("   ", str(e).splitlines()[0][:78])

try:
    (out_im * 2).sum().backward()
except RuntimeError as e:
    print("  inference tensor + backward()     -> RuntimeError:")
    print("   ", str(e).splitlines()[0][:78])

# out_ng អាចធ្វើជា leaf សម្រាប់ gradient បាន
out_ng2 = out_ng.detach().clone().requires_grad_(True)
(out_ng2 * 2).sum().backward()
print("  no_grad tensor អាច .requires_grad_(True) បាន:", out_ng2.grad is not None)

# ---------------------------------------------------------------------------
# ២. វាស់ល្បឿន (លើម៉ាស៊ីននេះ)
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print(f"{'mode':<26}{'រយៈពេលសម្រាប់ 200 forward passes':>34}")
print("=" * 64)

def timeit(fn, reps: int = 200) -> float:
    fn()                                    # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return time.perf_counter() - t0

def run_nograd():
    with torch.no_grad():
        return model(x)


def run_inference():
    with torch.inference_mode():
        return model(x)


t_grad = timeit(lambda: model(x))
t_nograd = timeit(run_nograd)
t_inf = timeit(run_inference)

print(f"{'គ្មាន context (grad on)':<26}{t_grad * 1000:>30.1f} ms")
print(f"{'torch.no_grad()':<26}{t_nograd * 1000:>30.1f} ms")
print(f"{'torch.inference_mode()':<26}{t_inf * 1000:>30.1f} ms")
print("\n(លេខទាំងនេះវាស់លើម៉ាស៊ីនដែលកំពុងរត់ឯកសារនេះ — មិនមែន benchmark ជាសកលទេ)")

# ---------------------------------------------------------------------------
# ៣. Batch inference — វិធីត្រឹមត្រូវសម្រាប់ទិន្នន័យច្រើន
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("Batch inference")
print("=" * 64)
dataset = torch.randn(1000, 32)
batch_size = 128
all_preds, all_probs = [], []

t0 = time.perf_counter()
with torch.inference_mode():
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs)
        all_preds.append(probs.argmax(dim=1))

probs = torch.cat(all_probs, dim=0)
preds = torch.cat(all_preds, dim=0)
dt = time.perf_counter() - t0

print(f"ទិន្នន័យ 1000 សំណាក, batch_size={batch_size}")
print(f"  preds shape : {tuple(preds.shape)}  dtype={preds.dtype}")
print(f"  probs shape : {tuple(probs.shape)}")
print(f"  រយៈពេល      : {dt * 1000:.1f} ms")
assert preds.shape == (1000,)
assert probs.shape == (1000, 10)
assert torch.allclose(probs.sum(dim=1), torch.ones(1000), atol=1e-5)

# ចំនួនសំណាកក្នុងថ្នាក់នីមួយៗ
counts = torch.bincount(preds, minlength=10)
print(f"  ការចែកចាយថ្នាក់: {counts.tolist()}")
assert counts.sum().item() == 1000

# ---------------------------------------------------------------------------
# ៤. Top-k predictions និង confidence
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("Top-3 predictions")
print("=" * 64)
topk_probs, topk_idx = torch.topk(probs, k=3, dim=1)
for i in range(3):
    print(f"  សំណាក {i}: " +
          ", ".join(f"class {int(c)}={float(p):.3f}" for c, p in zip(topk_idx[i], topk_probs[i])))
assert topk_idx.shape == (1000, 3)

# ---------------------------------------------------------------------------
# ៥. torch.compile — ការបង្កើនល្បឿនដោយស្វ័យប្រវត្តិ
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("torch.compile")
print("=" * 64)
print("torch.compile មាន:", hasattr(torch, "compile"))

try:
    compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
    with torch.inference_mode():
        out_c = compiled(x)
    assert torch.allclose(out_c, out_ng, atol=1e-5)
    print("  compiled forward លទ្ធផលដូច eager:", torch.allclose(out_c, out_ng, atol=1e-5))

    def run_compiled():
        with torch.inference_mode():
            return compiled(x)

    run_compiled()                        # ឱ្យ compile រួចរាល់
    run_compiled()
    t_comp = timeit(run_compiled, reps=200)
    print(f"  compiled រយៈពេល 200 passes: {t_comp * 1000:.1f} ms")
    print("  ⚠ លទ្ធផលប្រែប្រួលខ្លាំងតាម hardware/backend — កុំចាត់ទុកជា benchmark")
except Exception as e:
    print(f"  torch.compile មិនអាចប្រើបាននៅទីនេះ: {type(e).__name__}: {str(e)[:70]}")
    print("  (វាត្រូវការ inductor backend និងប្រព័ន្ធដែលគាំទ្រ)")

print("\nOK — inference ទាំងអស់ដំណើរការត្រឹមត្រូវ")
