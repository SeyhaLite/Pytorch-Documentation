"""ឧបករណ៍ debug និងទម្លាប់ល្អ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. ការធ្វើរោគវិនិច្ឆ័យ NaN / Inf
# ---------------------------------------------------------------------------
print("=" * 70)
print("១. ការធ្វើរោគវិនិច្ឆ័យ NaN / Inf")
print("=" * 70)


def check_tensor(name: str, t: torch.Tensor) -> bool:
    """ឧបករណ៍ត្រួតពិនិត្យដែលអ្នកអាចដាក់ក្នុង training loop។"""
    has_nan = bool(torch.isnan(t).any())
    has_inf = bool(torch.isinf(t).any())
    if has_nan or has_inf:
        print(f"    ✗ {name:<18} NaN={has_nan} Inf={has_inf} "
              f"min={t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else float('nan'):.3e} "
              f"max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else float('nan'):.3e}")
    else:
        print(f"    ✓ {name:<18} min={t.min().item():+.4e} max={t.max().item():+.4e} "
              f"mean={t.mean().item():+.4e}")
    return not (has_nan or has_inf)


good = torch.randn(100)
exploded = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 1e40])
underflow = torch.tensor([1e-40, 1e-45, 0.0])

ok1 = check_tensor("weights", good)
ok2 = check_tensor("exploded", exploded)
ok3 = check_tensor("underflow", underflow)
assert ok1 and not ok2

# មូលហេតុទូទៅនៃ NaN
print("\n  មូលហេតុទូទៅនៃ NaN:")
causes = [
    ("log(0)", lambda: torch.log(torch.tensor(0.0))),
    ("0/0", lambda: torch.tensor(0.0) / torch.tensor(0.0)),
    ("sqrt(-1)", lambda: torch.sqrt(torch.tensor(-1.0))),
    ("exp(1000)", lambda: torch.exp(torch.tensor(1000.0))),
    ("lr ធំពេក -> overflow", lambda: torch.tensor(1e30) * 1e30),
]
for label, fn in causes:
    r = fn()
    print(f"    {label:<26} = {r.item()}")

# ---------------------------------------------------------------------------
# ២. torch.autograd.set_detect_anomaly()
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("២. torch.autograd.set_detect_anomaly()")
print("=" * 70)

model = nn.Linear(4, 2)
x = torch.randn(8, 4)

print("  បើក anomaly detection ដើម្បីស្វែងរកប្រភពនៃ NaN ក្នុង backward:")
with torch.autograd.set_detect_anomaly(True):
    # បង្កើត NaN ដោយចេតនាតាមរយៈ log(0)
    logits = model(x)
    bad = torch.log(torch.zeros(1)) * logits
    loss = bad.sum()
    try:
        loss.backward()
        print("    (backward បញ្ចប់ដោយគ្មានកំហុស — NaN កើតឡើងនៅ forward)")
    except RuntimeError as e:
        print("    ✗ RuntimeError ត្រូវបានចាប់យក:")
        for line in str(e).splitlines()[:6]:
            print("      │", line[:88])

print("\n  ⚠ detect_anomaly ធ្វើឱ្យយឺតយ៉ាងខ្លាំង — ប្រើតែពេល debug ប៉ុណ្ណោះ")

# ---------------------------------------------------------------------------
# ៣. Gradient health checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("៣. Gradient health checks")
print("=" * 70)

deep = nn.Sequential(*[nn.Linear(8, 8) for _ in range(30)])
loss = deep(torch.randn(4, 8) * 5).sum()
loss.backward()


def report_gradients(model: nn.Module) -> dict:
    stats = {"n_params": 0, "zero": 0, "nan": 0, "exploded": 0, "norms": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        stats["n_params"] += 1
        g = p.grad
        norm = g.norm().item()
        stats["norms"].append(norm)
        if norm == 0:
            stats["zero"] += 1
        if torch.isnan(g).any():
            stats["nan"] += 1
        if norm > 1e3:
            stats["exploded"] += 1
    return stats


st = report_gradients(deep)
print(f"  parameters ជាមួយ grad : {st['n_params']}")
print(f"  gradient norm អតិបរមា  : {max(st['norms']):.3e}")
print(f"  gradient norm អប្បបរមា  : {min(st['norms']):.3e}")
print(f"  gradient = 0           : {st['zero']}")
print(f"  gradient = NaN         : {st['nan']}")
print(f"  gradient > 1e3         : {st['exploded']}")
assert st["n_params"] > 0

print("\n  ការបកស្រាយ:")
print("    grad = 0 គ្រប់ទីកន្លែង  -> ភ្លេច loss.backward() ឬ layer មិនជាប់ graph")
print("    grad = NaN            -> overflow, log(0), 0/0 ឬ lr ធំពេក")
print("    grad ធំខ្លាំង          -> exploding gradient ប្រើ clip_grad_norm_")
print("    grad តូចខ្លាំង (<1e-7) -> vanishing gradient ប្រើ ReLU/residual/normalization")

# ---------------------------------------------------------------------------
# ៤. Hooks ដើម្បីតាមដាន activation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("៤. Forward hooks — តាមដាន activation គ្រប់ជាន់")
print("=" * 70)

net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
activations: dict[str, dict] = {}


def register_stats_hook(name: str):
    def hook(module, inputs, output):
        if torch.is_tensor(output):
            activations[name] = {
                "shape": tuple(output.shape),
                "mean": output.mean().item(),
                "std": output.std().item(),
                "min": output.min().item(),
                "max": output.max().item(),
                "dead_pct": float((output == 0).float().mean().item() * 100),
            }
    return hook


handles = []
for name, module in net.named_modules():
    if name and not list(module.children()):
        handles.append(module.register_forward_hook(register_stats_hook(name)))

with torch.no_grad():
    net(torch.randn(32, 8))

print(f"  {'layer':<10}{'shape':<14}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'dead%':>8}")
for name, s in activations.items():
    print(f"  {name:<10}{str(s['shape']):<14}{s['mean']:>10.4f}{s['std']:>10.4f}"
          f"{s['min']:>10.4f}{s['max']:>10.4f}{s['dead_pct']:>8.1f}")

print("\n  ការបកស្រាយ:")
print("    std តូចខ្លាំង (<< 0.01)  -> activation រលាយ (vanishing)")
print("    std ធំខ្លាំង (>> 10)     -> activation ផ្ទុះ (exploding)")
print("    dead% ខ្ពស់ជាមួយ ReLU   -> dying ReLU ពិចារណា LeakyReLU/GELU")
print("    mean ឆ្ងាយពី 0          -> ត្រូវការ normalization")

# ដក hooks ចេញជានិច្ច បើមិនដូច្នោះទេវានៅជាប់
for h in handles:
    h.remove()
print(f"\n  បានដក hooks ចេញ ({len(handles)}) — ចាំបាច់ បើមិនដូច្នោះទេ memory leak")

# ---------------------------------------------------------------------------
# ៕. ការបង្ហាញ tensor ឱ្យងាយអាន
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("៕. torch.set_printoptions")
print("=" * 70)
big_t = torch.randn(10, 10)
print("លំនាំដើម:")
print(big_t[:2, :4])

torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)
print("\nprecision=3, sci_mode=False:")
print(big_t[:2, :4])

torch.set_printoptions(profile="default")
print("\nបានស្ដារលំនាំដើម (profile='default')")

# ---------------------------------------------------------------------------
# ៖. torch.fx — សង្កេតមើល computation graph
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("៖. torch.fx.symbolic_trace — មើល graph")
print("=" * 70)
try:
    import torch.fx

    traced = torch.fx.symbolic_trace(net)
    print(traced.graph)
    print("\n  ប្រើប្រាស់៖ យល់ពីលំដាប់ប្រតិបត្តិការ ស្វែងរកជាន់ដែលមិនរំពឹងទុក")
except Exception as e:
    print(f"  torch.fx មិនអាច trace បាន: {type(e).__name__}")

# ---------------------------------------------------------------------------
# ៗ. Checklist ធ្វើរោគវិនិច្ឆ័យ
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ៗ. Checklist ពេលម៉ូដែលមិនរៀន")
print("=" * 70)
steps = [
    "តើ loss ថយចុះទេ? បើទេ -> ពិនិត្យ lr, zero_grad, backward, optimizer",
    "តើ gradient ហូរទេ? report_gradients() ខាងលើ",
    "តើ activation សមហេតុផលទេ? forward hooks",
    "តើទិន្នន័យត្រឹមត្រូវទេ? ពិនិត្យសំណាកខ្លះដោយភ្នែក, ផ្ទៀងផ្ទាត់ label",
    "តើម៉ូដែលអាច overfit batch តូចមួយបានទេ? សាកល្បងលើសំណាក 16 ដោយគ្មាន shuffle",
    "តើ train() / eval() ត្រឹមត្រូវទេ?",
    "តើ loss function សមនឹងបញ្ហាទេ? (logits vs probabilities)",
    "តើ NaN លេចឡើងនៅ iteration ណា? កត់ត្រាលើកដំបូង",
]
for i, s in enumerate(steps, 1):
    print(f"  {i}. {s}")

print("\n  គន្លឹះ: ធ្វើឱ្យវាតូចជាងមុនសិន — ទិន្នន័យ 32 សំណាក, ម៉ូដែលតូច,")
print("  គ្មាន augmentation, គ្មាន regularization។ បើវាមិនរៀនទេ បញ្ហានៅក្នុងកូដ។")

print("\nOK — ឧបករណ៍ debug ទាំងអស់ដំណើរការត្រឹមត្រូវ")
