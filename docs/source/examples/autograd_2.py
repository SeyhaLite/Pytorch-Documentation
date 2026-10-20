"""Backpropagation ពេញលេញលើ neural network តូចមួយ — ដើរតាមជំហាននីមួយៗ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
គោលបំណង៖ បង្ហាញថា loss.backward() បំពេញ .grad របស់ parameter គ្រប់ជាន់។
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. ម៉ូដែល 2 ជាន់ និងទិន្នន័យសប្បនិម្មិត
# ---------------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.Tanh(),
    nn.Linear(8, 1),
)
loss_fn = nn.MSELoss()

x = torch.randn(16, 4)                 # (batch=16, features=4)
target = torch.randn(16, 1)            # (batch=16, output=1)

print("input  :", tuple(x.shape))
print("target :", tuple(target.shape))
print("parameters របស់ម៉ូដែល៖")
for name, p in model.named_parameters():
    print(f"  {name:<18} shape={str(tuple(p.shape)):<10} requires_grad={p.requires_grad} grad={p.grad}")

# ---------------------------------------------------------------------------
# ២. Forward pass
# ---------------------------------------------------------------------------
print("-" * 62)
pred = model(x)
print("pred shape:", tuple(pred.shape), "(ត្រូវតែដូច target)")
assert pred.shape == target.shape

loss = loss_fn(pred, target)
print("loss      :", loss.item())
print("loss.grad_fn:", loss.grad_fn)   # MseLossBackward0 — ភស្តុតាងថា graph ត្រូវបានកត់ត្រា

# ---------------------------------------------------------------------------
# ៣. Backward pass — គណនា gradient គ្រប់ parameter
# ---------------------------------------------------------------------------
print("-" * 62)
loss.backward()

total_grad_norm = 0.0
for name, p in model.named_parameters():
    assert p.grad is not None, f"{name} គ្មាន .grad!"
    g = p.grad
    total_grad_norm += g.pow(2).sum().item()
    print(f"  {name:<18} grad shape={str(tuple(g.shape)):<10} "
          f"mean={g.mean().item():+.6f}  max={g.abs().max().item():.6f}")

total_grad_norm = total_grad_norm ** 0.5
print(f"\ngradient norm សរុប: {total_grad_norm:.6f}")
assert total_grad_norm > 0, "gradient មិនគួរសូន្យទាំងអស់ទេ"

# ---------------------------------------------------------------------------
# ៤. ផ្ទៀងផ្ទាត់ gradient ដោយ finite differences
# ---------------------------------------------------------------------------
print("-" * 62)
print("ផ្ទៀងផ្ទាត់ d(loss)/d(param) ដោយវិធីសាស្ត្រ numeric:")
eps = 1e-4
param = model[0].weight            # Linear layer ទី 0
i, j = 1, 2                        # ពិនិត្យធាតុ [1, 2]

with torch.no_grad():
    original = param[i, j].item()

    param[i, j] = original + eps
    loss_plus = loss_fn(model(x), target).item()

    param[i, j] = original - eps
    loss_minus = loss_fn(model(x), target).item()

    param[i, j] = original         # ស្ដារឡើងវិញ

numeric_grad = (loss_plus - loss_minus) / (2 * eps)
autograd_grad = param.grad[i, j].item()

print(f"  numeric  gradient = {numeric_grad:+.8f}")
print(f"  autograd gradient = {autograd_grad:+.8f}")
print(f"  ភាពខុសគ្នា         = {abs(numeric_grad - autograd_grad):.2e}")
assert abs(numeric_grad - autograd_grad) < 1e-3

# ---------------------------------------------------------------------------
# ៥. torch.autograd.grad() — យក gradient ដោយមិនរក្សាទុកក្នុង .grad
# ---------------------------------------------------------------------------
print("-" * 62)
pred2 = model(x)
loss2 = loss_fn(pred2, target)
grads = torch.autograd.grad(loss2, list(model.parameters()), retain_graph=True)
print("torch.autograd.grad ត្រឡប់:", len(grads), "tensors")
print("  grads[0] shape:", tuple(grads[0].shape))
assert len(grads) == sum(1 for _ in model.parameters())

# ---------------------------------------------------------------------------
# ៦. gradcheck លើ function តូចមួយ (វិធីសាស្ត្រផ្លូវការ)
# ---------------------------------------------------------------------------
print("-" * 62)
def f(t: torch.Tensor) -> torch.Tensor:
    return (t ** 3).sum()

point = torch.randn(5, dtype=torch.float64, requires_grad=True)
ok = torch.autograd.gradcheck(f, (point,), eps=1e-6, atol=1e-4)
print("torch.autograd.gradcheck(f) ->", ok)
assert ok

print("-" * 62)
print("OK — backpropagation និង gradient ទាំងអស់ត្រឹមត្រូវ")
