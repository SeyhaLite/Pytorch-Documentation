"""Broadcasting៖ ច្បាប់ ការប្រើប្រាស់ និងកំហុសដែលលាក់ខ្លួន។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

ច្បាប់ broadcasting (អានពីវិមាត្រចុងក្រោយទៅមុខ)៖
  1. បើចំនួនវិមាត្រមិនស្មើគ្នា បន្ថែមវិមាត្រទំហំ 1 នៅខាងមុខ។
  2. សម្រាប់វិមាត្រនីមួយៗ ទំហំត្រូវតែ (ក) ស្មើគ្នា ឬ (ខ) មួយក្នុងចំណោមវាស្មើ 1។
  3. ទំហំ 1 ត្រូវបាន "ពង្រីក" (repeat) ដោយស្វ័យប្រវត្តិ ដោយមិនចម្លង memory។
"""

import torch

# ---------------------------------------------------------------------------
# ១. ឧទាហរណ៍មូលដ្ឋាន
# ---------------------------------------------------------------------------
a = torch.ones(4, 3)          # shape (4, 3)
b = torch.tensor([1.0, 2.0, 3.0])   # shape (3,)

c = a + b
print("a shape:", tuple(a.shape), " b shape:", tuple(b.shape), " -> a+b shape:", tuple(c.shape))
print(c)
# (3,) ត្រូវបានគេមើលឃើញជា (1, 3) ហើយពង្រីកទៅ (4, 3)
assert c.shape == (4, 3)
assert torch.equal(c, torch.tensor([[2.0, 3.0, 4.0]] * 4)), c

# scalar broadcast ទៅគ្រប់ទីកន្លែង
d = a * 10
assert d.shape == (4, 3)
print("\na * 10 shape:", tuple(d.shape))

# ---------------------------------------------------------------------------
# ២. torch.broadcast_shapes() — ពិនិត្យមុនគណនា
# ---------------------------------------------------------------------------
print("-" * 58)
pairs = [
    ((4, 3), (3,)),
    ((8, 1, 6, 1), (7, 1, 5)),
    ((2, 3), (3, 2)),          # នេះនឹងបរាជ័យ
]
for s1, s2 in pairs:
    try:
        result = torch.broadcast_shapes(s1, s2)
        print(f"broadcast_shapes({s1}, {s2}) = {tuple(result)}")
    except RuntimeError as e:
        print(f"broadcast_shapes({s1}, {s2}) -> RuntimeError: {str(e).splitlines()[0][:60]}")

# ---------------------------------------------------------------------------
# ៣. ករណីពិត៖ standardization តាម feature
# ---------------------------------------------------------------------------
print("-" * 58)
torch.manual_seed(0)
batch = torch.randn(32, 10)        # (batch_size, features)

mean = batch.mean(dim=0)           # shape (10,)
std = batch.std(dim=0)             # shape (10,)
normalized = (batch - mean) / std  # broadcasting: (32,10) - (10,)

print("batch      :", tuple(batch.shape))
print("mean       :", tuple(mean.shape))
print("normalized :", tuple(normalized.shape))
print("mean បន្ទាប់ normalize (គួរ ~0):",
      [round(v, 6) for v in normalized.mean(dim=0).tolist()[:4]], "…")
print("std  បន្ទាប់ normalize (គួរ ~1):",
      [round(v, 4) for v in normalized.std(dim=0).tolist()[:4]], "…")

assert normalized.shape == (32, 10)
assert torch.allclose(normalized.mean(dim=0), torch.zeros(10), atol=1e-6)
assert torch.allclose(normalized.std(dim=0), torch.ones(10), atol=1e-4)

# ---------------------------------------------------------------------------
# ៤. ករណីពិត៖ pairwise distance ដោយមិនប្រើ loop
# ---------------------------------------------------------------------------
print("-" * 58)
p = torch.randn(5, 3)      # 5 ចំណុច វិមាត្រ 3
q = torch.randn(4, 3)      # 4 ចំណុច វិមាត្រ 3

# unsqueeze ដើម្បីបង្កើតវិមាត្រសម្រាប់ broadcast
diff = p.unsqueeze(1) - q.unsqueeze(0)   # (5,1,3) - (1,4,3) -> (5,4,3)
dist = diff.pow(2).sum(dim=-1).sqrt()    # (5,4)

print("p.unsqueeze(1) :", tuple(p.unsqueeze(1).shape))
print("q.unsqueeze(0) :", tuple(q.unsqueeze(0).shape))
print("diff           :", tuple(diff.shape))
print("dist           :", tuple(dist.shape))
assert dist.shape == (5, 4)
assert torch.equal(dist, dist.T.contiguous() if p is q else dist)  # មិន symmetric ទេ ព្រោះ p != q
assert (dist >= 0).all()

# ---------------------------------------------------------------------------
# ៥. កំហុសដែលលាក់ខ្លួន (silent bug) — សំខាន់បំផុត!
# ---------------------------------------------------------------------------
print("-" * 58)
# ចេតនា៖ គុណ loss មួយៗជាមួយ weight មួយៗ (4 ធាតុ)
losses = torch.tensor([1.0, 2.0, 3.0, 4.0])   # shape (4,)
weights = torch.tensor([0.1, 0.2, 0.3, 0.4])  # shape (4,)

correct = losses * weights
print("correct  shape:", tuple(correct.shape), "->", correct.tolist())
assert correct.shape == (4,)

# កំហុស៖ reshape ខុស ធ្វើឱ្យ broadcast បង្កើត matrix ដែលអ្នកមិនចង់បាន
wrong = losses.reshape(4, 1) * weights.reshape(1, 4)
print("wrong    shape:", tuple(wrong.shape), "<- រំពឹង (4,) តែទទួល (4,4)!")
print(wrong)
assert wrong.shape == (4, 4)

# វិធីការពារ៖ ផ្ទៀងផ្ទាត់ shape ជានិច្ច
if wrong.shape != correct.shape:
    print("\nការពារ៖ assert result.shape == expected_shape មុនប្រើលទ្ធផល")

# ---------------------------------------------------------------------------
# ៦. កំហុសដែល broadcast មិនអាចធ្វើបាន
# ---------------------------------------------------------------------------
print("-" * 58)
try:
    torch.ones(4, 3) + torch.ones(4, 2)
except RuntimeError as e:
    print("(4,3) + (4,2) -> RuntimeError:")
    print(" ", str(e).splitlines()[0][:90])

try:
    torch.ones(3) + torch.ones(4)
except RuntimeError as e:
    print("(3,) + (4,)  -> RuntimeError:")
    print(" ", str(e).splitlines()[0][:90])

print("-" * 58)
print("OK — broadcasting ទាំងអស់ដំណើរការត្រឹមត្រូវ")
