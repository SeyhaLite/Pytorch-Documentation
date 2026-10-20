"""ប្រមាណវិធីលើ tensor៖ element-wise, matrix, reduction, in-place។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch

torch.manual_seed(0)
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

# ---------------------------------------------------------------------------
# ១. Element-wise (ម្ដងមួយធាតុ)
# ---------------------------------------------------------------------------
print("a + b      :", (a + b).tolist())
print("a - b      :", (a - b).tolist())
print("a * b      :", (a * b).tolist(), "  # ហៅផងដែរថា Hadamard product")
print("a / b      :", (a / b).tolist())
print("torch.pow  :", torch.pow(a, 2).tolist())
print("torch.exp  :", [[round(v, 4) for v in row] for row in torch.exp(a).tolist()])
print("torch.log  :", [[round(v, 4) for v in row] for row in torch.log(a).tolist()])
print("torch.sqrt :", [[round(v, 4) for v in row] for row in torch.sqrt(a).tolist()])
print("torch.abs  :", torch.abs(-a).tolist())
print("torch.clamp:", torch.clamp(a * 5, min=5.0, max=15.0).tolist())

assert torch.equal(a * b, torch.tensor([[10.0, 40.0], [90.0, 160.0]]))

# ---------------------------------------------------------------------------
# ២. Matrix multiply — ចំណុចដែលអ្នកចាប់ផ្ដើមច្រឡំញឹកញាប់បំផុត
# ---------------------------------------------------------------------------
print("-" * 55)
m1 = torch.randn(2, 3)
m2 = torch.randn(3, 4)

print("a * b   shape:", tuple((a * b).shape), "  <- element-wise")
print("a @ b   shape:", tuple((a @ b).shape), "  <- matrix multiply")
print("m1 @ m2 shape:", tuple((m1 @ m2).shape), "  <- (2,3) @ (3,4) = (2,4)")

# វិធីសរសេរដូចគ្នាទាំង 3
r1 = m1 @ m2
r2 = torch.matmul(m1, m2)
r3 = m1.mm(m2)
assert torch.allclose(r1, r2) and torch.allclose(r1, r3)
print("m1 @ m2 == torch.matmul(m1,m2) == m1.mm(m2) : True")

# dot product សម្រាប់ vector 1D
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
print("torch.dot(v1,v2) :", torch.dot(v1, v2).item(), "  # 1*4 + 2*5 + 3*6 = 32")
assert torch.dot(v1, v2).item() == 32.0

# ---------------------------------------------------------------------------
# ៣. Reductions (បង្រួមវិមាត្រ)
# ---------------------------------------------------------------------------
print("-" * 55)
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("x =\n", x)
print("x.sum()          :", x.sum().item())             # scalar
print("x.sum(dim=0)     :", x.sum(dim=0).tolist(), " shape", tuple(x.sum(dim=0).shape))
print("x.sum(dim=1)     :", x.sum(dim=1).tolist(), " shape", tuple(x.sum(dim=1).shape))
print("x.mean(dim=1)    :", x.mean(dim=1).tolist())
print("x.max(dim=1)     :", x.max(dim=1))                  # (values, indices)
print("x.argmax(dim=1)  :", x.argmax(dim=1).tolist())
print("x.std(dim=0)     :", [round(v, 4) for v in x.std(dim=0).tolist()])

assert x.sum().item() == 66.0                  # 0+1+...+11
assert x.sum(dim=0).shape == (4,)
assert x.sum(dim=1).shape == (3,)
assert x.argmax(dim=1).tolist() == [3, 3, 3]

# keepdim=True រក្សាវិមាត្រ — សំខាន់ណាស់សម្រាប់ broadcasting
print("-" * 55)
mean_keep = x.mean(dim=1, keepdim=True)
print("x.mean(dim=1, keepdim=True) shape:", tuple(mean_keep.shape))
centered = x - mean_keep       # broadcasting ដោយស្វ័យប្រវត្តិ
print("x - mean  (shape):", tuple(centered.shape))
assert mean_keep.shape == (3, 1)
assert centered.shape == (3, 4)
assert torch.allclose(centered.mean(dim=1), torch.zeros(3), atol=1e-6)

# ---------------------------------------------------------------------------
# ៤. In-place operations (បញ្ចប់ដោយ _)
# ---------------------------------------------------------------------------
print("-" * 55)
y = torch.ones(3)
y.add_(5)          # y = y + 5
y.mul_(2)          # y = y * 2
y.zero_()          # y = 0
print("បន្ទាប់ add_(5), mul_(2), zero_() :", y.tolist())
assert torch.equal(y, torch.zeros(3))

# ---------------------------------------------------------------------------
# ៥. ប្រៀបធៀប (ត្រឡប់ tensor bool)
# ---------------------------------------------------------------------------
print("-" * 55)
print("a > 2      :", (a > 2).tolist())
print("a == b     :", (a == b).tolist())
print("(a>2).sum():", (a > 2).sum().item())
print("torch.all  :", torch.all(a > 0).item())
print("torch.any  :", torch.any(a > 3).item())

# ---------------------------------------------------------------------------
# ៦. ភាពជាក់លាក់នៃ float — កុំប្រើ == លើ float
# ---------------------------------------------------------------------------
print("-" * 55)
# float32 មានខ្ទង់សំខាន់តែ ~7 ខ្ទង់ ដូច្នេះការបូកបន្តិចម្ដងៗបង្កើតកំហុស rounding
acc = torch.full((10,), 0.1).sum()      # បូក 0.1 ចំនួន 10 ដង
exact = torch.tensor(1.0)
print(f"បូក 0.1 ចំនួន 10 ដង = {acc.item():.20f}")
print(f"1.0                 = {exact.item():.20f}")
print("acc == exact        :", bool(acc == exact), " <- ខុស ដោយសារ rounding")
print("torch.isclose       :", bool(torch.isclose(acc, exact)))
print("torch.allclose      :", torch.allclose(acc, exact))

diff = torch.tensor(0.3) - torch.tensor(0.1)
print(f"\n0.3 - 0.1 = {diff.item():.20f}")
print(f"0.2       = {torch.tensor(0.2).item():.20f}")
print("equal     :", bool(diff == torch.tensor(0.2)))
print("allclose  :", torch.allclose(diff, torch.tensor(0.2)))

assert not bool(acc == exact)
assert torch.allclose(acc, exact)
assert not bool(diff == torch.tensor(0.2))
assert torch.allclose(diff, torch.tensor(0.2))

# float64 មានភាពជាក់លាក់ខ្ពស់ជាង ប៉ុន្តែនៅតែមានដែនកំណត់
d1 = torch.tensor(0.1, dtype=torch.float64) + torch.tensor(0.2, dtype=torch.float64)
d2 = torch.tensor(0.3, dtype=torch.float64)
print(f"\nfloat64: 0.1+0.2 == 0.3 -> {bool(d1 == d2)}  (ភាគច្រើននៅតែខុស)")

print("-" * 55)
print("OK — ប្រមាណវិធីទាំងអស់ដំណើរការត្រឹមត្រូវ")
