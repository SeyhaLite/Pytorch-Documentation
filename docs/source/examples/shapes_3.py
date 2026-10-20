"""អនុវត្តជាក់ស្ដែង៖ គណនាមធ្យមភាគតាមថ្នាក់ និង softmax ដោយប្រើ broadcasting។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch

torch.manual_seed(7)

# ---------------------------------------------------------------------------
# ១. ទិន្នន័យ៖ សិស្ស 6 នាក់ មុខវិជ្ជា 4  ចែកជា 2 ក្រុម
# ---------------------------------------------------------------------------
scores = torch.tensor([
    [85.0, 90.0, 78.0, 92.0],
    [70.0, 65.0, 80.0, 75.0],
    [95.0, 88.0, 91.0, 89.0],
    [60.0, 72.0, 68.0, 64.0],
    [88.0, 85.0, 90.0, 87.0],
    [55.0, 60.0, 58.0, 62.0],
])
groups = torch.tensor([0, 1, 0, 1, 0, 1])   # 0 = ក្រុម A, 1 = ក្រុម B

print("scores shape :", tuple(scores.shape))   # (6, 4)
print("groups shape :", tuple(groups.shape))   # (6,)

# ---------------------------------------------------------------------------
# ២. មធ្យមភាគក្រុម ដោយប្រើ boolean mask + broadcasting
# ---------------------------------------------------------------------------
print("-" * 58)
for g in [0, 1]:
    mask = (groups == g).unsqueeze(1)    # (6,) -> (6,1) ដើម្បី broadcast លើឈរ
    selected = scores[mask.expand_as(scores)]
    print(f"ក្រុម {g}: មធ្យមភាគ = {scores[mask.expand_as(scores)].reshape(-1, 4).mean(dim=0).tolist()}")
    assert mask.shape == (6, 1)
    del selected

# វិធីសាមញ្ញជាង និងលឿនជាង៖ ប្រើ index ដោយផ្ទាល់
mean_a = scores[groups == 0].mean(dim=0)
mean_b = scores[groups == 1].mean(dim=0)
print("-" * 58)
print("ក្រុម A (groups==0) មធ្យមភាគតាមមុខវិជ្ជា:", mean_a.tolist())
print("ក្រុម B (groups==1) មធ្យមភាគតាមមុខវិជ្ជា:", mean_b.tolist())
assert mean_a.shape == (4,) and mean_b.shape == (4,)

# ផ្ទៀងផ្ទាត់ដោយគណនាដោយដៃ
manual_a = scores[[0, 2, 4]].mean(dim=0)
assert torch.allclose(mean_a, manual_a)

# ---------------------------------------------------------------------------
# ៣. softmax ដោយដៃ — បង្ហាញពីហេតុអ្វីត្រូវដក max
# ---------------------------------------------------------------------------
print("-" * 58)
logits = torch.randn(3, 5) * 3      # (batch=3, classes=5)

# វិធីខុស (អាច overflow ជាមួយលេខធំ)
naive = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)

# វិធីត្រឹមត្រូវ (ដក max ដើម្បីស្ថិរភាពលេខ)
shifted = logits - logits.max(dim=1, keepdim=True).values
stable = torch.exp(shifted) / torch.exp(shifted).sum(dim=1, keepdim=True)

# វិធីរបស់ PyTorch
official = torch.softmax(logits, dim=1)

print("naive   shape :", tuple(naive.shape))
print("stable  shape :", tuple(stable.shape))
print("official shape:", tuple(official.shape))
print("naive ≈ official   :", torch.allclose(naive, official, atol=1e-6))
print("stable ≈ official  :", torch.allclose(stable, official, atol=1e-6))
print("ផលបូកតាមជួរ (គួរ = 1.0):", official.sum(dim=1).tolist())

assert torch.allclose(official.sum(dim=1), torch.ones(3), atol=1e-6)
assert torch.allclose(stable, official, atol=1e-6)

# បង្ហាញ overflow ពិតប្រាកដ
big = torch.tensor([1000.0, 1000.0, 1000.0])
print("\nexp(1000) =", torch.exp(big).tolist(), " <- inf!")
big_safe = torch.exp(big - big.max())
print("exp(1000 - max) =", big_safe.tolist(), " <- សុវត្ថិភាព")
assert torch.isinf(torch.exp(big)).all()
assert torch.isfinite(big_safe).all()

# ---------------------------------------------------------------------------
# ៤. one-hot encoding ដោយប្រើ broadcasting
# ---------------------------------------------------------------------------
print("-" * 58)
labels = torch.tensor([2, 0, 4, 1])
n_classes = 5

one_hot = (torch.arange(n_classes) == labels.unsqueeze(1)).to(torch.float32)
print("labels   :", labels.tolist())
print("one_hot  :\n", one_hot)
assert one_hot.shape == (4, 5)
assert one_hot.sum(dim=1).tolist() == [1.0, 1.0, 1.0, 1.0]
assert torch.equal(one_hot, torch.nn.functional.one_hot(labels, n_classes).float())
print("ផ្ទៀងផ្ទាត់ជាមួយ F.one_hot : ដូចគ្នា")

print("-" * 58)
print("OK — ការអនុវត្ត broadcasting ទាំងអស់ត្រឹមត្រូវ")
