"""Tensor ដំបូងរបស់អ្នក។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
បង្ហាញ៖ ការបង្កើត tensor, shape, dtype, device និងការគណនាមូលដ្ឋាន។
"""

import torch

# ---------------------------------------------------------------------------
# ១. បង្កើត tensor ពី Python list
# ---------------------------------------------------------------------------
data = [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]]
t = torch.tensor(data)

print("tensor:")
print(t)
print(f"shape    : {tuple(t.shape)}")   # (2, 3) — 2 ជួរ, 3 ឈរ
print(f"dtype    : {t.dtype}")          # torch.float32
print(f"device   : {t.device}")         # cpu
print(f"ndim     : {t.ndim}")           # 2

# ធាតុទាំងអស់ត្រូវតែមានប្រភេទដូចគ្នា — PyTorch សន្មត់ float32 ព្រោះមានចំនួនទសភាគ
assert t.shape == (2, 3)
assert t.dtype == torch.float32

# ---------------------------------------------------------------------------
# ២. ចូលមើលធាតុ (indexing)
# ---------------------------------------------------------------------------
print("-" * 40)
print("t[0]      =", t[0])            # ជួរទី 0 ទាំងមូល -> shape (3,)
print("t[0, 1]   =", t[0, 1])         # ធាតុជួរ 0 ឈរ 1 -> scalar tensor
print("t[:, 2]   =", t[:, 2])         # ឈរទី 2 ទាំងមូល -> shape (2,)
print("t[-1]     =", t[-1])           # ជួរចុងក្រោយ

assert t[0].shape == (3,)
assert t[0, 1].item() == 2.0
assert t[:, 2].shape == (2,)

# ---------------------------------------------------------------------------
# ៣. គណនារវាង tensor (element-wise)
# ---------------------------------------------------------------------------
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

print("-" * 40)
print("a + b =", a + b)        # [11., 22., 33.]
print("a * b =", a * b)        # [10., 40., 90.]  — គុណម្ដងមួយធាតុ មិនមែន matrix multiply
print("b / a =", b / a)        # [10., 10., 10.]
print("a ** 2 =", a ** 2)      # [1., 4., 9.]

assert torch.equal(a + b, torch.tensor([11.0, 22.0, 33.0]))
assert torch.equal(a * b, torch.tensor([10.0, 40.0, 90.0]))

# ---------------------------------------------------------------------------
# ៤. គណនាសង្ខេប (reductions)
# ---------------------------------------------------------------------------
print("-" * 40)
print("a.sum()      =", a.sum().item())       # 6.0
print("a.mean()     =", a.mean().item())      # 2.0
print("a.max()      =", a.max().item())       # 3.0
print("t.sum(dim=0) =", t.sum(dim=0))         # បូកតាមជួរ -> shape (3,)
print("t.sum(dim=1) =", t.sum(dim=1))         # បូកតាមឈរ -> shape (2,)

assert a.sum().item() == 6.0
assert t.sum(dim=0).shape == (3,)
assert t.sum(dim=1).shape == (2,)

# ---------------------------------------------------------------------------
# ៥. យកតម្លៃ Python ចេញពី tensor
# ---------------------------------------------------------------------------
scalar = a.sum()
print("-" * 40)
print("scalar tensor :", scalar)
print(".item()       :", scalar.item())   # 6.0 ជា Python float
print("list          :", a.tolist())      # [1.0, 2.0, 3.0]

assert isinstance(scalar.item(), float)
assert a.tolist() == [1.0, 2.0, 3.0]

print("-" * 40)
print("OK — ឧទាហរណ៍ទាំងអស់ដំណើរការត្រឹមត្រូវ")
