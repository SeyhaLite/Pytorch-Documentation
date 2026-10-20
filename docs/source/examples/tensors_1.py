"""វិធីបង្កើត tensor ទាំងអស់ដែលអ្នកត្រូវការជាប្រចាំ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch

# ---------------------------------------------------------------------------
# ១. ពីទិន្នន័យ Python (list, tuple, numpy)
# ---------------------------------------------------------------------------
from_list = torch.tensor([[1, 2], [3, 4]])
print("from list  :", from_list.tolist(), from_list.dtype)

# torch.tensor() ចម្លងទិន្នន័យជានិច្ច។ បើចង់ចែក memory ជាមួយ numpy
# ប្រើ torch.from_numpy() វិញ។
import numpy as np

arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
from_numpy = torch.from_numpy(arr)
print("from numpy :", from_numpy.tolist(), from_numpy.dtype)

# ---------------------------------------------------------------------------
# ២. តាមរូបរាង (shape) — មិនត្រូវការបញ្ចូលតម្លៃ
# ---------------------------------------------------------------------------
zeros = torch.zeros(2, 3)          # សូន្យទាំងអស់
ones = torch.ones(2, 3)            # មួយទាំងអស់
empty = torch.empty(2, 3)          # មិនបានកំណត់តម្លៃ (អាន memory ចាស់)
full = torch.full((2, 3), 7.0)     # បំពេញដោយតម្លៃជាក់លាក់
eye = torch.eye(3)                 # identity matrix

print("zeros      :", tuple(zeros.shape), zeros.dtype)
print("ones       :", tuple(ones.shape))
print("empty      :", tuple(empty.shape), "(តម្លៃមិនទៀងទាត់ — កុំពឹងវា!)")
print("full       :", full[0].tolist())
print("eye        :", eye.tolist())
assert torch.equal(zeros, torch.zeros_like(ones))
assert torch.equal(eye, torch.diag(torch.ones(3)))

# ---------------------------------------------------------------------------
# ៣. លំដាប់លេខ និង random
# ---------------------------------------------------------------------------
arange = torch.arange(0, 10, 2)    # ដូច Python range: ចាប់ 0 ដល់ 10 រំលង 2
linspace = torch.linspace(0.0, 1.0, steps=5)   # ចែកចន្លោះស្មើគ្នា

torch.manual_seed(1234)            # seed ដើម្បីឱ្យលេខ random ដូចគ្នារាល់ការរត់
rand = torch.rand(2, 3)            # uniform [0, 1)
randn = torch.randn(2, 3)          # normal(0, 1)
randint = torch.randint(0, 10, (2, 3))  # ចំនួនគត់ក្នុង [0, 10)

print("arange     :", arange.tolist())
print("linspace   :", [round(v, 3) for v in linspace.tolist()])
print("rand       :", [round(v, 4) for v in rand.flatten().tolist()])
print("randn      :", [round(v, 4) for v in randn.flatten().tolist()])
print("randint    :", randint.tolist())

assert arange.tolist() == [0, 2, 4, 6, 8]
assert linspace.shape == (5,)
assert rand.min().item() >= 0.0 and rand.max().item() < 1.0
assert randint.min().item() >= 0 and randint.max().item() < 10

# ---------------------------------------------------------------------------
# ៤. បង្កើតតាមគំរូ tensor ដែលមានស្រាប់ (*_like)
# ---------------------------------------------------------------------------
ref = torch.zeros(4, 5, dtype=torch.float16)
like = torch.randn(4, 5).to(ref)                 # ប្ដូរ dtype/device តាម ref
zeros_like = torch.zeros_like(ref)               # រក្សា shape + dtype + device
ones_like = torch.ones_like(ref)

print("-" * 45)
print("ref dtype        :", ref.dtype)
print("zeros_like dtype :", zeros_like.dtype, zeros_like.shape)
assert zeros_like.shape == ref.shape and zeros_like.dtype == ref.dtype

# ---------------------------------------------------------------------------
# ៥. dtype ទាំងអស់ដែលប្រើញឹកញាប់
# ---------------------------------------------------------------------------
dtypes = [torch.float32, torch.float64, torch.float16, torch.bfloat16,
          torch.int64, torch.int32, torch.bool]
print("-" * 45)
for dt in dtypes:
    t = torch.ones(3, dtype=dt)
    print(f"  {str(dt):<18} 1 byte = {t.element_size():<3} ឧ. {t[0].item()}")

print("-" * 45)
print("OK — ការបង្កើត tensor ទាំងអស់ដំណើរការត្រឹមត្រូវ")
