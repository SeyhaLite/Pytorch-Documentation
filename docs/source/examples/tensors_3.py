"""Indexing និង Slicing លើ tensor។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
ច្បាប់សំខាន់៖ indexing ភាគច្រើនត្រឡប់ *view* (ចែក memory រួមគ្នា)។
លើកលែងតែ advanced/boolean indexing ដែលចម្លងទិន្នន័យ។
"""

import torch

x = torch.arange(20).reshape(4, 5)
print("x =")
print(x)
print("-" * 50)

# ១. Index ធម្មតា
print("x[0]     :", x[0].tolist())        # ជួរទី 0
print("x[0, 2]  :", x[0, 2].item())       # ធាតុជាក់លាក់
print("x[-1]    :", x[-1].tolist())       # ជួរចុងក្រោយ
assert x[0, 2].item() == 2

# ២. Slicing  [start:stop:step]
print("x[1:3]   :", x[1:3].tolist())      # ជួរ 1 និង 2 (មិនរាប់ 3)
print("x[:, 1]  :", x[:, 1].tolist())     # ឈរទី 1 ទាំងមូល
print("x[:, -2:]:", x[:, -2:].tolist())   # ឈរ 2 ចុងក្រោយ
print("x[::2]   :", x[::2].tolist())      # រំលងរៀងរាល់ជួរ
assert x[:, 1].tolist() == [1, 6, 11, 16]

# ៣. Ellipsis (…) = "វិមាត្រដែលនៅសល់ទាំងអស់"
y = torch.arange(24).reshape(2, 3, 4)
print("-" * 50)
print("y shape      :", tuple(y.shape))
print("y[..., 0]    :", y[..., 0].tolist())   # យក index 0 នៃវិមាត្រចុងក្រោយ
print("y[0, ...]    :", y[0, ...].tolist())
assert y[..., 0].shape == (2, 3)

# ៤. Advanced indexing (list ឬ tensor នៃ index) -> ចម្លងទិន្នន័យ
rows = torch.tensor([0, 3])
picked = x[rows]
print("-" * 50)
print("x[[0,3]]   :", picked.tolist())
assert picked.shape == (2, 5)

# ៥. Boolean masking — មានប្រយោជន៍បំផុតសម្រាប់ស្វែងរកតម្លៃ
mask = x > 10
print("mask       :")
print(mask)
print("x[x > 10]  :", x[mask].tolist())          # tensor 1D នៃតម្លៃដែលត្រូវ
print("ចំនួន      :", mask.sum().item())
assert x[mask].tolist() == [11, 12, 13, 14, 15, 16, 17, 18, 19]

# ៦. torch.where(លក្ខខណ្ឌ, បើពិត, បើខុស) — រក្សា shape
clamped = torch.where(x > 10, x, torch.zeros_like(x))
print("-" * 50)
print("torch.where(x > 10, x, 0) =")
print(clamped)
assert clamped.shape == x.shape
# ធាតុដែល x <= 10 ត្រូវបានជំនួសដោយ 0 ដូច្នេះគ្មានតម្លៃណាក្នុងចន្លោះ 1..10
assert clamped[(clamped >= 1) & (clamped <= 10)].numel() == 0

# ៧. nonzero() និង take_along_dim
idx = torch.nonzero(x > 15, as_tuple=False)
print("nonzero(x > 15) =")
print(idx)
assert idx.shape[1] == 2   # (ចំនួនធាតុ, 2 វិមាត្រ)

# ៨. ការកែតម្លៃតាម index
z = x.clone()
z[0, 0] = -1               # កែធាតុតែមួយ
z[:, 4] = 0                # កែឈរទាំងមូល
z[z < 0] = 99              # កែតាម mask
print("-" * 50)
print("z បន្ទាប់ពីកែ =")
print(z)
assert z[0, 0].item() == 99
assert (z[:, 4] == 0).all()

# ---------------------------------------------------------------------------
# កំហុសដែលជួបញឹកញាប់
# ---------------------------------------------------------------------------
print("-" * 50)
try:
    bad = x[1.5]           # index ត្រូវតែជាចំនួនគត់
except IndexError as e:
    print("IndexError  (ត្រឹមត្រូវ):", str(e).split("\n")[0][:70])

try:
    bad = x[10]            # index លើសទំហំ
except IndexError as e:
    print("IndexError  (ត្រឹមត្រូវ):", str(e).split("\n")[0][:70])

print("-" * 50)
print("OK — indexing និង slicing ដំណើរការត្រឹមត្រូវ")
