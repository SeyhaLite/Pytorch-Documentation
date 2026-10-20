"""ការផ្លាស់ប្ដូររូបរាង៖ reshape, view, permute, transpose, cat, stack។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch

x = torch.arange(24).reshape(2, 3, 4)
print("x shape:", tuple(x.shape))

# ---------------------------------------------------------------------------
# ១. view() និង reshape()
# ---------------------------------------------------------------------------
v = x.view(6, 4)          # តម្រូវឱ្យ tensor contiguous
r = x.reshape(6, 4)       # សុវត្ថិភាពជាង — ចម្លងទិន្នន័យបើចាំបាច់
print("view(6,4)    :", tuple(v.shape))
print("reshape(6,4) :", tuple(r.shape))
assert v.shape == r.shape == (6, 4)

# ប្រើ -1 ដើម្បីឱ្យ PyTorch គណនាវិមាត្រមួយដោយខ្លួនឯង
auto = x.reshape(-1, 4)
print("reshape(-1,4):", tuple(auto.shape))
assert auto.shape == (6, 4)

flatten = x.reshape(-1)
print("reshape(-1)  :", tuple(flatten.shape))
assert flatten.shape == (24,)

# view() បរាជ័យលើ tensor មិន contiguous
xt = x.transpose(0, 1)    # shape (3, 2, 4) — មិន contiguous
try:
    xt.view(-1)
    raise SystemExit("គួរតែបរាជ័យ")
except RuntimeError as e:
    print("-" * 55)
    print("view() លើ tensor មិន contiguous -> RuntimeError:")
    print(" ", str(e).splitlines()[0][:80])
    fixed = xt.contiguous().view(-1)
    print("ដំណោះស្រាយ: .contiguous().view(-1) shape =", tuple(fixed.shape))
    assert fixed.shape == (24,)

# ---------------------------------------------------------------------------
# ២. transpose() និង permute()
# ---------------------------------------------------------------------------
print("-" * 55)
print("transpose(0,1) :", tuple(x.transpose(0, 1).shape))   # (3, 2, 4)
print("permute(2,0,1) :", tuple(x.permute(2, 0, 1).shape))    # (4, 2, 3)
print(".T (2D only)   :", tuple(torch.arange(6).reshape(2, 3).T.shape))

# permute មានប្រយោជន៍សម្រាប់រូបភាព៖ NHWC -> NCHW
img_nhwc = torch.randn(8, 32, 32, 3)        # (batch, H, W, channels)
img_nchw = img_nhwc.permute(0, 3, 1, 2)     # (batch, channels, H, W)
print("NHWC", tuple(img_nhwc.shape), "-> NCHW", tuple(img_nchw.shape))
assert img_nchw.shape == (8, 3, 32, 32)

# ---------------------------------------------------------------------------
# ៣. unsqueeze() និង squeeze()
# ---------------------------------------------------------------------------
print("-" * 55)
t = torch.zeros(3, 4)
print("t                :", tuple(t.shape))
print("t.unsqueeze(0)   :", tuple(t.unsqueeze(0).shape))   # (1, 3, 4) — បន្ថែម batch dim
print("t.unsqueeze(-1)  :", tuple(t.unsqueeze(-1).shape))  # (3, 4, 1)
print("t.unsqueeze(0).squeeze(0) :", tuple(t.unsqueeze(0).squeeze(0).shape))

# squeeze() ដកវិមាត្រដែលមានទំហំ 1 ចេញ — ប្រយ័ត្ន!
s = torch.zeros(1, 3, 1, 4)
print("squeeze() លើ", tuple(s.shape), "->", tuple(s.squeeze().shape))
assert s.squeeze().shape == (3, 4)

# ---------------------------------------------------------------------------
# ៤. torch.cat() vs torch.stack()
# ---------------------------------------------------------------------------
print("-" * 55)
p = torch.ones(2, 3)
q = torch.zeros(2, 3)

cat0 = torch.cat([p, q], dim=0)   # បន្ថែមតាមវិមាត្រដែលមាន
stack0 = torch.stack([p, q], dim=0)  # បង្កើតវិមាត្រថ្មី

print("cat([p,q], dim=0)   :", tuple(cat0.shape))     # (4, 3)
print("stack([p,q], dim=0) :", tuple(stack0.shape))   # (2, 2, 3)
assert cat0.shape == (4, 3)
assert stack0.shape == (2, 2, 3)

# កំហុសធម្មតា៖ cat លើវិមាត្រខុស
try:
    torch.cat([torch.ones(2, 3), torch.ones(2, 5)], dim=0)
    raise SystemExit("គួរតែបរាជ័យ")
except RuntimeError as e:
    print("cat ដែល shape មិនត្រូវគ្នា -> RuntimeError:")
    print(" ", str(e).splitlines()[0][:80])

# ---------------------------------------------------------------------------
# ៥. split() និង chunk()
# ---------------------------------------------------------------------------
print("-" * 55)
big = torch.arange(10)
print("split(3)  ->", [tuple(s.shape) for s in torch.split(big, 3)])
print("chunk(3)  ->", [tuple(c.shape) for c in torch.chunk(big, 3)])
assert [s.numel() for s in torch.split(big, 3)] == [3, 3, 3, 1]

# ---------------------------------------------------------------------------
# ៦. repeat() vs expand() — expand មិនចម្លង memory
# ---------------------------------------------------------------------------
print("-" * 55)
e = torch.tensor([[1.0, 2.0]])
print("expand(3, 2)  :", tuple(e.expand(3, 2).shape), "| memory ចែករួម")
print("repeat(3, 1)  :", tuple(e.repeat(3, 1).shape), "| ចម្លង memory ពិត")
print("expand stride :", e.expand(3, 2).stride(), "(0 មានន័យថាគ្មានការចម្លង)")
assert e.expand(3, 2).stride()[0] == 0

print("-" * 55)
print("OK — ការផ្លាស់ប្ដូររូបរាងទាំងអស់ដំណើរការត្រឹមត្រូវ")
