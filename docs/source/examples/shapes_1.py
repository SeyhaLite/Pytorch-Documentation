"""Shape, Dimensions, Stride និងរបៀបអានវា។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch

# ---------------------------------------------------------------------------
# ១. វិមាត្រមានន័យយ៉ាងដូចម្តេច?
# ---------------------------------------------------------------------------
scalar = torch.tensor(3.14)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.ones(2, 3)
batch = torch.ones(4, 2, 3)
image = torch.ones(8, 3, 32, 32)      # NCHW
seq = torch.ones(2, 10, 512)          # (batch, sequence, hidden)

print(f"scalar : shape={tuple(scalar.shape)}  ndim={scalar.ndim}  numel={scalar.numel()}")
print(f"vector : shape={tuple(vector.shape)}  ndim={vector.ndim}  numel={vector.numel()}")
print(f"matrix : shape={tuple(matrix.shape)}  ndim={matrix.ndim}  numel={matrix.numel()}")
print(f"batch  : shape={tuple(batch.shape)}   ndim={batch.ndim}   numel={batch.numel()}")
print(f"image  : shape={tuple(image.shape)}   <- (batch, channels, height, width)")
print(f"seq    : shape={tuple(seq.shape)}    <- (batch, sequence_length, embedding_dim)")

assert scalar.ndim == 0 and scalar.numel() == 1
assert vector.shape == (3,)
assert image.numel() == 8 * 3 * 32 * 32

# ចំណាំ៖ scalar tensor មាន ndim = 0 ហើយ .item() ទើបយកជា Python float បាន
assert isinstance(scalar.item(), float)

# ---------------------------------------------------------------------------
# ២. Stride — របៀប PyTorch រក្សាទុក tensor ក្នុង memory
# ---------------------------------------------------------------------------
print("-" * 58)
x = torch.arange(12).reshape(3, 4)
print("x =\n", x)
print("x.shape  :", tuple(x.shape))
print("x.stride():", x.stride())
# stride (4, 1) មានន័យថា៖ ដើម្បីទៅជួរបន្ទាប់ លោត 4 ធាតុ; ឈរបន្ទាប់ លោត 1

# ផ្ទៀងផ្ទាត់ដោយដៃ
for i in range(3):
    for j in range(4):
        expected = i * x.stride()[0] + j * x.stride()[1]
        assert x[i, j].item() == x.flatten()[expected].item()
print("ផ្ទៀងផ្ទាត់: x[i,j] == flat[i*stride0 + j*stride1] សម្រាប់គ្រប់ i,j -> True")

# transpose ប្ដូរលំដាប់ stride ប៉ុន្តែមិនប៉ះ memory
xt = x.T
print("\nx.T shape :", tuple(xt.shape))
print("x.T stride:", xt.stride())
print("contiguous:", xt.is_contiguous(), "(stride មិនចុះលំដាប់)")
assert xt.stride() == (1, 4)

# contiguous() ចម្លងទិន្នន័យឱ្យរៀងជាបន្តបន្ទាប់ឡើងវិញ
xc = xt.contiguous()
print("បន្ទាប់ .contiguous() stride:", xc.stride(), "| contiguous:", xc.is_contiguous())
assert xc.is_contiguous()

# ---------------------------------------------------------------------------
# ៣. Tensor memory ចែករួម (views)
# ---------------------------------------------------------------------------
print("-" * 58)
base = torch.arange(6)
view1 = base.view(2, 3)
view2 = base[1:4]

view1[0, 0] = 100
print("កែ view1[0,0]=100 ->")
print("  base  =", base.tolist())
print("  view2 =", view2.tolist(), "(ចាប់ផ្ដើមពី index 1 ដូច្នេះមិនប្ដូរ)")
assert base[0].item() == 100
assert view2.tolist() == [1, 2, 3]

# storage តែមួយ
print("  base.untyped_storage().size()  :", base.untyped_storage().size(), "bytes")
print("  view1.untyped_storage().size() :", view1.untyped_storage().size(), "bytes")

# ---------------------------------------------------------------------------
# ៤. របៀប debug shape — ទម្លាប់ល្អ
# ---------------------------------------------------------------------------
print("-" * 58)
def show(name: str, t: torch.Tensor) -> None:
    """បង្ហាញ shape/dtype/device ជារបៀបងាយអាន។"""
    print(f"{name:<22} shape={str(tuple(t.shape)):<16} dtype={str(t.dtype):<16} dev={t.device}")

show("input", torch.randn(32, 3, 224, 224))
show("after conv", torch.randn(32, 16, 112, 112))
show("after flatten", torch.randn(32, 16 * 112 * 112))
show("logits", torch.randn(32, 10))

print("-" * 58)
print("OK — shape/stride ទាំងអស់ត្រឹមត្រូវ")
