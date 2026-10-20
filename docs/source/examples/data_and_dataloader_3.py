"""Transforms និងការរៀបចំទិន្នន័យមុនចូល model។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
មិនពឹងលើ torchvision — ប្រើតែ torch ដើម្បីឱ្យដំណើរការគ្រប់ environment។
"""

import torch
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. Standardization (normalize) — រូបមន្ត z-score
# ---------------------------------------------------------------------------
print("=" * 66)
print("Standardization")
print("=" * 66)
data = torch.randn(1000, 4) * torch.tensor([1.0, 10.0, 0.1, 100.0]) + 5.0
print(f"មុន : mean={[round(v, 2) for v in data.mean(0).tolist()]}")
print(f"      std ={[round(v, 2) for v in data.std(0).tolist()]}")

# គណនា statistics ពី *training set* តែប៉ុណ្ណោះ ហើយប្រើវាសម្រាប់ទាំងអស់
mu = data.mean(dim=0)
sigma = data.std(dim=0)
normalized = (data - mu) / (sigma + 1e-8)
print(f"បន្ទាប់: mean={[round(v, 6) for v in normalized.mean(0).tolist()]}")
print(f"      std ={[round(v, 4) for v in normalized.std(0).tolist()]}")
assert torch.allclose(normalized.mean(0), torch.zeros(4), atol=1e-5)
assert torch.allclose(normalized.std(0), torch.ones(4), atol=1e-2)

print("\n⚠ កំហុសធម្មតា៖ គណនា mean/std ពីទិន្នន័យទាំងមូល (រួម validation)")
print("   នេះជា data leakage។ គណនាពី train split តែប៉ុណ្ណោះ។")

# ---------------------------------------------------------------------------
# ២. Compose transforms ជា class
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("Compose transforms")
print("=" * 66)


class Compose:
    """ដូច torchvision.transforms.Compose ប៉ុន្តែសាមញ្ញ។"""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        inner = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose([{inner}])"


class Standardize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean, self.std = mean, std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)


class AddNoise:
    """Data augmentation សាមញ្ញសម្រាប់ទិន្នន័យលេខ។"""

    def __init__(self, std: float = 0.01, generator: torch.Generator | None = None) -> None:
        self.std = std
        self.g = generator

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.std * torch.randn(x.shape, generator=self.g)


class ToFloat:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


train_tf = Compose([ToFloat(), Standardize(mu, sigma), AddNoise(0.01)])
eval_tf = Compose([ToFloat(), Standardize(mu, sigma)])     # គ្មាន augmentation ពេលវាយតម្លៃ!
print("train transform:", train_tf)
print("eval  transform:", eval_tf)

sample = data[0]
print("\nសំណាកដើម    :", [round(v, 3) for v in sample.tolist()])
print("បន្ទាប់ train_tf:", [round(v, 3) for v in train_tf(sample).tolist()])
print("បន្ទាប់ eval_tf :", [round(v, 3) for v in eval_tf(sample).tolist()])

# ---------------------------------------------------------------------------
# ៣. Data augmentation សម្រាប់រូបភាព (គំរូដោយប្រើ tensor)
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("Data augmentation សម្រាប់រូបភាព (គំរូ)")
print("=" * 66)
img = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 16.0
print("រូបភាពដើម (1, 4, 4):")
print(img[0])


def random_horizontal_flip(t: torch.Tensor, p: float = 0.5,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """flip ឈរ (width) ដោយចៃដន្យ។ t: (C, H, W)"""
    if torch.rand(1, generator=generator).item() < p:
        return torch.flip(t, dims=[-1])
    return t


def random_crop(t: torch.Tensor, size: int,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """កាត់យកផ្នែក (size, size) ពីទីតាំងចៃដន្យ។"""
    _, h, w = t.shape
    top = torch.randint(0, h - size + 1, (1,), generator=generator).item()
    left = torch.randint(0, w - size + 1, (1,), generator=generator).item()
    return t[:, top:top + size, left:left + size]


g = torch.Generator().manual_seed(5)
for i in range(3):
    aug = random_crop(random_horizontal_flip(img, generator=g), size=3, generator=g)
    print(f"\naugmentation #{i + 1} -> shape {tuple(aug.shape)}")
    print(aug[0])
    assert aug.shape == (1, 3, 3)

print("\n⚠ សំខាន់: អនុវត្ត augmentation តែលើ training set ប៉ុណ្ណោះ។")
print("   validation/test ត្រូវការតែ resize + normalize។")

# ---------------------------------------------------------------------------
# ៤. Dataset ពេញលេញជាមួយ transform
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("Dataset ពេញលេញជាមួយ transform")
print("=" * 66)


class TransformedDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform=None) -> None:
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        item = self.x[index]
        if self.transform is not None:
            item = self.transform(item)
        return item, self.y[index]


labels = torch.randint(0, 3, (1000,))
train_ds = TransformedDataset(data, labels, transform=train_tf)
test_ds = TransformedDataset(data, labels, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(0))
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

xb, yb = next(iter(train_loader))
print(f"train batch : x{tuple(xb.shape)} dtype={xb.dtype}  y{tuple(yb.shape)} dtype={yb.dtype}")
print(f"  x mean={xb.mean().item():+.4f} std={xb.std().item():.4f}")
assert xb.shape == (64, 4) and yb.dtype == torch.int64

xt, yt = next(iter(test_loader))
print(f"test  batch : x{tuple(xt.shape)}  y{tuple(yt.shape)}")

# ---------------------------------------------------------------------------
# ៥. collate_fn ជាមួយ transform កម្រិត batch (លឿនជាង per-sample)
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("Batch-level transform ក្នុង collate_fn")
print("=" * 66)


def batch_collate(batch):
    """ធ្វើ normalize ម្ដងលើ batch ទាំងមូល — លឿនជាង per-sample។"""
    xs = torch.stack([b[0] for b in batch]).float()
    ys = torch.stack([b[1] for b in batch])
    xs = (xs - mu) / (sigma + 1e-8)
    return xs, ys


raw_ds = TransformedDataset(data, labels, transform=None)   # គ្មាន transform
fast_loader = DataLoader(raw_ds, batch_size=64, collate_fn=batch_collate, shuffle=True,
                         generator=torch.Generator().manual_seed(0))
xf, yf = next(iter(fast_loader))
print(f"batch-normalized x shape: {tuple(xf.shape)} mean={xf.mean().item():+.4f}")
assert xf.shape == (64, 4)

print("\nOK — transforms និង data preparation ទាំងអស់ដំណើរការត្រឹមត្រូវ")
