"""Dataset ផ្ទាល់ខ្លួន៖ អនុវត្ត __len__ និង __getitem__។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
គ្មានការទាញយកទិន្នន័យពីអ៊ីនធឺណិត — ទិន្នន័យត្រូវបានបង្កើតក្នុងកូដ។
"""

from typing import Optional

import torch
from torch.utils.data import Dataset, TensorDataset


# ---------------------------------------------------------------------------
# ១. Dataset សាមញ្ញបំផុត៖ TensorDataset (មានស្រាប់ក្នុង PyTorch)
# ---------------------------------------------------------------------------
torch.manual_seed(0)
features = torch.randn(100, 8)
labels = torch.randint(0, 3, (100,))

tensor_ds = TensorDataset(features, labels)
print("TensorDataset")
print("  len      :", len(tensor_ds))
print("  [0]      :", [tuple(t.shape) for t in tensor_ds[0]])
print("  [0][1]   :", tensor_ds[0][1].item())
assert len(tensor_ds) == 100


# ---------------------------------------------------------------------------
# ២. Dataset ផ្ទាល់ខ្លួន — គំរូដែលអ្នកគួរប្រើសម្រាប់ឯកសារ/រូបភាពពិត
# ---------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    """Dataset គំរូដែលបង្កើតទិន្នន័យនៅពេលហៅ __getitem__ (lazy)។

    នៅក្នុងគម្រោងពិត អ្នកនឹងអានឯកសារនៅទីនេះ ឧទាហរណ៍៖
        path = self.paths[index]
        image = read_image(path)
        label = self.labels[index]
        return self.transform(image), label
    """

    def __init__(
        self,
        size: int = 100,
        features: int = 8,
        num_classes: int = 3,
        seed: int = 0,
        transform: Optional[callable] = None,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # កំណត់ទិន្នន័យជាមុន ដើម្បីឱ្យលទ្ធផលដូចគ្នារាល់ការរត់
        self.x = torch.randn(size, features, generator=g)
        self.y = torch.randint(0, num_classes, (size,), generator=g)
        self.transform = transform

    def __len__(self) -> int:
        """ចំនួនសំណាក — DataLoader ពឹងលើវាសម្រាប់ shuffling និង progress bar។"""
        return self.x.shape[0]

    def __getitem__(self, index: int):
        """ត្រឡប់សំណាកមួយ។ អាចជា tuple (x, y)។

        ចំណាំ៖ __getitem__ ត្រូវតែត្រឡប់ **tensor** (ឬអ្វីដែល default_collate
        អាចដាក់ជា batch បាន)។ បើវាត្រឡប់ numpy array ឬ PIL Image
        អ្នកត្រូវប្ដូរទៅ tensor ក្នុង transform។
        """
        x = self.x[index]
        y = self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


ds = SyntheticDataset(size=50, features=8, num_classes=3, seed=7)
print("\nSyntheticDataset ផ្ទាល់ខ្លួន")
print("  len(ds)      :", len(ds))
x0, y0 = ds[0]
print("  ds[0][0] shape:", tuple(x0.shape), x0.dtype)
print("  ds[0][1]      :", y0.item(), y0.dtype)
assert len(ds) == 50
assert x0.shape == (8,)
assert y0.dtype == torch.int64

# negative index ក៏ដំណើរការដែរ (Python list semantics)
xn, _ = ds[-1]
assert torch.equal(xn, ds[len(ds) - 1][0])
print("  ds[-1] == ds[len-1] : True")

# កំហុសធម្មតា៖ index លើសទំហំ
try:
    ds[9999]
except IndexError as e:
    print("  ds[9999] -> IndexError (ត្រឹមត្រូវ)")


# ---------------------------------------------------------------------------
# ៣. Dataset ជាមួយ transform
# ---------------------------------------------------------------------------
def normalize(t: torch.Tensor) -> torch.Tensor:
    """standardize តាម feature។"""
    return (t - t.mean()) / (t.std() + 1e-8)


ds_t = SyntheticDataset(size=20, seed=3, transform=normalize)
xt, _ = ds_t[0]
print("\nជាមួយ transform:")
print("  mean ≈", round(xt.mean().item(), 6), " std ≈", round(xt.std().item(), 4))
assert abs(xt.mean().item()) < 1e-5

# ---------------------------------------------------------------------------
# ៤. IterableDataset — សម្រាប់ទិន្នន័យហូរ (streaming) ដែលមិនដឹងទំហំ
# ---------------------------------------------------------------------------
from torch.utils.data import IterableDataset


class StreamDataset(IterableDataset):
    """បង្កើតសំណាកម្ដងមួយៗ — មិនចាំបាច់ __len__។"""

    def __init__(self, seed: int = 0, limit: int = 20) -> None:
        super().__init__()
        self.seed = seed
        self.limit = limit

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.limit):
            yield torch.randn(4, generator=g), torch.randint(0, 2, (1,), generator=g).item()


stream = StreamDataset(seed=11, limit=5)
items = list(stream)
print("\nIterableDataset:")
print("  ចំនួនសំណាកដែលទទួល:", len(items))
print("  សំណាកទី 0 shape:", tuple(items[0][0].shape))
assert len(items) == 5

# ព្រមាន៖ IterableDataset មិនគាំទ្រ random access ឬ sampler ធម្មតាទេ
print("  ⚠ IterableDataset មិនអាច ds[i] បានទេ — ត្រូវប្រើ for loop")

# ---------------------------------------------------------------------------
# ៥. វិធីផ្សេងទៀតក្នុងការបង្កើត Dataset
# ---------------------------------------------------------------------------
print("\nវិធីផ្សេងទៀត:")

# ក) ពី dictionary នៃ tensors
d = {"input": torch.randn(30, 5), "target": torch.randn(30, 1)}
print("  dict of tensors -> TensorDataset:", len(TensorDataset(d["input"], d["target"])))

# ខ) ពី list នៃ tuples
pairs = [(torch.tensor([float(i)]), torch.tensor(i % 2)) for i in range(10)]
print("  list of tuples  -> len:", len(pairs))

print("\nOK — Dataset ទាំងអស់ដំណើរការត្រឹមត្រូវ")
