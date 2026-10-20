"""DataLoader៖ batching, shuffling, workers និង collate_fn។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import time

import torch
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# ទិន្នន័យសប្បនិម្មិត
N, F, C = 100, 8, 3
x = torch.randn(N, F)
y = torch.randint(0, C, (N,))
ds = TensorDataset(x, y)

# ---------------------------------------------------------------------------
# ១. parameters សំខាន់ៗរបស់ DataLoader
# ---------------------------------------------------------------------------
loader = DataLoader(
    ds,
    batch_size=16,
    shuffle=True,          # ច្របល់លំដាប់រាល់ epoch — សំខាន់សម្រាប់ការបណ្ដុះបណ្ដាល
    drop_last=False,       # True = ទម្លាក់ batch ចុងក្រោយបើមិនគ្រប់
    num_workers=0,         # 0 = ផ្ទុកក្នុង process មេ (សាមញ្ញ ប៉ុន្តែយឺត)
    pin_memory=False,      # True សម្រាប់ GPU ដើម្បីប្ដូរទិន្នន័យលឿន
)

print("DataLoader")
print("  dataset size :", len(ds))
print("  batch_size   :", loader.batch_size)
print("  ចំនួន batches:", len(loader))
print("  គណនា        :", f"ceil({N}/{16}) = {(N + 15) // 16}")
assert len(loader) == (N + 15) // 16 == 7

# ដើរលើ batches
print("\nដើរលើ loader:")
for i, (xb, yb) in enumerate(loader):
    if i < 3 or i == len(loader) - 1:
        print(f"  batch {i}: x{tuple(xb.shape)} y{tuple(yb.shape)} y[:4]={yb[:4].tolist()}")
    assert xb.shape[1] == F and yb.ndim == 1

# ---------------------------------------------------------------------------
# ២. shuffle=True ផ្លាស់ប្ដូរលំដាប់រាល់ epoch
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
first_labels_e1 = next(iter(loader))[1][:5].tolist()
first_labels_e2 = next(iter(loader))[1][:5].tolist()
print("epoch 1, 5 labels ដំបូង:", first_labels_e1)
print("epoch 2, 5 labels ដំបូង:", first_labels_e2)
print("(ជាទូទៅខុសគ្នា ព្រោះ shuffle)")

# drop_last=True vs False
loader_drop = DataLoader(ds, batch_size=16, drop_last=True)
print(f"\ndrop_last=False -> {len(loader)} batches (batch ចុងក្រោយមាន {N % 16} សំណាក)")
print(f"drop_last=True  -> {len(loader_drop)} batches (ទម្លាក់ 4 សំណាកចុងក្រោយ)")
assert len(loader_drop) == N // 16
last_batch = list(loader_drop)[-1][0]
assert last_batch.shape[0] == 16, "ជាមួយ drop_last គ្រប់ batch មានទំហំដូចគ្នា"

# ---------------------------------------------------------------------------
# ៣. collate_fn ផ្ទាល់ខ្លួន
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)


def my_collate(batch):
    """batch = list នៃ (x, y) tuples ចំនួន batch_size។

    default_collate ដាក់ tensors ជាមួយគ្នាដោយស្វ័័យប្រវត្តិ។
    collate_fn ផ្ទាល់ខ្លួនមានប្រយោជន៍ពេលសំណាកមានប្រវែងមិនស្មើគ្នា
    (ឧ. ប្រយោគ) — អ្នកត្រូវ pad វា។
    """
    xs, ys = zip(*batch)
    xs = torch.stack(xs)                     # (B, F)
    ys = torch.stack(ys)                     # (B,)
    return {"inputs": xs, "labels": ys, "count": len(xs)}


loader_dict = DataLoader(ds, batch_size=8, collate_fn=my_collate)
b = next(iter(loader_dict))
print("collate_fn ត្រឡប់ dict ជាមួយ keys:", list(b.keys()))
print("  inputs:", tuple(b["inputs"].shape), "| labels:", tuple(b["labels"].shape), "| count:", b["count"])
assert b["inputs"].shape == (8, F)


# padding សម្រាប់លំដាប់ប្រវែងមិនស្មើ
def pad_collate(batch):
    """Pad sequences ទៅប្រវែងអតិបរមានៃ batch នេះ។"""
    lengths = [seq.shape[0] for seq, _ in batch]
    max_len = max(lengths)
    dim = batch[0][0].shape[1]
    padded = torch.zeros(len(batch), max_len, dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.tensor([lab for _, lab in batch])
    for i, (seq, _) in enumerate(batch):
        padded[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = True
    return {"seq": padded, "mask": mask, "labels": labels, "lengths": lengths}


ragged = [(torch.randn(n, 4), i % 2) for i, n in enumerate([7, 3, 10, 5])]
ragged_loader = DataLoader(ragged, batch_size=4, collate_fn=pad_collate)
rb = next(iter(ragged_loader))
print("\npad_collate ជាមួយលំដាប់ប្រវែង [7,3,10,5]:")
print("  seq shape :", tuple(rb["seq"].shape), "(padded ទៅ 10)")
print("  mask[1]   :", rb["mask"][1].tolist())
print("  lengths   :", rb["lengths"])
assert rb["seq"].shape == (4, 10, 4)
assert rb["mask"][1].sum().item() == 3

# ---------------------------------------------------------------------------
# ៤. num_workers និងការវាស់ល្បឿន (លើម៉ាស៊ីននេះ)
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
print("num_workers (ចំណាំ៖ workers > 0 មានប្រយោជន៍តែពេល __getitem__ ធ្ងន់ ឧ. អានរូបភាព):")
big = TensorDataset(torch.randn(20000, 32), torch.randint(0, 2, (20000,)))
for workers in [0, 2]:
    dl = DataLoader(big, batch_size=512, num_workers=workers, shuffle=False)
    t0 = time.perf_counter()
    n = sum(1 for _ in dl)
    dt = time.perf_counter() - t0
    print(f"  num_workers={workers}: {n} batches ក្នុង {dt:.3f}s")

# ---------------------------------------------------------------------------
# ៥. generator សម្រាប់ reproducibility
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
g1 = torch.Generator().manual_seed(123)
g2 = torch.Generator().manual_seed(123)
l1 = DataLoader(ds, batch_size=8, shuffle=True, generator=g1)
l2 = DataLoader(ds, batch_size=8, shuffle=True, generator=g2)
a = next(iter(l1))[1].tolist()
b2 = next(iter(l2))[1].tolist()
print("ជាមួយ generator seed ដូចគ្នា, batch ដំបូងដូចគ្នា?", a == b2)
assert a == b2

print("\nOK — DataLoader ទាំងអស់ដំណើរការត្រឹមត្រូវ")
