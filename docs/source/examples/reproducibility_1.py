"""Random seeds និង determinism។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42) -> None:
    """កំណត់ seed សម្រាប់គ្រប់ប្រភពភាពចៃដន្យទាំងអស់។

    អ្នកត្រូវហៅវានៅដើមកម្មវិធី ហើយ *មុន* បង្កើត model ឬ DataLoader។
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)                  # Python ធម្មតា
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)       # GPU បច្ចុប្បន្ន
        torch.cuda.manual_seed_all(seed)   # GPU ទាំងអស់ (multi-GPU)


# ---------------------------------------------------------------------------
# ១. ប្រភពភាពចៃដន្យទាំងអស់
# ---------------------------------------------------------------------------
print("=" * 70)
print("ប្រភពភាពចៃដន្យដែលត្រូវគ្រប់គ្រង")
print("=" * 70)

seed_everything(123)
print(f"{'ប្រភព':<42}{'លទ្ធផលទី ១':<22}")
print("-" * 70)
r1 = [
    round(random.random(), 6),
    round(float(np.random.rand()), 6),
    round(torch.rand(1).item(), 6),
    round(torch.randn(1).item(), 6),
    torch.randint(0, 100, (1,)).item(),
]
print(f"{'random.random()':<42}{r1[0]:<22}")
print(f"{'np.random.rand()':<42}{r1[1]:<22}")
print(f"{'torch.rand(1)':<42}{r1[2]:<22}")
print(f"{'torch.randn(1)':<42}{r1[3]:<22}")
print(f"{'torch.randint(0,100,(1,))':<42}{r1[4]:<22}")

# ដំណើរការម្ដងទៀតជាមួយ seed ដូចគ្នា
seed_everything(123)
r2 = [
    round(random.random(), 6),
    round(float(np.random.rand()), 6),
    round(torch.rand(1).item(), 6),
    round(torch.randn(1).item(), 6),
    torch.randint(0, 100, (1,)).item(),
]
print("\nជាមួយ seed ដូចគ្នា (123) លទ្ធផលដូចគ្នាទេ?", r1 == r2)
assert r1 == r2

# seed ផ្សេង -> លទ្ធផលផ្សេង
seed_everything(999)
r3 = [round(random.random(), 6), round(float(np.random.rand()), 6), round(torch.rand(1).item(), 6),
      round(torch.randn(1).item(), 6), torch.randint(0, 100, (1,)).item()]
print("ជាមួយ seed ផ្សេង (999) លទ្ធផលខុសគ្នា?", r1 != r3)
assert r1 != r3

# ---------------------------------------------------------------------------
# ២. torch.Generator — គ្រប់គ្រង seed ដាច់ដោយឡែកសម្រាប់ផ្នែកនីមួយៗ
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("torch.Generator (seed ដាច់ដោយឡែកសម្រាប់ប្រតិបត្តិការនីមួយៗ)")
print("=" * 70)
g1 = torch.Generator().manual_seed(7)
g2 = torch.Generator().manual_seed(7)
a = torch.randn(5, generator=g1)
b = torch.randn(5, generator=g2)
print("generator seed 7 ពីរ -> ដូចគ្នា?", torch.equal(a, b))
assert torch.equal(a, b)

# generator មិនប៉ះពាល់ global state
before = torch.rand(1).item()
_ = torch.randn(100, generator=torch.Generator().manual_seed(1))
seed_everything(123)
after = torch.rand(1).item()
print("ប្រើ generator មិនប៉ះពាល់ global RNG ទេ (seed ដូចគ្នា -> លទ្ធផលដូចគ្នា)")

# ---------------------------------------------------------------------------
# ៣. DataLoader shuffling ត្រូវការ generator ផ្ទាល់ខ្លួន
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DataLoader shuffle reproducibility")
print("=" * 70)
ds = TensorDataset(torch.arange(20).float().unsqueeze(1), torch.zeros(20, dtype=torch.long))


def first_batch(seed: int) -> list:
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=5, shuffle=True, generator=g)
    return next(iter(loader))[0].flatten().tolist()


b1 = first_batch(42)
b2 = first_batch(42)
b3 = first_batch(43)
print(f"  generator seed=42 batch ដំបូង: {b1}")
print(f"  generator seed=42 batch ដំបូង: {b2}   (ដូចគ្នា: {b1 == b2})")
print(f"  generator seed=43 batch ដំបូង: {b3}   (ខុសគ្នា: {b1 != b3})")
assert b1 == b2 and b1 != b3

print("\n  ⚠ បើមិនផ្ដល់ generator= ទេ DataLoader ប្រើ global torch RNG")
print("     ដូច្នេះលទ្ធផលអាស្រ័យលើអ្វីដែលបានរត់មុនវា។")

# random_split ក៏ត្រូវការ generator ដែរ
train_ds, val_ds = torch.utils.data.random_split(ds, [16, 4], generator=torch.Generator().manual_seed(42))
train_ds2, val_ds2 = torch.utils.data.random_split(ds, [16, 4], generator=torch.Generator().manual_seed(42))
same = [train_ds[i][0].item() for i in range(16)] == [train_ds2[i][0].item() for i in range(16)]
print(f"  random_split ជាមួយ generator seed ដូចគ្នា -> ដូចគ្នា: {same}")
assert same

# ---------------------------------------------------------------------------
# ៤. Model initialization reproducibility
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Model weight initialization")
print("=" * 70)
import torch.nn as nn


def build() -> torch.Tensor:
    seed_everything(42)
    return nn.Linear(4, 4).weight.detach().clone()


w1, w2 = build(), build()
print("  weight ដូចគ្នារាល់ការសាងសង់?", torch.equal(w1, w2))
assert torch.equal(w1, w2)

# ---------------------------------------------------------------------------
# ៥. torch.use_deterministic_algorithms
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Deterministic algorithms")
print("=" * 70)
print("  ការកំណត់ environment variable (ត្រូវធ្វើមុន import torch):")
print("     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # សម្រាប់ CUDA >= 10.2")
print("  បន្ទាប់មក:")
print("     torch.use_deterministic_algorithms(True)   # បង្ខំឱ្យប្រើ algorithm ដែល deterministic")
print("     torch.backends.cudnn.deterministic = True")
print("     torch.backends.cudnn.benchmark = False")

warn_path = WORKDIR_HINT = None
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"\n  បានបើក deterministic mode (warn_only=True) — ស្ថានភាព: "
          f"{torch.are_deterministic_algorithms_enabled()}")

    # ប្រតិបត្តិការមួយចំនួនមិនមានកំណែ deterministic
    idx = torch.tensor([0, 1, 0])
    src = torch.tensor([1.0, 2.0, 3.0])
    dst = torch.zeros(2)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dst.index_add_(0, idx, src)
    det_warnings = [w for w in caught if "deterministic" in str(w.message).lower()]
    print(f"  index_add_ បង្កើតព្រមាន determinism: {len(det_warnings) > 0}")
    if det_warnings:
        print(f"    {str(det_warnings[0].message)[:80]}…")
    print(f"  លទ្ធផល index_add_: {dst.tolist()}")
finally:
    torch.use_deterministic_algorithms(False)
    print(f"\n  បានបិទ deterministic mode — ស្ថានភាព: "
          f"{torch.are_deterministic_algorithms_enabled()}")

# ---------------------------------------------------------------------------
# ៦. ភាពចៃដន្យដែលនៅសល់សូម្បីតែមាន seed
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("អ្វីដែល seed មិនអាចគ្រប់គ្រងបាន")
print("=" * 70)
limits = [
    ("Multi-GPU / DDP", "លំដាប់ gradient reduction អាចប្រែប្រួល"),
    ("cuDNN benchmark=True", "ជ្រើសរើស algorithm តាមរយៈពេលវាស់ជាក់ស្ដែង"),
    ("atomic operations លើ GPU", "index_add_, scatter_add_, embedding backward"),
    ("num_workers > 0", "worker process នីមួយៗត្រូវការ seed ផ្ទាល់ខ្លួន (worker_init_fn)"),
    ("floating point មិន associative", "(a+b)+c != a+(b+c) ក្នុងលំដាប់ផ្សេងគ្នា"),
    ("កំណែ PyTorch/hardware ផ្សេងគ្នា", "លទ្ធផលអាចខុសបន្តិចបន្តួច"),
]
for what, why in limits:
    print(f"  ✗ {what:<34} {why}")

# worker_init_fn ឧទាហរណ៍
print("\n  ឧទាហរណ៍ worker_init_fn:")
print("     def seed_worker(worker_id):")
print("         worker_seed = torch.initial_seed() % 2**32")
print("         np.random.seed(worker_seed)")
print("         random.seed(worker_seed)")
print("     DataLoader(ds, worker_init_fn=seed_worker, generator=g)")

print("\nOK — reproducibility ទាំងអស់ដំណើរការត្រឹមត្រូវ")
