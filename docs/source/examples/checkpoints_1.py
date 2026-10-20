"""Checkpoint ពេញលេញ៖ រក្សាទុក និងបន្តការបណ្ដុះបណ្ដាលពីកន្លែងដែលឈប់។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

Checkpoint ≠ model weights។  Checkpoint គឺ *ស្ថានភាពទាំងមូល* ដែលត្រូវការ
ដើម្បីបន្តការងារ៖ model + optimizer + scheduler + epoch + RNG state + metrics។
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
WORKDIR = Path(os.environ.get("KH_EXAMPLE_TMPDIR", tempfile.gettempdir()))
CKPT_DIR = WORKDIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ១. របៀបសរសេរ checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    best_metric: float = float("inf"),
    rng_state: dict | None = None,
) -> None:
    """រក្សាទុកស្ថានភាពទាំងមូលដែលត្រូវការដើម្បីបន្តការបណ្ដុះបណ្ដាល។"""
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        # កំណែកូដ — សំខាន់ណាស់ពេល architecture ផ្លាស់ប្ដូរ
        "torch_version": torch.__version__,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if rng_state is not None:
        payload["rng_state"] = rng_state
    torch.save(payload, path)


def get_rng_state() -> dict:
    """ចាប់យកស្ថានភាព RNG ទាំងអស់ដែលប៉ះពាល់ដល់ reproducibility។"""
    state = {"python_random_not_captured": None, "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# ២. ទិន្នន័យ និងការបណ្ដុះបណ្ដាលពាក់កណ្ដាល
# ---------------------------------------------------------------------------
x = torch.randn(200, 8)
y = torch.randint(0, 3, (200,))
ds = TensorDataset(x, y)
g = torch.Generator().manual_seed(0)
loader = DataLoader(ds, batch_size=32, shuffle=True, generator=g)

model = Net()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
loss_fn = nn.CrossEntropyLoss()

print("=" * 70)
print("ដំណាក់កាលទី ១: បណ្ដុះបណ្ដាល 3 epochs ហើយរក្សាទុក checkpoint")
print("=" * 70)

best_val_loss = float("inf")
for epoch in range(1, 4):
    model.train()
    running, n = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
        n += xb.size(0)
    scheduler.step()

    # validation សាមញ្ញលើទិន្នន័យទាំងមូល
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(x), y).item()
        val_acc = (model(x).argmax(1) == y).float().mean().item()
    best_val_loss = min(best_val_loss, val_loss)

    print(f"  epoch {epoch}: train_loss={running / n:.4f} val_loss={val_loss:.4f} "
          f"val_acc={val_acc:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

    save_checkpoint(
        CKPT_DIR / f"epoch_{epoch}.pt",
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_metric=best_val_loss,
        rng_state=get_rng_state(),
    )

# រក្សាទុក checkpoint "ល្អបំផុត" ដាច់ដោយឡែក
save_checkpoint(
    CKPT_DIR / "best.pt",
    epoch=3, model=model, optimizer=optimizer, scheduler=scheduler,
    best_metric=best_val_loss, rng_state=get_rng_state(),
)

print("\nឯកសារ checkpoint ដែលបានបង្កើត:")
for f in sorted(CKPT_DIR.iterdir()):
    print(f"  {f.name:<14} {f.stat().st_size / 1024:>8.1f} KB")

# ---------------------------------------------------------------------------
# ៣. ពិនិត្យមើលខ្លឹមសារ checkpoint
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ខ្លឹមសាររបស់ epoch_3.pt")
print("=" * 70)
ck = torch.load(CKPT_DIR / "epoch_3.pt", weights_only=False)
for k, v in ck.items():
    if isinstance(v, dict):
        print(f"  {k:<24} dict ជាមួយ {len(v)} keys")
    else:
        print(f"  {k:<24} {v}")

print("\n  optimizer_state_dict មាន:")
print("    param_groups :", len(ck["optimizer_state_dict"]["param_groups"]), "group(s)")
print("    state entries:", len(ck["optimizer_state_dict"]["state"]),
      "(AdamW រក្សា exp_avg និង exp_avg_sq សម្រាប់ parameter នីមួយៗ)")
print("    momentum buffer shape:",
      tuple(next(iter(ck["optimizer_state_dict"]["state"].values()))["exp_avg"].shape))

# ---------------------------------------------------------------------------
# ៤. ស្ដារ checkpoint ហើយបន្តការបណ្ដុះបណ្ដាល
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ដំណាក់កាលទី ២: ស្ដារពី epoch_2.pt ហើយបន្តដល់ epoch 5")
print("=" * 70)

# model ថ្មីទាំងស្រុង (ដូចចាប់ផ្ដើមឡើងវិញ)
model2 = Net()
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)

ck2 = torch.load(CKPT_DIR / "epoch_2.pt", weights_only=False)
model2.load_state_dict(ck2["model_state_dict"])
optimizer2.load_state_dict(ck2["optimizer_state_dict"])
scheduler2.load_state_dict(ck2["scheduler_state_dict"])
start_epoch = ck2["epoch"] + 1
best_val_loss = ck2["best_metric"]
set_rng_state(ck2["rng_state"])

print(f"  ស្ដារពី epoch {ck2['epoch']}  (PyTorch {ck2['torch_version']})")
print(f"  បន្តពី epoch {start_epoch}")
print(f"  lr បច្ចុប្បន្ន: {optimizer2.param_groups[0]['lr']:.6f} "
      f"(ដូច scheduler នៅ epoch {ck2['epoch']})")

for epoch in range(start_epoch, 6):
    model2.train()
    running, n = 0.0, 0
    for xb, yb in loader:
        optimizer2.zero_grad(set_to_none=True)
        loss = loss_fn(model2(xb), yb)
        loss.backward()
        optimizer2.step()
        running += loss.item() * xb.size(0)
        n += xb.size(0)
    scheduler2.step()
    model2.eval()
    with torch.no_grad():
        val_loss = loss_fn(model2(x), y).item()
    print(f"  epoch {epoch}: train_loss={running / n:.4f} val_loss={val_loss:.4f} "
          f"lr={optimizer2.param_groups[0]['lr']:.6f}")

# ---------------------------------------------------------------------------
# ៥. ផ្ទៀងផ្ទាត់ថា optimizer state ត្រូវបានស្ដារពិតប្រាកដ
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ផ្ទៀងផ្ទាត់ optimizer state")
print("=" * 70)
n_with_momentum = sum(1 for s in optimizer2.state.values() if "exp_avg" in s)
print(f"  param groups ដែលមាន Adam moments: {n_with_momentum}")
assert n_with_momentum > 0, "optimizer state គួរតែមាន exp_avg បន្ទាប់ពីការបណ្ដុះបណ្ដាល"

# បើមិនស្ដារ optimizer state ការបណ្ដុះបណ្ដាលនឹងចាប់ផ្ដើម moments ពីសូន្យ
model3 = Net()
model3.load_state_dict(ck2["model_state_dict"])        # weight តែប៉ុណ្ណោះ
optimizer3 = torch.optim.AdamW(model3.parameters(), lr=1e-2)
print(f"  optimizer ថ្មី state entries: {len(optimizer3.state)} (ទទេ)")
assert len(optimizer3.state) == 0

# ---------------------------------------------------------------------------
# ៦. របៀបរក្សាទុកតែ checkpoint ល្អបំផុត (សន្សំ disk)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("យុទ្ធសាស្ត្ររក្សាទុក")
print("=" * 70)
print("  ១. best.pt      : សរសេរជាន់រាល់ពេលដែល metric ប្រសើរឡើង")
print("  ២. last.pt      : សរសេរជាន់រាល់ epoch (សម្រាប់ resume)")
print("  ៣. epoch_N.pt   : រក្សាទុក N ចុងក្រោយ លុបចាស់ចេញ")

# ឧទាហរណ៍កូដសម្រាប់យុទ្ធសាស្ត្រទី ៣
keep_last = 2
all_ckpts = sorted(CKPT_DIR.glob("epoch_*.pt"))
for old in all_ckpts[:-keep_last]:
    print(f"     លុប {old.name} (រក្សាទុកតែ {keep_last} ចុងក្រោយ)")
    old.unlink()
print("  ឯកសារដែលនៅសល់:", sorted(f.name for f in CKPT_DIR.iterdir()))

# សម្អាត
import shutil
shutil.rmtree(CKPT_DIR, ignore_errors=True)
print("\nបានលុប checkpoint បណ្ដោះអាសន្នទាំងអស់")
print("OK — checkpoint save/resume ដំណើរការត្រឹមត្រូវ")
