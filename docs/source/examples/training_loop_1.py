"""Training loop ពេញលេញដែលអាចប្រើក្នុងគម្រោងពិតបាន។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
រយៈពេល៖ ប្រហែល 3-6 វិនាទីលើ CPU ធម្មតា។

គម្រូនេះរួមបញ្ចូលអ្វីដែលគម្រោងពិតត្រូវការ៖
  - train/val split ដោយ reproducible
  - model.train() / model.eval()
  - torch.no_grad() សម្រាប់ validation
  - gradient clipping
  - learning rate schedule
  - early stopping + រក្សាទុក weight ល្អបំផុត
  - ការវាស់រយៈពេល និងការកត់ត្រា metrics
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# ការកំណត់ (hyperparameters)
# ---------------------------------------------------------------------------
SEED = 42
N_TRAIN, N_VAL = 800, 200
IN_FEATURES, NUM_CLASSES = 16, 4
HIDDEN = 32
BATCH_SIZE = 32
EPOCHS = 6
LR = 1e-2
WEIGHT_DECAY = 1e-3
CLIP_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Metrics:
    """កត់ត្រា metrics រាល់ epoch។"""
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ១. Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# ២. ទិន្នន័យសប្បនិម្មិតដែលអាចបែងចែកបាន
# ---------------------------------------------------------------------------
def make_loaders(seed: int) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    total = N_TRAIN + N_VAL
    x = torch.randn(total, IN_FEATURES, generator=g)
    # បង្កើត label ពី linear projection + noise ដើម្បីឱ្យរៀនបាន
    w = torch.randn(IN_FEATURES, NUM_CLASSES, generator=g)
    logits = x @ w
    y = logits.argmax(dim=1)

    ds = TensorDataset(x, y)
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [N_TRAIN, N_VAL],
        generator=torch.Generator().manual_seed(seed),   # split ដូចគ្នារាល់ការរត់
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# ៣. ម៉ូដែល
# ---------------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ៤. វដ្ដមួយ epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """គណនា loss និង accuracy លើ loader ទាំងមូល។"""
    model.eval()                       # <- សំខាន់! បិទ Dropout, ប្រើ BN running stats
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_samples += xb.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()                      # <- សំខាន់! បើក Dropout, BN រៀន statistics
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # ជំនួស zero_grad() ធម្មតា — សន្សំ memory
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()

        # gradient clipping មុន optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_samples += xb.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# ៥. ដំណើរការបណ្ដុះបណ្ដាលទាំងមូល
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    print(f"device          : {device}")
    print(f"train / val     : {N_TRAIN} / {N_VAL}")
    print(f"epochs          : {EPOCHS}")
    print(f"batch size      : {BATCH_SIZE}")
    print(f"lr / weight_dec : {LR} / {WEIGHT_DECAY}")
    print("=" * 66)

    train_loader, val_loader = make_loaders(SEED)
    model = Classifier(IN_FEATURES, HIDDEN, NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters      : {n_params:,}")
    print("-" * 66)
    print(f"{'epoch':>5} | {'train loss':>10} {'train acc':>10} | "
          f"{'val loss':>9} {'val acc':>8} | {'lr':>8} | {'sec':>5}")
    print("-" * 66)

    metrics = Metrics()
    best_val_loss = float("inf")
    best_state = None
    patience, bad_epochs = 2, 0
    overall_t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.perf_counter() - t0

        metrics.train_loss.append(train_loss)
        metrics.train_acc.append(train_acc)
        metrics.val_loss.append(val_loss)
        metrics.val_acc.append(val_acc)
        metrics.epoch_seconds.append(dt)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # ចម្លង state_dict ទុក (clone ជាន់គ្នា ព្រោះ state_dict ចែក tensor)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            marker = "  <- ល្អបំផុត"
        else:
            bad_epochs += 1
            marker = f"  (មិនប្រសើរ {bad_epochs}/{patience})"

        print(f"{epoch:>5} | {train_loss:>10.4f} {train_acc:>10.4f} | "
              f"{val_loss:>9.4f} {val_acc:>8.4f} | {lr_now:>8.6f} | {dt:>5.2f}{marker}")

        scheduler.step()               # ម្ដងក្នុងមួយ epoch បន្ទាប់ពី optimizer.step()

        if bad_epochs >= patience and epoch < EPOCHS:
            print(f"\nearly stopping នៅ epoch {epoch} (patience={patience})")
            break

    # ស្ដារ weight ល្អបំផុត
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nបានស្ដារ weight នៃ epoch ល្អបំផុត (val_loss = {best_val_loss:.4f})")

    print("=" * 66)
    print(f"រយៈពេលសរុប: {time.perf_counter() - overall_t0:.2f}s")
    final_val_loss, final_val_acc = evaluate(model, val_loader, loss_fn, device)
    print(f"val loss ចុងក្រោយ: {final_val_loss:.4f}")
    print(f"val acc  ចុងក្រោយ: {final_val_acc:.4f}")

    # ការផ្ទៀងផ្ទាត់ដោយស្វ័យប្រវត្តិ (ធ្វើឱ្យឯកសារនេះអាចសាកល្បងក្នុង CI)
    assert len(metrics.train_loss) >= 1
    assert all(0.0 <= a <= 1.0 for a in metrics.val_acc)
    assert final_val_acc > 0.6, "ម៉ូដែលគួរតែរៀនបានលើសពីការទាយចៃដន្យ (25%)"
    assert final_val_loss < 1.2, "loss គួរតែថយចុះពីតម្លៃដើម (~ln(4)=1.386)"
    print("\nOK — training loop ដំណើរការត្រឹមត្រូវ (accuracy > 60%, loss ថយចុះ)")


if __name__ == "__main__":
    main()
