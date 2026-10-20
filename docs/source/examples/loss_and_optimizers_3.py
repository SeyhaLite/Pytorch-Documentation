"""Learning rate schedulers និង gradient clipping។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

ច្បាប់សំខាន់៖ ហៅ scheduler.step() **មួយដងក្នុងមួយ epoch**
(ឬមួយដងក្នុងមួយ step សម្រាប់ OneCycleLR) — បន្ទាប់ពី optimizer.step()។
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. មើលក្រាហ្វ learning rate នៃ schedulers ផ្សេងៗ
# ---------------------------------------------------------------------------
def trace(make_scheduler, steps: int, needs_metric: bool = False) -> list[float]:
    """រត់ dummy loop ហើយកត់ត្រា lr រាល់ជំហាន។

    needs_metric=True សម្រាប់ ReduceLROnPlateau ដែលត្រូវការ
    scheduler.step(loss) មិនមែន scheduler.step() ទេ។
    """
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = make_scheduler(opt, steps)
    lrs = []
    # loss សប្បនិម្មិតដែលថយចុះបន្តិចម្ដងៗ ដើម្បីឱ្យ plateau scheduler មានអ្វីមើល
    fake_loss = 1.0
    for step in range(steps):
        opt.zero_grad()
        loss = model(torch.randn(4, 2)).sum()
        loss.backward()
        opt.step()
        lrs.append(opt.param_groups[0]["lr"])
        if needs_metric:
            fake_loss = 1.0 / (step + 1)
            sched.step(fake_loss)          # <- បញ្ជូន metric
        else:
            sched.step()
    return lrs


print("=" * 72)
print(f"{'scheduler':<34}{'lr ដើម':>10}{'lr ចុង':>10}{'lr អប្បបរមា':>12}")
print("=" * 72)

schedulers = [
    ("StepLR(step_size=30, gamma=0.1)", lambda o, n: torch.optim.lr_scheduler.StepLR(o, 30, 0.1)),
    ("MultiStepLR([30,60], gamma=0.1)", lambda o, n: torch.optim.lr_scheduler.MultiStepLR(o, [30, 60], 0.1)),
    ("ExponentialLR(gamma=0.95)", lambda o, n: torch.optim.lr_scheduler.ExponentialLR(o, 0.95)),
    ("CosineAnnealingLR(T_max=100)", lambda o, n: torch.optim.lr_scheduler.CosineAnnealingLR(o, 100)),
    ("ReduceLROnPlateau(patience=5)", lambda o, n: torch.optim.lr_scheduler.ReduceLROnPlateau(o, patience=5, factor=0.5), True),
    ("OneCycleLR(max_lr=0.1, steps=100)", lambda o, n: torch.optim.lr_scheduler.OneCycleLR(o, 0.1, total_steps=n)),
    ("LinearLR(start=0.1, total_iters=20)", lambda o, n: torch.optim.lr_scheduler.LinearLR(o, 0.1, total_iters=20)),
]

N = 100
for entry in schedulers:
    name, make = entry[0], entry[1]
    needs_metric = entry[2] if len(entry) > 2 else False
    lrs = trace(make, N, needs_metric=needs_metric)
    print(f"{name:<34}{lrs[0]:>10.5f}{lrs[-1]:>10.6f}{min(lrs):>12.6f}")
    assert lrs[0] > 0

print("\nចំណាំសំខាន់អំពី ReduceLROnPlateau:")
print("  scheduler ភាគច្រើនហៅ scheduler.step() ដោយគ្មាន argument។")
print("  ReduceLROnPlateau ខុសគេ — វាត្រូវការ metric៖")
print("      scheduler.step(val_loss)     # ហៅម្ដងក្នុងមួយ epoch")
print("  បើភ្លេច អ្នកនឹងទទួល TypeError: step() missing 1 required argument: 'metrics'")

# ---------------------------------------------------------------------------
# ២. Cosine annealing ជាមួយ warm restarts
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
lrs = trace(lambda o, n: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=25), 100)
print("CosineAnnealingWarmRestarts(T_0=25) — lr នៅជំហាន 0, 25, 50, 75:")
for i in [0, 24, 25, 49, 50, 74, 75]:
    print(f"  step {i:>3}: lr = {lrs[i]:.6f}")
assert lrs[0] > lrs[24], "lr គួរតែចុះក្នុងមួយ cycle"

# ---------------------------------------------------------------------------
# ៣. Warmup + cosine decay (រូបមន្តដែលគេប្រើច្រើនបំផុតសម្រាប់ transformer)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Warmup 10 steps -> Cosine decay 90 steps")
warmup = torch.optim.lr_scheduler.LambdaLR(
    torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1),
    lr_lambda=lambda epoch: 1.0,
)

model = nn.Linear(4, 1)
base_lr = 0.1
opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
warm_steps, total_steps = 10, 100

import math


def lr_lambda(step: int) -> float:
    """1.0 ក្នុងអំឡុង warmup បន្ទាប់មក cosine decay ទៅ 0។"""
    if step < warm_steps:
        return (step + 1) / warm_steps
    progress = (step - warm_steps) / max(1, total_steps - warm_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
recorded = []
for step in range(total_steps):
    opt.zero_grad()
    nn.functional.mse_loss(model(torch.randn(8, 4)), torch.randn(8, 1)).backward()
    opt.step()
    recorded.append(opt.param_groups[0]["lr"])
    sched.step()

for i in [0, 4, 9, 10, 50, 99]:
    print(f"  step {i:>3}: lr = {recorded[i]:.6f}")
assert recorded[0] < recorded[9], "warmup ត្រូវតែកើនឡើង"
assert recorded[-1] < recorded[10], "decay ត្រូវតែចុះ"

# ---------------------------------------------------------------------------
# ៤. SequentialLR — ភ្ជាប់ schedulers ច្រើនគ្នា
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
opt2 = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)
seq_sched = torch.optim.lr_scheduler.SequentialLR(
    opt2,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(opt2, start_factor=0.1, total_iters=10),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=40),
    ],
    milestones=[10],
)
combined = []
for _ in range(50):
    combined.append(opt2.param_groups[0]["lr"])
    opt2.step()
    seq_sched.step()
print(f"  step 0 (warmup ចាប់ផ្ដើម): lr = {combined[0]:.6f}")
print(f"  step 9 (warmup បញ្ចប់)  : lr = {combined[9]:.6f}")
print(f"  step 49 (cosine ចុង)     : lr = {combined[49]:.6f}")
assert combined[0] < combined[9]

# ---------------------------------------------------------------------------
# ៥. Gradient clipping — ការពារ exploding gradients
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Gradient clipping")
deep = nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)])
big_in = torch.randn(4, 8) * 50.0
loss = deep(big_in).sum()
loss.backward()

def total_norm(m: nn.Module) -> float:
    return torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=float("inf")).item()

before = total_norm(deep)
torch.nn.utils.clip_grad_norm_(deep.parameters(), max_norm=1.0)
after = total_norm(deep)
print(f"  gradient norm មុន clip : {before:.4f}")
print(f"  gradient norm បន្ទាប់ clip(max_norm=1.0): {after:.4f}")
assert after <= before + 1e-6

# clip_grad_value_ — កំណត់តម្លៃធាតុនីមួយៗ
torch.nn.utils.clip_grad_value_(deep.parameters(), clip_value=0.01)
maxabs = max(p.grad.abs().max().item() for p in deep.parameters())
print(f"  clip_grad_value_(0.01) -> max |grad| = {maxabs:.4f}")
assert maxabs <= 0.01 + 1e-6

print("\nOK — schedulers និង gradient clipping ដំណើរការត្រឹមត្រូវ")
