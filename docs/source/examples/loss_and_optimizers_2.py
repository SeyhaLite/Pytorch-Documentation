"""Optimizers៖ SGD, Adam, AdamW និងរបៀបប្រើ param groups។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import copy

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. ការប្រើ optimizer ធម្មតា
# ---------------------------------------------------------------------------
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

print("optimizer:", type(optimizer).__name__)
print("param groups:", len(optimizer.param_groups))
print("hyperparameters ក្នុង group 0:")
for k, v in optimizer.param_groups[0].items():
    if k != "params":
        print(f"  {k:<14} = {v}")
print("ចំនួន tensors ក្នុង group 0:", len(optimizer.param_groups[0]["params"]))

# ---------------------------------------------------------------------------
# ២. វដ្ដ 5 ជំហានដែលមិនអាចខ្វះបាន
# ---------------------------------------------------------------------------
print("-" * 66)
x = torch.randn(64, 10)
y = torch.randn(64, 1)
loss_fn = nn.MSELoss()

for step in range(1, 6):
    optimizer.zero_grad()          # ១. សម្អាត gradient ចាស់
    pred = model(x)                # ២. forward
    loss = loss_fn(pred, y)        # ៣. loss
    loss.backward()                # ៤. backward — បំពេញ .grad
    optimizer.step()               # ៥. ធ្វើបច្ចុប្បន្នភាព weight
    if step % 1 == 0:
        print(f"  step {step}: loss = {loss.item():.6f}")

# ---------------------------------------------------------------------------
# ៣. ប្រៀបធៀប optimizers លើបញ្ហាតែមួយ
# ---------------------------------------------------------------------------
print("-" * 66)
print("ប្រៀបធៀប optimizer លើបញ្ហាតែមួយ (100 steps, ទិន្នន័យដូចគ្នា):")
print(f"{'optimizer':<34}{'loss ដើម':>10}{'loss ចុង':>10}")

xb = torch.randn(128, 10)
yb = (xb[:, :1] * 3.0 - xb[:, 1:2] * 2.0 + 1.0) + 0.1 * torch.randn(128, 1)

configs = [
    ("SGD(lr=0.01)", lambda p: torch.optim.SGD(p, lr=0.01)),
    ("SGD(lr=0.01, momentum=0.9)", lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
    ("Adam(lr=0.01)", lambda p: torch.optim.Adam(p, lr=0.01)),
    ("AdamW(lr=0.01, wd=0.01)", lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.01)),
    ("RMSprop(lr=0.01)", lambda p: torch.optim.RMSprop(p, lr=0.01)),
    ("Adagrad(lr=0.1)", lambda p: torch.optim.Adagrad(p, lr=0.1)),
]

for name, make in configs:
    torch.manual_seed(42)                      # ដើម្បីឱ្យ weight ដើមដូចគ្នា
    m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = make(m.parameters())
    lf = nn.MSELoss()

    first = last = None
    for _ in range(100):
        opt.zero_grad()
        loss = lf(m(xb), yb)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
        last = loss.item()
    print(f"{name:<34}{first:>10.4f}{last:>10.6f}")

# ---------------------------------------------------------------------------
# ៤. Adam vs AdamW — ភាពខុសគ្នាសំខាន់
# ---------------------------------------------------------------------------
print("-" * 66)
print("Adam vs AdamW (weight_decay = 0.1):")
print("  Adam  : បញ្ចូល L2 penalty ទៅក្នុង gradient មុន adaptive scaling")
print("  AdamW : ដក weight ចេញដោយផ្ទាល់ (decoupled) — ត្រឹមត្រូវជាង")
m1 = nn.Linear(4, 4)
m2 = copy.deepcopy(m1)
o1 = torch.optim.Adam(m1.parameters(), lr=0.01, weight_decay=0.1)
o2 = torch.optim.AdamW(m2.parameters(), lr=0.01, weight_decay=0.1)
inp = torch.randn(8, 4)
tgt = torch.randn(8, 4)
for _ in range(5):
    for m, o in ((m1, o1), (m2, o2)):
        o.zero_grad()
        nn.functional.mse_loss(m(inp), tgt).backward()
        o.step()
diff = (m1.weight - m2.weight).abs().max().item()
print(f"  ភាពខុសគ្នាអតិបរមានៃ weight បន្ទាប់ 5 steps: {diff:.6e}")
assert diff > 0, "Adam និង AdamW គួរតែផ្ដល់លទ្ធផលខុសគ្នា"

# ---------------------------------------------------------------------------
# ៥. Parameter groups — lr ផ្សេងគ្នាសម្រាប់ជាន់ផ្សេងគ្នា
# ---------------------------------------------------------------------------
print("-" * 66)
backbone = nn.Linear(10, 32)
head = nn.Linear(32, 1)

opt = torch.optim.SGD([
    {"params": backbone.parameters(), "lr": 1e-4, "name": "backbone"},
    {"params": head.parameters(), "lr": 1e-2, "name": "head", "weight_decay": 0.0},
])
for g in opt.param_groups:
    print(f"  group '{g.get('name')}': lr={g['lr']} weight_decay={g['weight_decay']}")
assert len(opt.param_groups) == 2

# ---------------------------------------------------------------------------
# ៦. state_dict របស់ optimizer (សម្រាប់ checkpoint)
# ---------------------------------------------------------------------------
print("-" * 66)
state = opt.state_dict()
print("optimizer.state_dict() keys:", list(state.keys()))
print("param_groups ក្នុង state_dict:", len(state["param_groups"]))
print("state (momentum buffers ។ល។):", len(state["state"]), "entries")

print("-" * 66)
print("OK — optimizers ទាំងអស់ដំណើរការត្រឹមត្រូវ")
