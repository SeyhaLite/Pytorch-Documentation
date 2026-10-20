"""ម៉ូដែល neural network ដំបូង + វដ្ដបណ្ដុះបណ្ដាលខ្លី។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
បង្ហាញ៖ nn.Module, forward pass, loss, backward(), optimizer.step()
ទិន្នន័យ៖ បង្កើតដោយកូដ (synthetic) ដើម្បីឱ្យលទ្ធផលដូចគ្នារាល់ការរត់។
"""

import torch
import torch.nn as nn

# កំណត់ seed ដើម្បីឱ្យលទ្ធផលដូចគ្នារាល់ការរត់
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. ទិន្នន័យសប្បនិម្មិត៖ ចំណុច 2D ដែលត្រូវចែកជា 2 ថ្នាក់
# ---------------------------------------------------------------------------
n = 200
half = n // 2
base = torch.randn(half, 2)
# ថ្នាក់ 0 នៅជុំវិញ (-3, -3)  ថ្នាក់ 1 នៅជុំវិញ (+3, +3) ដូច្នេះវាដាច់ពីគ្នាច្បាស់
features = torch.cat([base - 3.0, base + 3.0], dim=0)
labels = torch.tensor([0] * half + [1] * half)

print("features shape :", tuple(features.shape))   # (200, 2)
print("labels   shape :", tuple(labels.shape))     # (200,)
print("labels dtype   :", labels.dtype)            # torch.int64
assert features.shape == (200, 2)
assert labels.shape == (200,)

# ---------------------------------------------------------------------------
# ២. កំណត់ម៉ូដែល
# ---------------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(2, 16),   # ធាតុចូល 2 -> លាក់ 16
    nn.ReLU(),
    nn.Linear(16, 2),   # លាក់ 16 -> 2 ថ្នាក់
)

n_params = sum(p.numel() for p in model.parameters())
print("-" * 50)
print(model)
print(f"ចំនួន parameters សរុប : {n_params}")
# Linear(2,16): 2*16 weight + 16 bias = 48
# Linear(16,2): 16*2 weight + 2 bias  = 34
assert n_params == 48 + 34, f"រំពឹង 82 parameters តែទទួល {n_params}"

# ---------------------------------------------------------------------------
# ៣. Loss function និង Optimizer
# ---------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ---------------------------------------------------------------------------
# ៤. វដ្ដបណ្ដុះបណ្ដាល (training loop)
# ---------------------------------------------------------------------------
epochs = 30
batch_size = 32

for epoch in range(epochs):
    model.train()                       # បើក dropout / BatchNorm training mode
    epoch_loss = 0.0
    n_batches = 0

    perm = torch.randperm(n)            # ច្របល់ទិន្នន័យរាល់ epoch
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        xb, yb = features[idx], labels[idx]

        optimizer.zero_grad()           # ១. សម្អាត gradient ចាស់
        logits = model(xb)              # ២. forward pass -> (batch, 2)
        loss = loss_fn(logits, yb)      # ៣. គណនា loss
        loss.backward()                 # ៤. backpropagation -> បំពេញ .grad
        optimizer.step()                # ៥. ធ្វើបច្ចុប្បន្នភាព weight

        epoch_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        avg = epoch_loss / n_batches
        print(f"epoch {epoch + 1:>3} | loss = {avg:.4f}")

# ---------------------------------------------------------------------------
# ៥. វាយតម្លៃ
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    preds = model(features).argmax(dim=1)
accuracy = (preds == labels).float().mean().item()

print("-" * 50)
print(f"accuracy លើទិន្នន័យទាំងអស់ : {accuracy:.3f}")
assert accuracy > 0.9, "ម៉ូដែលគួរតែរៀនបានល្អលើទិន្នន័យសាមញ្ញនេះ"
print("OK — ម៉ូបានរៀនដោយជោគជ័យ")
