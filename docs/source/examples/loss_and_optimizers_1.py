"""Loss functions៖ ជ្រើសរើសឱ្យត្រូវនឹងបញ្ហា និងយល់ពី input/output shape។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

ច្បាប់មាសសម្រាប់ classification:
    nn.CrossEntropyLoss  ទទួល **logits** (មិនទាន់ softmax) + **class index** (int64)
    មិនមែន one-hot ទេ!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. CrossEntropyLoss — multi-class classification
# ---------------------------------------------------------------------------
print("=" * 68)
print("nn.CrossEntropyLoss  (multi-class classification)")
print("=" * 68)
batch, classes = 4, 3
logits = torch.randn(batch, classes)        # (4, 3) — លេខឆៅ មិនទាន់ softmax
target = torch.tensor([0, 2, 1, 1])         # (4,)   — class index, dtype int64

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)
print("logits :", tuple(logits.shape), logits.dtype)
print("target :", tuple(target.shape), target.dtype)
print("loss   :", loss.item(), "(scalar)")
assert loss.ndim == 0

# ផ្ទៀងផ្ទាត់ថាវា = LogSoftmax + NLLLoss
manual = F.nll_loss(F.log_softmax(logits, dim=1), target)
print("ផ្ទៀងផ្ទាត់ nll_loss(log_softmax(...)) =", round(manual.item(), 6))
assert torch.allclose(loss, manual)

# កំហុសធម្មតាទី ១: ផ្ដល់ softmax រួចហើយ (double softmax)
wrong = loss_fn(torch.softmax(logits, dim=1), target)
print(f"\n⚠ softmax រួចហើយបញ្ចូល CrossEntropyLoss -> loss = {wrong.item():.4f}")
print(f"  loss ត្រឹមត្រូវ = {loss.item():.4f}  (ខុសគ្នា! gradient នឹងខូច)")

# កំហុសធម្មតាទី ២: ផ្ដល់ one-hot ជំនួស class index
try:
    one_hot = F.one_hot(target, classes).float()
    loss_fn(logits, one_hot)
    print("\n⚠ one-hot ត្រូវបានទទួល ប៉ុន្តែតម្លៃខុសគ្នា — កុំធ្វើបែបនេះ")
except Exception as e:
    print("\n⚠ one-hot ->", type(e).__name__, str(e).splitlines()[0][:60])

# ---------------------------------------------------------------------------
# ២. BCEWithLogitsLoss — binary classification (ឬ multi-label)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("nn.BCEWithLogitsLoss  (binary / multi-label)")
print("=" * 68)
logits_b = torch.randn(4, 1)                 # (4, 1)
target_b = torch.tensor([[1.0], [0.0], [1.0], [0.0]])   # float មិនមែន int!

bce = nn.BCEWithLogitsLoss()
loss_b = bce(logits_b, target_b)
print("logits :", tuple(logits_b.shape))
print("target :", tuple(target_b.shape), target_b.dtype, "(ត្រូវតែ float)")
print("loss   :", loss_b.item())
assert loss_b.ndim == 0

# Multi-label: ច្រើនថ្នាក់ក្នុងពេលតែមួយ (ឧ. រូបភាពមានទាំងឆ្មា និងឆ្កែ)
logits_m = torch.randn(4, 5)
target_m = (torch.rand(4, 5) > 0.6).float()
loss_m = nn.BCEWithLogitsLoss()(logits_m, target_m)
print(f"\nmulti-label: logits {tuple(logits_m.shape)}, target {tuple(target_m.shape)}, loss {loss_m.item():.4f}")

# កំហុសធម្មតា: target dtype ខុស
try:
    bce(logits_b, torch.tensor([1, 0, 1, 0]).reshape(4, 1))   # int64
except Exception as e:
    print("⚠ target int64 ->", type(e).__name__, str(e).splitlines()[0][:65])

# ---------------------------------------------------------------------------
# ៣. MSELoss / L1Loss — regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("nn.MSELoss / nn.L1Loss  (regression)")
print("=" * 68)
pred = torch.randn(4, 2)
true = torch.randn(4, 2)

mse = nn.MSELoss()(pred, true)
l1 = nn.L1Loss()(pred, true)
huber = nn.SmoothL1Loss()(pred, true)
print(f"MSELoss       : {mse.item():.4f}   (អស្សិលភាពខ្ពស់ចំពោះ outlier)")
print(f"L1Loss        : {l1.item():.4f}   (ធន់នឹង outlier ជាង)")
print(f"SmoothL1Loss  : {huber.item():.4f}   (Huber — ល្អបំផុតសម្រាប់ភាគច្រើន)")
assert mse.ndim == 0

# reduction options
print("\nreduction options:")
for red in ["mean", "sum", "none"]:
    out = nn.MSELoss(reduction=red)(pred, true)
    print(f"  '{red}' -> shape {tuple(out.shape)}")
assert nn.MSELoss(reduction="none")(pred, true).shape == pred.shape

# ---------------------------------------------------------------------------
# ៤. NLLLoss និងការប្រើ class weights សម្រាប់ទិន្នន័យមិនស្មើគ្នា
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("class weights សម្រាប់ imbalanced dataset")
print("=" * 68)
labels = torch.tensor([0] * 90 + [1] * 10)         # 90% ថ្នាក់ 0
counts = torch.bincount(labels).float()
weights = counts.sum() / (len(counts) * counts)     # inverse frequency
print("ចំនួនតាមថ្នាក់   :", counts.tolist())
print("computed weights :", [round(w, 3) for w in weights.tolist()])

logits_im = torch.randn(100, 2)
plain = nn.CrossEntropyLoss()(logits_im, labels).item()
weighted = nn.CrossEntropyLoss(weight=weights)(logits_im, labels).item()
print(f"\nloss គ្មាន weight : {plain:.4f}")
print(f"loss មាន weight   : {weighted:.4f}")

# label smoothing — ការពារ overconfidence
smooth = nn.CrossEntropyLoss(label_smoothing=0.1)(logits_im, labels).item()
print(f"loss ជាមួយ label_smoothing=0.1 : {smooth:.4f}")

# ---------------------------------------------------------------------------
# ៥. តារាងសង្ខេប៖ ជ្រើសរើស loss ត្រឹមត្រូវ
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("តារាងសង្ខេប")
print("=" * 68)
print(f"{'បញ្ហា':<26}{'loss function':<28}{'target dtype'}")
print("-" * 68)
print(f"{'multi-class (1 ថ្នាក់)':<22}{'nn.CrossEntropyLoss':<28}int64 (class index)")
print(f"{'binary':<26}{'nn.BCEWithLogitsLoss':<28}float32 (0.0/1.0)")
print(f"{'multi-label':<26}{'nn.BCEWithLogitsLoss':<28}float32 (0.0/1.0)")
print(f"{'regression':<26}{'nn.MSELoss / SmoothL1Loss':<28}float32")
print(f"{'count / ranking':<26}{'nn.PoissonNLLLoss':<28}float32")

print("\nOK — loss functions ទាំងអស់ដំណើរការត្រឹមត្រូវ")
