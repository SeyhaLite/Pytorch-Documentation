"""គណនា metrics៖ accuracy, precision, recall, F1 និង confusion matrix។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
មិនពឹងលើ torchmetrics — គណនាដោយប្រើ PyTorch សុទ្ធ។
"""

import torch

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. ទិន្នន័យគំរូ
# ---------------------------------------------------------------------------
n, c = 200, 3
logits = torch.randn(n, c)
y_true = torch.randint(0, c, (n,))
y_pred = logits.argmax(dim=1)

print("y_true shape:", tuple(y_true.shape), y_true.dtype)
print("y_pred shape:", tuple(y_pred.shape), y_pred.dtype)

# ---------------------------------------------------------------------------
# ២. Accuracy
# ---------------------------------------------------------------------------
acc = (y_pred == y_true).float().mean().item()
print(f"\naccuracy = {acc:.4f}")
assert 0.0 <= acc <= 1.0

# ---------------------------------------------------------------------------
# ៣. Confusion matrix (ជួរ = ការពិត, ឈរ = ការទាយ)
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """តារាង (num_classes, num_classes)។

    cm[i, j] = ចំនួនសំណាកដែលជាការពិតថ្នាក់ i ហើយត្រូវបានទាយថាថ្នាក់ j។
    """
    idx = y_true * num_classes + y_pred
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


cm = confusion_matrix(y_true, y_pred, c)
print("\nconfusion matrix (ជួរ=ពិត, ឈរ=ទាយ):")
print(cm)
assert cm.sum().item() == n, "ផលបូកធាតុទាំងអស់ត្រូវតែស្មើចំនួនសំណាក"

# ---------------------------------------------------------------------------
# ៤. Precision, Recall, F1 តាមថ្នាក់
# ---------------------------------------------------------------------------
def prf1(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """គណនា precision/recall/F1 ពី confusion matrix។"""
    tp = cm.diagonal().float()                 # ការទាយត្រូវក្នុងថ្នាក់នីមួយៗ
    pred_pos = cm.sum(dim=0).float()           # សរុបតាមឈរ = ទាយថាជាថ្នាក់ j
    true_pos = cm.sum(dim=1).float()           # សរុបតាមជួរ = ការពិតជាថ្នាក់ i

    precision = tp / pred_pos.clamp(min=1)
    recall = tp / true_pos.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return precision, recall, f1


precision, recall, f1 = prf1(cm)
print(f"\n{'ថ្នាក់':<8}{'precision':>11}{'recall':>9}{'f1':>9}{'គាំទ្រ':>9}")
for i in range(c):
    print(f"{i:<8}{precision[i].item():>11.4f}{recall[i].item():>9.4f}"
          f"{f1[i].item():>9.4f}{cm[i].sum().item():>9}")

# macro average = មធ្យមភាគសាមញ្ញតាមថ្នាក់
macro_p, macro_r, macro_f1 = precision.mean().item(), recall.mean().item(), f1.mean().item()
print(f"\nmacro avg: precision={macro_p:.4f} recall={macro_r:.4f} f1={macro_f1:.4f}")

# weighted average = មធ្យមភាគថ្លឹងតាមចំនួនសំណាកក្នុងថ្នាក់
support = cm.sum(dim=1).float()
w_p = (precision * support).sum().item() / support.sum().item()
w_r = (recall * support).sum().item() / support.sum().item()
w_f1 = (f1 * support).sum().item() / support.sum().item()
print(f"weighted : precision={w_p:.4f} recall={w_r:.4f} f1={w_f1:.4f}")

assert 0.0 <= macro_f1 <= 1.0
assert abs(w_r - acc) < 1e-5, "weighted recall ត្រូវតែស្មើ accuracy"

# ---------------------------------------------------------------------------
# ៥. Binary metrics ដោយផ្ទាល់
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
y_bin = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
p_bin = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

tp = ((y_bin == 1) & (p_bin == 1)).sum().item()
tn = ((y_bin == 0) & (p_bin == 0)).sum().item()
fp = ((y_bin == 0) & (p_bin == 1)).sum().item()
fn = ((y_bin == 1) & (p_bin == 0)).sum().item()

print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"accuracy  = {(tp + tn) / len(y_bin):.4f}")
print(f"precision = {tp / max(tp + fp, 1):.4f}   (ក្នុងអ្វីដែលទាយ positive តើប៉ុន្មានត្រូវ?)")
print(f"recall    = {tp / max(tp + fn, 1):.4f}   (ក្នុង positive ពិត តើរកឃើញប៉ុន្មាន?)")
print(f"specificity = {tn / max(tn + fp, 1):.4f}")
assert tp + tn + fp + fn == len(y_bin)

# ---------------------------------------------------------------------------
# ៦. Top-k accuracy (ប្រើសម្រាប់ image classification)
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
topk = torch.topk(logits, k=2, dim=1).indices      # (n, 2)
top1 = (topk[:, 0] == y_true).float().mean().item()
top2 = (topk == y_true.unsqueeze(1)).any(dim=1).float().mean().item()
print(f"top-1 accuracy = {top1:.4f}")
print(f"top-2 accuracy = {top2:.4f}")
assert top2 >= top1

# ---------------------------------------------------------------------------
# ៧. Regression metrics
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
pred_r = torch.randn(100)
true_r = pred_r + 0.3 * torch.randn(100)          # ទាយជិតត្រូវ

mse = torch.nn.functional.mse_loss(pred_r, true_r).item()
mae = torch.nn.functional.l1_loss(pred_r, true_r).item()
ss_res = ((true_r - pred_r) ** 2).sum().item()
ss_tot = ((true_r - true_r.mean()) ** 2).sum().item()
r2 = 1.0 - ss_res / ss_tot

print(f"MSE = {mse:.4f}")
print(f"RMSE = {mse ** 0.5:.4f}")
print(f"MAE = {mae:.4f}")
print(f"R^2 = {r2:.4f}   (1.0 = ល្អឥតខ្ចោះ, 0 = ដូចទាយមធ្យមភាគ)")
assert r2 > 0.5

# ---------------------------------------------------------------------------
# ៨. វាយតម្លៃដោយមិនរំលង batch (streaming) — សម្រាប់ទិន្នន័យធំ
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
total_cm = torch.zeros(c, c, dtype=torch.long)
n_batches = 4
for b in range(n_batches):
    batch_logits = torch.randn(50, c)
    batch_true = torch.randint(0, c, (50,))
    batch_pred = batch_logits.argmax(1)
    total_cm += confusion_matrix(batch_true, batch_pred, c)

print(f"confusion matrix បូកពី {n_batches} batches:")
print(total_cm)
print(f"សំណាកសរុប = {total_cm.sum().item()}")
assert total_cm.sum().item() == 50 * n_batches
stream_acc = total_cm.diagonal().sum().item() / total_cm.sum().item()
print(f"streaming accuracy = {stream_acc:.4f}")

print("\nOK — metrics ទាំងអស់ដំណើរការត្រឹមត្រូវ")
