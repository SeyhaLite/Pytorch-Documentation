"""កំហុសធម្មតាក្នុង training loop និងវិធីដោះស្រាយ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(64, 8)
y = torch.randint(0, 3, (64,))


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3))


print("=" * 68)
print("កំហុសទី ១: ភ្លេច model.eval() ពេលវាយតម្លៃ")
print("=" * 68)
m = make_model()
m.train()
with torch.no_grad():
    acc_train_mode = (m(x).argmax(1) == y).float().mean().item()
m.eval()
with torch.no_grad():
    p1 = m(x).argmax(1)
    p2 = m(x).argmax(1)
    acc_eval_mode = (p1 == y).float().mean().item()

print(f"  model.train() ពេលវាយតម្លៃ -> accuracy = {acc_train_mode:.4f}")
print(f"  model.eval()               -> accuracy = {acc_eval_mode:.4f}")
print(f"  eval() លទ្ធផលដូចគ្នារាល់ដង? {torch.equal(p1, p2)}")
print("  => Dropout កំពុងច្របល់លទ្ធផល បើភ្លេច eval()")

print()
print("=" * 68)
print("កំហុសទី ២: គណនា validation loss ក្រោម torch.no_grad() ដែលបាត់")
print("=" * 68)
m2 = make_model()
m2.eval()
loss_fn = nn.CrossEntropyLoss()

# ខុស៖ គ្មាន no_grad -> graph ត្រូវបានរក្សាទុក -> memory កើនឡើង
torch.manual_seed(1)
losses_bad = []
for i in range(5):
    losses_bad.append(loss_fn(m2(x), y))
mem_bad = sum(l.untyped_storage().size() for l in losses_bad)

# ត្រូវ៖ មាន no_grad
torch.manual_seed(1)
losses_good = []
for i in range(5):
    with torch.no_grad():
        losses_good.append(loss_fn(m2(x), y))

print(f"  គ្មាន no_grad: loss ជាប់ grad_fn? {losses_bad[0].grad_fn is not None}")
print(f"  មាន no_grad : loss ជាប់ grad_fn? {losses_good[0].grad_fn is not None}")
print(f"  តម្លៃដូចគ្នា? {torch.allclose(losses_bad[0], losses_good[0])}")
assert losses_bad[0].grad_fn is not None
assert losses_good[0].grad_fn is None
print("  => លទ្ធផលដូចគ្នា ប៉ុន្តែគ្មាន no_grad រក្សា graph ទុក (memory leak)")

print()
print("=" * 68)
print("កំហុសទី ៣: ភ្លេច optimizer.zero_grad()")
print("=" * 68)
m3 = make_model()
opt = torch.optim.SGD(m3.parameters(), lr=0.01)

norms_forget, norms_correct = [], []
for step in range(4):
    # ភ្លេច zero_grad
    loss = loss_fn(m3(x), y)
    loss.backward()
    n = torch.nn.utils.clip_grad_norm_(m3.parameters(), float("inf")).item()
    norms_forget.append(round(n, 4))
    opt.step()
print("  ភ្លេច zero_grad -> gradient norms:", norms_forget, "(កើនឡើង)")

m4 = make_model()
opt2 = torch.optim.SGD(m4.parameters(), lr=0.01)
for step in range(4):
    opt2.zero_grad()                 # <- ត្រឹមត្រូវ
    loss = loss_fn(m4(x), y)
    loss.backward()
    n = torch.nn.utils.clip_grad_norm_(m4.parameters(), float("inf")).item()
    norms_correct.append(round(n, 4))
    opt2.step()
print("  មាន zero_grad -> gradient norms:", norms_correct, "(ថេរជាង)")
assert norms_forget[-1] > norms_correct[-1]

print()
print("=" * 68)
print("កំហុសទី ៤: រក្សាទុក tensor loss ជំនួស .item()")
print("=" * 68)
m5 = make_model()
opt3 = torch.optim.SGD(m5.parameters(), lr=0.01)
history_bad, history_good = [], []
for step in range(3):
    opt3.zero_grad()
    loss = loss_fn(m5(x), y)
    history_bad.append(loss)          # ជាប់ graph
    history_good.append(loss.item())  # float ធម្មតា
    loss.backward()
    opt3.step()

print("  history_bad  types:", {type(v).__name__ for v in history_bad})
print("  history_good types:", {type(v).__name__ for v in history_good}, history_good)
assert all(isinstance(v, float) for v in history_good)
assert all(hasattr(v, "grad_fn") for v in history_bad)
print("  => ប្រើ loss.item() ឬ loss.detach() ជានិច្ចសម្រាប់ logging")

print()
print("=" * 68)
print("កំហុសទី ៥: loss មិនថយចុះ — checklist ធ្វើរោគវិនិច្ឆ័យ")
print("=" * 68)
checklist = [
    ("learning rate ធំពេក/តូចពេក?", "សាកល្បង 10x តូចជាង និង 10x ធំជាង"),
    ("ភ្លេច zero_grad()?", "gradient ប្រមូលផ្ដុំ ធ្វើឱ្យ update ខុសទិស"),
    ("ភ្លេច loss.backward()?", "parameter មិនដែលទទួល gradient"),
    ("loss function ខុស?", "CrossEntropyLoss ត្រូវការ logits មិនមែន softmax"),
    ("target dtype ខុស?", "class index ត្រូវតែជា int64"),
    ("shape មិនត្រូវ?", "logits (N, C) និង target (N,)"),
    ("ទិន្នន័យខូច/label ច្របល់?", "ពិនិត្យសំណាកខ្លះដោយភ្នែក"),
    ("gradient = 0 ឬ NaN?", "ពិនិត្យដោយ clip_grad_norm_ ឬ detect_anomaly"),
]
for i, (q, a) in enumerate(checklist, 1):
    print(f"  {i}. {q:<32} -> {a}")

# បង្ហាញ gradient norm ដើម្បីធ្វើរោគវិនិច្ឆ័យ
m6 = make_model()
opt4 = torch.optim.SGD(m6.parameters(), lr=0.01)
opt4.zero_grad()
loss = loss_fn(m6(x), y)
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(m6.parameters(), float("inf")).item()
n_zero = sum(1 for p in m6.parameters() if p.grad is None or p.grad.abs().max().item() == 0)
has_nan = any(torch.isnan(p.grad).any().item() for p in m6.parameters() if p.grad is not None)
print(f"\n  ពិនិត្យជាក់ស្ដែង: grad norm = {gnorm:.4f} | parameters ដែល grad=0: {n_zero} | មាន NaN: {has_nan}")
assert gnorm > 0 and not has_nan

print()
print("=" * 68)
print("OK — កំហុសទាំងអស់ត្រូវបានបង្ហាញ និងពន្យល់")
