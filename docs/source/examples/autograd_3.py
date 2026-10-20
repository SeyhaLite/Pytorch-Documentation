"""កំហុស Autograd ដែលជួបញឹកញាប់ និងវិធីដោះស្រាយ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
ឯកសារនេះបង្ហាញកំហុសដោយចេតនា — កំហុសនីមួយៗត្រូវបានចាប់យកដោយ try/except។
"""

import torch
import torch.nn as nn

print("=" * 62)
print("កំហុសទី ១: backward() លើ tensor ដែលមិនមែន scalar")
print("=" * 62)
try:
    v = torch.randn(3, requires_grad=True)
    (v * 2).backward()
except RuntimeError as e:
    print("RuntimeError:", str(e).splitlines()[0][:95])
    print("ដំណោះស្រាយ: ប្រើ .sum().backward() ឬផ្ដល់ gradient= ដោយផ្ទាល់")

print()
print("=" * 62)
print("កំហុសទី ២: backward() ពីរដងដោយគ្មាន retain_graph=True")
print("=" * 62)
try:
    a = torch.tensor(2.0, requires_grad=True)
    y = a ** 2
    y.backward()
    y.backward()          # graph ត្រូវបានលុបរួចហើយ
except RuntimeError as e:
    print("RuntimeError:", str(e).splitlines()[0][:95])
    print("ដំណោះស្រាយ: y.backward(retain_graph=True) បើត្រូវការ backward ច្រើនដង")

print()
print("=" * 62)
print("កំហុសទី ៣: ភ្លេច optimizer.zero_grad()")
print("=" * 62)
p = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([p], lr=0.1)

for step in range(3):
    # ភ្លេច opt.zero_grad() នៅទីនេះ!
    loss = (p - 5.0) ** 2
    loss.backward()
    print(f"  step {step + 1}: p.grad = {p.grad.item():+.4f}  <- កើនឡើងជានិច្ច មិនត្រឹមត្រូវ")
    opt.step()

print("ដំណោះស្រាយ: ហៅ opt.zero_grad() មុន loss.backward() ជានិច្ច")

# បង្ហាញកំណែត្រឹមត្រូវ
print("\nកំណែត្រឹមត្រូវ:")
p2 = torch.tensor(1.0, requires_grad=True)
opt2 = torch.optim.SGD([p2], lr=0.1)
for step in range(3):
    opt2.zero_grad()               # <- នេះជាអ្វីដែលខ្វះ
    loss = (p2 - 5.0) ** 2
    loss.backward()
    print(f"  step {step + 1}: p.grad = {p2.grad.item():+.4f}   p = {p2.item():.4f}")
    opt2.step()

print()
print("=" * 62)
print("កំហុសទី ៤: កែ tensor ដែលមាន requires_grad នៅខាងក្រៅ no_grad()")
print("=" * 62)
try:
    m = torch.tensor([1.0, 2.0], requires_grad=True)
    m.data = m.data * 2            # .data រំលងការត្រួតពិនិត្យ — គ្រោះថ្នាក់!
    m += 1                         # in-place លើ leaf tensor
except RuntimeError as e:
    print("RuntimeError:", str(e).splitlines()[0][:95])
    print("ដំណោះស្រាយ: ធ្វើ in-place update ក្នុង torch.no_grad(): ឬប្រើ optimizer")

with torch.no_grad():
    m2 = torch.tensor([1.0, 2.0], requires_grad=True)
    m2 += 1
print("  ក្នុង no_grad(): m2 += 1 ដំណើរការ ->", m2.tolist())

print()
print("=" * 62)
print("កំហុសទី ៥: parameter ថ្មីមិនត្រូវបានចុះបញ្ជីក្នុង optimizer")
print("=" * 62)
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# បន្ថែម parameter ថ្មីបន្ទាប់ពីបង្កើត optimizer
extra = nn.Parameter(torch.zeros(1))
try:
    y = model(torch.randn(4, 3)).sum() + extra.sum()
    y.backward()
    optimizer.step()               # extra មិនត្រូវបានធ្វើបច្ចុប្បន្នភាពទេ!
    print("extra.grad =", extra.grad.item(), "(មាន gradient...)")
    print("extra value =", extra.item(), "(ប៉ុន្តែមិនបានផ្លាស់ប្ដូរ — optimizer មិនស្គាល់វា)")
    assert extra.item() == 0.0
    print("ដំណោះស្រាយ: optimizer.add_param_group({'params': extra})")
    optimizer.add_param_group({"params": extra})
    optimizer.zero_grad()
    y = model(torch.randn(4, 3)).sum() + extra.sum()
    y.backward()
    optimizer.step()
    print("បន្ទាប់ add_param_group: extra =", round(extra.item(), 6), "(បានផ្លាស់ប្ដូរ ✓)")
    assert extra.item() != 0.0
except AssertionError:
    raise

print()
print("=" * 62)
print("កំហុសទី ៦: ប្រើ .item() / .detach() ខុសពេលវេលា ក្នុង training loop")
print("=" * 62)
model2 = nn.Linear(2, 1)
x = torch.randn(8, 2)
t = torch.randn(8, 1)

# ខុស៖ រក្សាទុក tensor ទាំងមូល -> memory leak (graph មិនត្រូវបានដោះលែង)
history_bad = []
history_good = []
for step in range(3):
    loss = nn.functional.mse_loss(model2(x), t)
    history_bad.append(loss)              # ភ្ជាប់ graph ទាំងមូលទុក!
    history_good.append(loss.item())      # យកតែលេខ Python
    loss.backward()

print("history_bad  :", [type(v).__name__ for v in history_bad], "<- tensor ជាប់ graph")
print("history_good :", [round(v, 4) for v in history_good], "<- float ធម្មតា")
print("ដំណោះស្រាយ: ប្រើ loss.item() ឬ loss.detach() មុនរក្សាទុក")

print()
print("=" * 62)
print("OK — កំហុសទាំងអស់ត្រូវបានបង្ហាញ និងពន្យល់ដំណោះស្រាយ")
