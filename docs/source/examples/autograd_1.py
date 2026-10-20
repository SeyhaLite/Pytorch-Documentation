"""Autograd មូលដ្ឋាន៖ requires_grad, backward(), .grad។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

Autograd = ប្រព័ន្ធគណនា derivative ដោយស្វ័យប្រវត្តិ។
PyTorch កត់ត្រារាល់ប្រមាណវិធីដែលធ្វើលើ tensor ដែលមាន requires_grad=True
ហើយបង្កើតជា "computational graph" ដែលអាចដើរថយក្រោយបាន។
"""

import torch

# ---------------------------------------------------------------------------
# ១. ឧទាហរណ៍សាមញ្ញបំផុត៖ y = x^2  ដូច្នេះ dy/dx = 2x
# ---------------------------------------------------------------------------
x = torch.tensor(3.0, requires_grad=True)
print("x =", x)
print("x.requires_grad =", x.requires_grad)
print("x.grad =", x.grad, "(នៅទទេ — មិនទាន់ backward)")

y = x ** 2
y.backward()              # គណនា dy/dx ហើយរក្សាទុកក្នុង x.grad

print("-" * 55)
print("y = x**2 =", y.item())
print("y.backward() រួច -> x.grad =", x.grad.item())
print("រំពឹង 2*x = 2*3 =", 2 * 3.0)
assert x.grad.item() == 6.0

# ---------------------------------------------------------------------------
# ២. ច្បាប់សំខាន់៖ backward() ត្រូវការ scalar
# ---------------------------------------------------------------------------
print("-" * 55)
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v * 2                       # shape (3,) — មិនមែន scalar

try:
    out.backward()
except RuntimeError as e:
    print("out.backward() លើ vector -> RuntimeError:")
    print(" ", str(e).splitlines()[0][:88])

# ដំណោះស្រាយ ១៖ បូកឱ្យទៅជា scalar ជាមុន
v.grad = None
out.sum().backward()
print("\nដំណោះស្រាយ ១: out.sum().backward() -> v.grad =", v.grad.tolist())
assert v.grad.tolist() == [2.0, 2.0, 2.0]

# ដំណោះស្រាយ ២៖ ផ្ដល់ gradient ដើម (មិនចាំបាច់ sum)
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
(w * 2).backward(gradient=torch.tensor([1.0, 0.0, 0.0]))
print("ដំណោះស្រាយ ២: backward(gradient=[1,0,0]) -> w.grad =", w.grad.tolist())
assert w.grad.tolist() == [2.0, 0.0, 0.0]

# ---------------------------------------------------------------------------
# ៣. gradient ប្រមូលផ្ដុំ (accumulate) — ហេតុអ្វីត្រូវ zero_grad()
# ---------------------------------------------------------------------------
print("-" * 55)
z = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    loss = z * 10
    loss.backward()               # បន្ថែមទៅ .grad ជានិច្ច មិនជំនួសទេ!
    print(f"  iteration {i + 1}: z.grad = {z.grad.item()}")

assert z.grad.item() == 30.0, z.grad.item()
print("=> gradient ត្រូវបានបូកបញ្ចូលគ្នា៖ d(z*10)/dz = 10 ហើយ backward() 3 ដង = 30")
print("   នេះហើយជាមូលហេតុដែល training loop ត្រូវហៅ optimizer.zero_grad()")
print("   នៅដើមរាល់ iteration — បើមិនដូច្នោះទេ gradient នឹងកើនឡើងឥតឈប់ឈរ។")

# សម្អាត gradient
z.grad = None                    # វិធីទី ១
print("\nបន្ទាប់ z.grad = None ->", z.grad)

z2 = torch.tensor(2.0, requires_grad=True)
(z2 * 10).backward()
print("មុន zero_grad  ->", z2.grad.item())
z2.grad.zero_()                  # វិធីទី ២ (in-place)
print("បន្ទាប់ grad.zero_() ->", z2.grad.item())
assert z2.grad.item() == 0.0

# ---------------------------------------------------------------------------
# ៤. detach() និង torch.no_grad()
# ---------------------------------------------------------------------------
print("-" * 55)
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a * 3                 # b ជាប់នឹង graph
c = b.detach()            # c ដាច់ចេញពី graph ប៉ុន្តែចែក memory

print("b.requires_grad =", b.requires_grad)
print("c.requires_grad =", c.requires_grad, "(detach រួច)")
print("b.grad_fn       =", b.grad_fn)
print("c.grad_fn       =", c.grad_fn)
assert b.requires_grad and not c.requires_grad

# ព្រមាន៖ c ចែក memory ជាមួយ b ដូច្នេះកែ c គឺកែ b
c[0] = 999.0
print("កែ c[0]=999 -> b =", b.tolist(), "(ចែក memory រួមគ្នា)")
assert b[0].item() == 999.0

with torch.no_grad():
    d = a * 5
print("\nក្នុង no_grad(): d.requires_grad =", d.requires_grad, "| d.grad_fn =", d.grad_fn)
assert not d.requires_grad

print("-" * 55)
print("OK — autograd មូលដ្ឋានដំណើរការត្រឹមត្រូវ")
