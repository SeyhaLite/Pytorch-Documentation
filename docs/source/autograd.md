# Gradient និង Autograd

**Autograd** គឺជាបេះដូងរបស់ PyTorch។ វាជាប្រព័ន្ធ **automatic differentiation** ដែល
គណនា derivative នៃអនុគមន៍ណាមួយដោយស្វ័យប្រវត្តិ។ បើអ្នកយល់ពី autograd អ្នកយល់
ពីរបៀបដែល neural networks រៀន។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ៣ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## Gradient គឺជាអ្វី?

**Gradient** គឺជា **derivative ភាគយក** (partial derivative) នៃអនុគមន៍មួយធៀបនឹង
អថេរមួយ។ វាឆ្លើយសំណួរ៖

> "បើខ្ញុំប្រែប្រួលអថេរនេះបន្តិច តើលទ្ធផលនឹងប្រែប្រួលប៉ុន្មាន ហើយក្នុងទិសដៅណា?"

ឧទាហរណ៍សាមញ្ញបំផុត៖ បើ `y = x²` នោះ `dy/dx = 2x`។ នៅ `x = 3` gradient = 6
មានន័យថា បើអ្នកបង្កើន `x` ដោយ 0.001 នោះ `y` នឹងកើនឡើងប្រហែល 0.006។

ក្នុង neural network យើងចង់បាន gradient នៃ **loss** ធៀបនឹង **weight រាប់លាន**។
Autograd គណនាវាទាំងអស់ក្នុងការដើរថយក្រោយតែមួយ។

## Computational Graph

ពេលអ្នកធ្វើប្រមាណវិធីលើ tensor ដែលមាន `requires_grad=True` PyTorch មិនគ្រាន់តែ
គណនាលទ្ធផលទេ — វាក៏ **កត់ត្រាប្រមាណវិធីនោះ** ទុកក្នុងក្រាហ្វផងដែរ។

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2                    # កត់ត្រា: y = PowBackward(x)
z = y + 1                     # កត់ត្រា: z = AddBackward(y)

print(z.grad_fn)              # <AddBackward0 object>
print(z.grad_fn.next_functions)   # ទៅកាន់ PowBackward0 -> AccumulateGrad(x)
```

`z.grad_fn` គឺជា "រូបមន្តត្រឡប់" ដែល PyTorch បានរក្សាទុក។ នៅពេលអ្នកហៅ
`z.backward()` វាដើរថយក្រោយតាមក្រាហ្វនេះ ដោយអនុវត្ត chain rule នៅគ្រប់ជំហាន។

```{admonition} PyTorch ប្រើ dynamic graph (define-by-run)
:class: note
ក្រាហ្វត្រូវបានសាងសង់ **ឡើងវិញរាល់ forward pass**។ នេះមានន័យថាអ្នកអាចប្រើ
`if`, `for`, `while` ធម្មតាក្នុង `forward()` — ក្រាហ្វនឹងសម្របខ្លួនទៅតាម។

នេះគឺជាភាពខុសគ្នាធំបំផុតពី TensorFlow 1.x (static graph) និងជាមូលហេតុដែល
PyTorch ងាយស្រួល debug។
```

## ១. មូលដ្ឋាន៖ requires_grad, backward(), .grad

```{literalinclude} examples/autograd_1.py
:language: python
:caption: autograd_1.py — ✅ បានសាកល្បងលើ CPU
```

### ច្បាប់សំខាន់ៗ

```{list-table}
:header-rows: 1
:widths: 30 70

* - ច្បាប់
  - ការពន្យល់
* - `backward()` ទាមទារ scalar
  - បើ output មិនមែនជា scalar អ្នកត្រូវបញ្ជាក់ `gradient=` ឬបូកវាជាមុន។ នេះជាមូលហេតុដែល loss function ទាំងអស់ត្រឡប់ scalar។
* - `.grad` មានតែលើ **leaf tensor**
  - Leaf = tensor ដែលអ្នកបង្កើតដោយផ្ទាល់ (មិនមែនលទ្ធផលនៃប្រមាណវិធី)។ Parameter របស់ model គឺជា leaf។
* - `.grad` **ប្រមូលផ្ដុំ** (accumulate)
  - `backward()` **បូកបន្ថែម** ទៅ `.grad` មិនមែនជំនួសទេ។ នេះជាមូលហេតុដែលត្រូវ `zero_grad()`។
* - ក្រាហ្វត្រូវបានលុបបន្ទាប់ពី `backward()`
  - ដើម្បីសន្សំ memory។ បើត្រូវការ backward ពីរដង ប្រើ `retain_graph=True`។
```

### ហេតុអ្វី backward() ទាមទារ scalar?

Derivative នៃ vector ធៀបនឹង vector គឺជា **matrix** (Jacobian) មិនមែន vector ទេ។
ដើម្បីជៀសវាងការផ្ទុក matrix ដ៏ធំ PyTorch ទាមទារឱ្យអ្នកបញ្ជាក់ **ទិសដៅ**៖

```python
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v * 2                     # shape (3,)

out.backward()                  # ✗ RuntimeError
out.sum().backward()            # ✓ បូកទៅជា scalar ជាមុន
out.backward(gradient=torch.tensor([1.0, 0.0, 0.0]))   # ✓ បញ្ជាក់ទិសដៅ
```

ក្នុងការអនុវត្តជាក់ស្ដែង loss function តែងតែបូក/មធ្យមភាគទៅជា scalar រួចហើយ
ដូច្នេះអ្នកមិនជួបបញ្ហានេះក្នុង training loop ធម្មតាទេ។

## ២. Backpropagation លើ Neural Network

```{literalinclude} examples/autograd_2.py
:language: python
:caption: autograd_2.py — ✅ បានសាកល្បងលើ CPU (រួមទាំងការផ្ទៀងផ្ទាត់ gradient ដោយ finite differences)
```

### Chain rule ដោយសង្ខេប

សម្រាប់អនុគមន៍បំពាក់បន្ទុក `L = f(g(h(x)))`៖

$$\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

Backpropagation គ្រាន់តែជា **ការអនុវត្ត chain rule ពីក្រោយទៅមុខ** ដោយរក្សាទុក
derivative កណ្ដាលដើម្បីកុំគណនាម្ដងទៀត។ នេះជាមូលហេតុដែលវាមានប្រសិទ្ធភាព៖
ការដើរម្ដងគត់ផ្ដល់ gradient សម្រាប់ parameter **រាប់លាន**។

### ការផ្ទៀងផ្ទាត់ gradient ដោយ finite differences

ឧទាហរណ៍ខាងលើបង្ហាញវិធីពិនិត្យថា autograd ត្រឹមត្រូវ៖

```python
eps = 1e-4
param[i, j] = original + eps
loss_plus = loss_fn(model(x), target).item()

param[i, j] = original - eps
loss_minus = loss_fn(model(x), target).item()

numeric_grad = (loss_plus - loss_minus) / (2 * eps)
assert abs(numeric_grad - param.grad[i, j].item()) < 1e-3
```

PyTorch មានឧបករណ៍សម្រាប់ធ្វើវាដោយស្វ័យប្រវត្តិ៖

```python
# ⚠️ gradcheck ទាមទារ float64 ដើម្បីឱ្យត្រឹមត្រូវ
point = torch.randn(5, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(f, (point,), eps=1e-6, atol=1e-4)   # True
```

```{admonition} gradcheck ត្រូវការ float64
:class: important
`torch.autograd.gradcheck()` ប្រៀបធៀបលេខដែលខុសគ្នាប្រហែល `1e-6`។
float32 មានភាពជាក់លាក់តែ `1e-7` ដូច្នេះវានឹងបរាជ័យដោយកំហុស rounding។
**ប្ដូរ input និង model ទៅ `torch.float64`** មុនហៅ `gradcheck`។
```

## ៣. detach, no_grad និង inference_mode

បីវិធីដើម្បីបញ្ឈប់ការកត់ត្រា gradient៖

```{list-table}
:header-rows: 1
:widths: 26 20 54

* - វិធី
  - វិសាលភាព
  - ប្រើពេល
* - `t.detach()`
  - tensor តែមួយ
  - កាត់ tensor មួយចេញពី graph ដើម្បីប្រើក្នុងកូដធម្មតា។ **នៅចែក memory!**
* - `with torch.no_grad():`
  - block
  - validation loop, ការវាយតម្លៃ, ការធ្វើបច្ចុប្បន្នភាពដោយដៃ
* - `with torch.inference_mode():`
  - block
  - inference សុទ្ធ។ លឿនជាង `no_grad` ប៉ុន្តែ tensor លទ្ធផលមិនអាចប្រើក្នុង autograd បានទេ
```

### ភាពខុសគ្នារវាង no_grad និង inference_mode

```python
with torch.no_grad():
    out = model(x)
out.requires_grad_(True)      # ✓ អនុញ្ញាត — អាចប្រើបន្តក្នុង graph ថ្មី

with torch.inference_mode():
    out = model(x)
out.requires_grad_(True)      # ✗ RuntimeError
```

`inference_mode` លឿនជាងព្រោះវាមិនកត់ត្រា **version counter** របស់ tensor ទេ
(យន្តការដែល autograd ប្រើដើម្បីរក in-place mutation)។ ប្រើវាសម្រាប់ production
inference ប៉ុន្តែប្រើ `no_grad` សម្រាប់ validation loop ដែលអ្នកអាចចង់ប្រើ tensor បន្ត។

មើល {doc}`evaluation-and-inference` សម្រាប់ការប្រៀបធៀបលម្អិត និងលេខវាស់ពិត។

## ៤. Leaf tensor និង in-place operations

**Leaf tensor** = tensor ដែលអ្នកបង្កើតដោយផ្ទាល់ (មិនមែនលទ្ធផលនៃប្រមាណវិធី)។
Parameter របស់ `nn.Module` គឺជា leaf ជានិច្ច។

````{admonition} អ្នកមិនអាចកែ leaf tensor ដោយ in-place បានទេ
:class: danger
```python
w = torch.zeros(3, requires_grad=True)
w += 1        # RuntimeError!
```

សារកំហុស៖

```text
RuntimeError: a leaf Variable that requires grad is being used in an in-place
operation
```

ហេតុអ្វី? ព្រោះ `.grad` ត្រូវបានភ្ជាប់ទៅ leaf នោះ។ ការកែវាដោយផ្ទាល់នឹងធ្វើឱ្យ
ទំនាក់ទំនងនោះខូច។ **Optimizer ធ្វើការកែនេះក្នុង `torch.no_grad()`** ដូច្នេះវាដំណើរការ៖

```python
with torch.no_grad():
    w += 1                    # ✓ អនុញ្ញាត
```

នេះគឺជាអ្វីដែល `optimizer.step()` ធ្វើខាងក្នុង។
````

### In-place លើ tensor មិនមែន leaf

សូម្បីតែលើ tensor កណ្ដាល in-place អាចខូច graph បើ tensor នោះត្រូវការសម្រាប់ backward៖

```python
w = torch.tensor([2.0], requires_grad=True)
z = w * 3          # z ត្រូវការតម្លៃដើមរបស់ w សម្រាប់ backward
w.mul_(2)          # ⚠️ កែ w នៅនឹងកន្លែង
z.backward()       # RuntimeError!
```

```text
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation
```

**វិធាន៖ ក្នុង forward pass សរសេរ `x = x + 1` ជំនួស `x += 1`។**

## ៥. ការប្រើ torch.autograd.grad() ដោយផ្ទាល់

ជួនកាលអ្នកចង់បាន gradient ដោយមិនរក្សាទុកក្នុង `.grad`៖

```python
grads = torch.autograd.grad(
    outputs=loss,
    inputs=list(model.parameters()),
    retain_graph=True,        # រក្សា graph បើអ្នកនឹង backward ម្ដងទៀត
    create_graph=False,       # True សម្រាប់ meta-learning (gradient ទីពីរ)
    allow_unused=True,        # ត្រឡប់ None សម្រាប់ parameter ដែលមិនជាប់ graph
)
```

ប្រើសម្រាប់៖
- **MAML / meta-learning** — ត្រូវការ gradient នៃ gradient
- **Gradient penalty** (WGAN-GP)
- **Adversarial examples** — gradient ធៀបនឹង input មិនមែន weight
- **ការវិភាគ** ដោយមិនប៉ះ `.grad`

## ៦. Gradient Accumulation សម្រាប់ Batch ធំ

បើ GPU របស់អ្នកមិនអាចសម batch ធំបាន អ្នកអាច **ប្រមូល gradient** លើ batch តូចៗ៖

```python
optimizer.zero_grad()
accumulation_steps = 4

for i, (xb, yb) in enumerate(loader):
    # ⚠️ ចែក loss ដើម្បីឱ្យ gradient មានទំហំដូច batch ធំ
    loss = loss_fn(model(xb), yb) / accumulation_steps
    loss.backward()                        # បូកបន្ថែមទៅ .grad

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                   # ធ្វើបច្ចុប្បន្នភាពម្ដងក្នុង 4 batch
        optimizer.zero_grad()
```

បច្ចេកទេសនេះផ្ដល់ **effective batch size = batch_size × accumulation_steps**
ដោយប្រើ memory តែសម្រាប់ batch តូចប៉ុណ្ណោះ។

```{admonition} ត្រូវចែក loss ដោយ accumulation_steps
:class: important
បើអ្នកមិនចែកទេ gradient នឹងធំជាង ៤ ដង ដែលស្មើនឹងការបង្កើន learning rate ៤ ដង។
ការចែកធ្វើឱ្យវាដូចគ្នាគណិតវិទ្យាទៅនឹងការប្រើ batch ធំជាមួយ `reduction='mean'`។
```

## ៧. កំហុស Autograd ទូទៅ

```{literalinclude} examples/autograd_3.py
:language: python
:caption: autograd_3.py — ✅ បានសាកល្បងលើ CPU (បង្ហាញកំហុសទាំងអស់ដោយចេតនា)
```

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `grad can be implicitly created only for scalar outputs`
  - `backward()` លើ vector
  - `.sum().backward()` ឬ `backward(gradient=...)`
* - `Trying to backward through the graph a second time`
  - graph ត្រូវបានលុបរួច
  - `backward(retain_graph=True)`
* - `element 0 of tensors does not require grad`
  - ប្រភពមិនមាន `requires_grad`
  - ពិនិត្យ `model.parameters()` មិនទទេ និង tensor មាន `requires_grad=True`
* - `a leaf Variable that requires grad is being used in an in-place operation`
  - កែ leaf tensor
  - ធ្វើក្នុង `torch.no_grad()`
* - `does not require grad and does not have a grad_fn`
  - model ត្រូវបានបង្កើតក្នុង `no_grad()`
  - ផ្លាស់ទីការបង្កើត model ចេញពី `no_grad()` block
* - gradient ទាំងអស់សូន្យ
  - ភ្លេច `loss.backward()` ឬ model មិនជាប់ graph
  - ពិនិត្យ `p.grad is None` បន្ទាប់ពី backward
* - gradient កើនឡើងជានិច្ច
  - ភ្លេច `optimizer.zero_grad()`
  - បន្ថែមវានៅដើម iteration
* - memory កើនឡើងជានិច្ចក្នុង training loop
  - រក្សាទុក loss tensor ជំនួស `.item()`
  - ប្រើ `loss.item()` ឬ `loss.detach()`
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **`requires_grad=True`** ធ្វើឱ្យ PyTorch កត់ត្រាប្រមាណវិធីលើ tensor នោះ។
2. **`backward()`** ដើរថយក្រោយតាម graph ហើយ **បូកបន្ថែម** ទៅ `.grad`។
3. **`zero_grad()`** មុនរាល់ `backward()` — បើមិនដូច្នោះទេ gradient ប្រមូលផ្ដុំ។
4. **ក្រាហ្វត្រូវបានលុប** បន្ទាប់ពី backward ម្ដង។ ប្រើ `retain_graph=True` បើត្រូវការម្ដងទៀត។
5. **`detach()` ចែក memory** — `detach().clone()` បើអ្នកចង់បានច្បាប់ចម្លងពិត។
6. **In-place operations អាចខូច graph** — សរសេរ `x = x + 1` ក្នុង forward pass។
7. **Gradient មានតែលើ leaf tensor** — parameter របស់ model គឺជា leaf។
```

## មេរៀនបន្ទាប់

{doc}`neural-networks` — របៀបសាងសង់ម៉ូដែលដោយប្រើ `torch.nn`។
