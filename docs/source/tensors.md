# Tensor៖ ការបង្កើត និងប្រតិបត្តិការ

**Tensor** គឺជាទម្រង់ទិន្នន័យមូលដ្ឋានរបស់ PyTorch។ អ្វីៗទាំងអស់ — ទិន្នន័យចូល, weight,
gradient, output — សុទ្ធតែជា tensor។ មេរៀននេះគ្របដណ្ដប់អ្វីគ្រប់យ៉ាងដែលអ្នកត្រូវការ
ដើម្បីធ្វើការជាមួយ tensor ប្រចាំថ្ងៃ។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ៥ ក្នុងមេរៀននេះបានសាកល្បងលើ
PyTorch `2.14.1+cpu` / Python 3.11។
```

## Tensor គឺជាអ្វី?

Tensor គឺជា **អារេពហុវិមាត្រ** (n-dimensional array) នៃលេខ ដែលមានប្រភេទតែមួយ។
វាស្រដៀងនឹង NumPy array ប៉ុន្តែមានសមត្ថភាពបន្ថែមពីរ៖

```{list-table}
:header-rows: 1
:widths: 30 70

* - សមត្ថភាព
  - អត្ថន័យ
* - **Device placement**
  - Tensor អាចរស់នៅលើ CPU ឬ GPU (`cuda`)។ ប្ដូរដោយ `.to(device)`។
* - **Autograd tracking**
  - Tensor ដែលមាន `requires_grad=True` ត្រូវបាន PyTorch កត់ត្រាប្រមាណវិធី ដើម្បីគណនា gradient ដោយស្វ័យប្រវត្តិ។ មើល {doc}`autograd`។
```

### Tensor vs NumPy array

```{list-table}
:header-rows: 1
:widths: 34 33 33

* - លក្ខណៈ
  - `torch.Tensor`
  - `numpy.ndarray`
* - រត់លើ GPU
  - ✅ បាន
  - ❌ ទេ
* - Autograd
  - ✅ បាន
  - ❌ ទេ
* - ប្ដូររូបរាង
  - `.view()` / `.reshape()`
  - `.reshape()`
* - Matrix multiply
  - `@` / `torch.matmul`
  - `@` / `np.dot`
* - ប្ដូរទៅគ្នា
  - `t.numpy()`
  - `torch.from_numpy(a)`
```

````{admonition} torch.from_numpy() ចែក memory រួមគ្នា
:class: warning
```python
import numpy as np, torch
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
arr[0] = 999.0
print(t)          # tensor([999.,   2.,   3.])  <- ប្ដូរតាម!
```

បើអ្នកចង់ចម្លងដាច់ដោយឡែក ប្រើ `torch.tensor(arr)` ឬ `torch.from_numpy(arr).clone()`។
ផ្ទុយទៅវិញ `t.numpy()` ក៏ចែក memory ដែរ។
````

## ១. របៀបបង្កើត Tensor

មានវិធីបង្កើត tensor ជាច្រើន។ ខាងក្រោមគឺអ្វីដែលអ្នកត្រូវការ ៩៥% នៃពេលវេលា។

```{literalinclude} examples/tensors_1.py
:language: python
:caption: tensors_1.py — ✅ បានសាកល្បងលើ CPU
```

### តារាងយោងរហ័ស

```{list-table}
:header-rows: 1
:widths: 34 22 44

* - ពាក្យបញ្ជា
  - លទ្ធផល
  - ប្រើពេល
* - `torch.tensor([[1,2],[3,4]])`
  - ចម្លងទិន្នន័យ
  - អ្នកមានលេខរួចហើយ (list, tuple)
* - `torch.from_numpy(arr)`
  - ចែក memory
  - ប្ដូរពី NumPy ដោយមិនចម្លង
* - `torch.zeros(2, 3)`
  - សូន្យទាំងអស់
  - បំពេញ buffer មុនប្រើ
* - `torch.ones(2, 3)`
  - ១ ទាំងអស់
  - mask, baseline
* - `torch.empty(2, 3)`
  - **មិនបានកំណត់**
  - ត្រូវការ memory លឿន ហើយនឹងសរសេរជាន់ភ្លាមៗ
* - `torch.full((2,3), 7.0)`
  - តម្លៃថេរ
  - បំពេញដោយតម្លៃជាក់លាក់
* - `torch.eye(3)`
  - identity matrix
  - ពិជគណិតលីនេអ៊ែរ
* - `torch.arange(0, 10, 2)`
  - `[0,2,4,6,8]`
  - ដូច Python `range`
* - `torch.linspace(0, 1, 5)`
  - ៥ តម្លៃស្មើគ្នា
  - អ័ក្សពេលវេលា, grid
* - `torch.rand(2, 3)`
  - uniform `[0, 1)`
  - ទិន្នន័យសាកល្បង
* - `torch.randn(2, 3)`
  - normal `N(0, 1)`
  - ទិន្នន័យសាកល្បង, weight init
* - `torch.randint(0, 10, (2,3))`
  - ចំនួនគត់ `[0, 10)`
  - label សាកល្បង
* - `torch.zeros_like(t)`
  - ដូច `t` គ្រប់យ៉ាង
  - រក្សា shape/dtype/device ដោយស្វ័យប្រវត្តិ
```

```{admonition} torch.empty() មានតម្លៃមិនទៀងទាត់
:class: danger
`torch.empty(2, 3)` ផ្ដល់ memory ដែលមិនទាន់សម្អាត — តម្លៃក្នុងវាគឺជាអ្វីដែលនៅសល់
ពីអ្វីដែលបានប្រើ memory នោះមុន។ វាអាចជា `0`, លេខធំខ្លាំង, `nan` ឬទិន្នន័យពីកម្មវិធីផ្សេង។

**កុំពឹងលើតម្លៃរបស់វា** — ប្រើវាតែពេលអ្នកនឹងសរសេរជាន់លើគ្រប់ធាតុភ្លាមៗ
(ឧទាហរណ៍ជា output buffer)។
```

## ២. លក្ខណៈសម្បត្តិ Tensor និង Memory Layout

យល់ពី `shape`, `stride` និង `is_contiguous()` គឺជាគន្លឹះសម្រាប់ដោះស្រាយកំហុស
`view size is not compatible` ដែលអ្នកនឹងជួបមិនខាន។

```{literalinclude} examples/tensors_2.py
:language: python
:caption: tensors_2.py — ✅ បានសាកល្បងលើ CPU
```

### យល់ពី stride

Tensor មិនត្រូវបានរក្សាទុកជាអារេ ២D ក្នុង memory ទេ — វាជា **បន្ទាត់តែមួយ** (1D)។
`stride` ប្រាប់ PyTorch ពីរបៀបបកស្រាយ index ពហុវិមាត្រទៅជាទីតាំងក្នុងបន្ទាត់នោះ។

សម្រាប់ tensor shape `(2, 3, 4)`៖

```text
memory:  [ 0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15 | 16 17 18 19 | 20 21 22 23 ]
stride:   (12, 4, 1)

x[i, j, k]  ->  memory[i*12 + j*4 + k*1]
x[1, 2, 3]  ->  memory[1*12 + 2*4 + 3]  =  memory[23]  =  23
```

### View vs Clone

```{list-table}
:header-rows: 1
:widths: 24 36 40

* - ប្រតិបត្តិការ
  - Memory
  - ចំណាំ
* - `view()`, `reshape()`, `.T`, `transpose()`, `t[1:3]`, `squeeze()`, `unsqueeze()`, `expand()`, `permute()`
  - **ចែករួម** (view)
  - កែ view → កែ tensor ដើម
* - `clone()`
  - **ចម្លង**
  - ដាច់ដោយឡែកទាំងស្រុង ប៉ុន្តែរក្សា `requires_grad`
* - `detach()`
  - **ចែករួម**
  - ដាច់ចេញពី autograd graph ប៉ុន្តែនៅចែក memory
* - `detach().clone()`
  - **ចម្លង**
  - ទម្រង់សុវត្ថិភាពបំផុតសម្រាប់រក្សាទុកលទ្ធផល
```

````{admonition} ពេលណាត្រូវប្រើ contiguous()?
:class: tip
`transpose()`, `permute()` និង slicing មួយចំនួនបង្កើត tensor **មិន contiguous**
(stride មិនចុះលំដាប់)។ `view()` ទាមទារ contiguous tensor ដូច្នេះវានឹងបរាជ័យ៖

```text
RuntimeError: view size is not compatible with input tensor's size and stride
```

ដំណោះស្រាយពីរ៖

- `t.contiguous().view(...)` — ចម្លង memory ជាលំដាប់បន្ទាប់មក view
- `t.reshape(...)` — PyTorch ធ្វើវាដោយស្វ័យប្រវត្តិ

ប្រើ `reshape()` ជាលំនាំដើម លុះត្រាតែអ្នក **ចង់** ឱ្យវាបរាជ័យលើ tensor មិន contiguous។
````

## ៣. Indexing និង Slicing

```{literalinclude} examples/tensors_3.py
:language: python
:caption: tensors_3.py — ✅ បានសាកល្បងលើ CPU
```

### ទម្រង់ indexing ទាំងអស់

```{list-table}
:header-rows: 1
:widths: 30 22 48

* - ទម្រង់
  - ឧទាហរណ៍ shape
  - លទ្ធផល
* - `x[i]`
  - `(4,5)` → `(5,)`
  - ជួរទី `i` ទាំងមូល
* - `x[i, j]`
  - `(4,5)` → scalar
  - ធាតុតែមួយ (នៅតែជា tensor — ប្រើ `.item()`)
* - `x[:, j]`
  - `(4,5)` → `(4,)`
  - ឈរទី `j` ទាំងមូល
* - `x[a:b]`
  - `(4,5)` → `(b-a, 5)`
  - slice ធម្មតា (មិនរាប់ `b`)
* - `x[..., k]`
  - `(2,3,4)` → `(2,3)`
  - `...` = វិមាត្រដែលនៅសល់ទាំងអស់
* - `x[idx_tensor]`
  - `(4,5)` → `(len(idx), 5)`
  - advanced indexing — **ចម្លងទិន្នន័យ**
* - `x[bool_mask]`
  - `(4,5)` → `(n_true,)`
  - boolean masking — **ចម្លងទិន្នន័យ** ហើយតែងតែផ្ដល់ tensor 1D
* - `torch.where(cond, a, b)`
  - ដូច `a`
  - ជ្រើសរើសម្ដងមួយធាតុ — **រក្សា shape**
```

````{admonition} Boolean masking បាត់បង់ shape
:class: warning
`x[x > 0]` តែងតែត្រឡប់ tensor **1D** ទោះបី `x` មានវិមាត្រច្រើនក៏ដោយ
ព្រោះចំនួនធាតុដែល `True` អាចមិនទៀងទាត់។

បើអ្នកត្រូវរក្សា shape ប្រើ `torch.where(cond, x, replacement)` វិញ៖

```python
torch.where(x > 0, x, torch.zeros_like(x))   # shape ដដែល
```
````

## ៤. ប្រមាណវិធីលើ Tensor

```{literalinclude} examples/tensors_4.py
:language: python
:caption: tensors_4.py — ✅ បានសាកល្បងលើ CPU
```

### Element-wise vs Matrix

នេះគឺជាកំហុសទី ១ របស់អ្នកដែលមកពី NumPy ឬគណិតវិទ្យា៖

```{list-table}
:header-rows: 1
:widths: 22 26 52

* - ត្រូវការ
  - ប្រើ
  - លទ្ធផលសម្រាប់ `(2,2)` ម៉ាទ្រីស
* - គុណម្ដងមួយធាតុ
  - `a * b` ឬ `torch.mul(a, b)`
  - `[[1*10, 2*20], [3*30, 4*40]]`
* - Matrix multiply
  - `a @ b`, `torch.matmul(a,b)`, `a.mm(b)`
  - ផលបូកតាមជួរ×ឈរ
* - Dot product (1D)
  - `torch.dot(v1, v2)`
  - scalar
* - Batched matmul
  - `torch.bmm(A, B)` ឬ `A @ B`
  - `(N,i,k) @ (N,k,j)` → `(N,i,j)`
```

### Reductions និង keepdim

```{list-table}
:header-rows: 1
:widths: 34 22 44

* - ប្រតិបត្តិការលើ `(3, 4)`
  - លទ្ធផល shape
  - អត្ថន័យ
* - `x.sum()`
  - scalar
  - បូកធាតុទាំងអស់
* - `x.sum(dim=0)`
  - `(4,)`
  - បូកជួរចូលគ្នា → មួយតម្លៃក្នុងមួយឈរ
* - `x.sum(dim=1)`
  - `(3,)`
  - បូកឈរចូលគ្នា → មួយតម្លៃក្នុងមួយជួរ
* - `x.sum(dim=1, keepdim=True)`
  - `(3, 1)`
  - ដូចខាងលើ ប៉ុន្តែរក្សាវិមាត្រដែលបាត់ជា ១
* - `x.mean(dim=(1,2))`
  - `(N,)` បើ `x` ជា `(N,C,H,W)`
  - បង្រួមច្រើនវិមាត្រក្នុងពេលតែមួយ
* - `x.max(dim=1)`
  - namedtuple `(values, indices)`
  - ⚠️ ត្រឡប់ពីរតម្លៃ មិនមែន tensor ទេ
* - `x.argmax(dim=1)`
  - `(3,)`
  - index នៃតម្លៃអតិបរមា
```

**ហេតុអ្វី `keepdim=True` សំខាន់?** ព្រោះវាធ្វើឱ្យ broadcasting ដំណើរការដោយស្វ័យប្រវត្តិ៖

```python
x = torch.randn(32, 10)                 # (batch, features)
mean = x.mean(dim=1, keepdim=True)      # (32, 1)  <- អាច broadcast ជាមួយ (32, 10)
centered = x - mean                     # (32, 10) ✓

mean = x.mean(dim=1)                    # (32,)    <- គ្មានវិមាត្រទី ២
centered = x - mean                     # RuntimeError!
```

មើល {doc}`shapes-and-broadcasting` សម្រាប់ច្បាប់ពេញលេញ។

### In-place operations (បញ្ចប់ដោយ _)

អនុគមន៍ដែលបញ្ចប់ដោយ `_` កែ tensor ដើមផ្ទាល់៖ `add_()`, `mul_()`, `zero_()`,
`clamp_()`, `fill_()`, `copy_()`, `normal_()`, `uniform_()`, `masked_fill_()`។

````{admonition} កុំប្រើ in-place លើ tensor ដែលត្រូវការសម្រាប់ backward
:class: danger
បើ tensor មួយត្រូវបាន PyTorch កត់ត្រាទុកសម្រាប់គណនា gradient ការកែវាដោយ in-place
នឹងបំផ្លាញ graph៖

```text
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation
```

វិធីសុវត្ថិភាព៖ ធ្វើ in-place កែតែក្នុង `with torch.no_grad():` ឬតាមរយៈ
`optimizer.step()` ប៉ុណ្ណោះ។ មើល {doc}`autograd`។
````

## ៥. ការផ្លាស់ប្ដូររូបរាង

```{literalinclude} examples/tensors_5.py
:language: python
:caption: tensors_5.py — ✅ បានសាកល្បងលើ CPU
```

### view / reshape / permute / transpose / cat / stack

```{list-table}
:header-rows: 1
:widths: 26 26 48

* - ប្រតិបត្តិការលើ `(2,3,4)`
  - លទ្ធផល
  - ចំណាំ
* - `.view(6, 4)`
  - `(6, 4)`
  - ទាមទារ contiguous។ ចែក memory។
* - `.reshape(6, 4)`
  - `(6, 4)`
  - ចម្លងបើចាំបាច់។ សុវត្ថិភាពជាង។
* - `.reshape(-1, 4)`
  - `(6, 4)`
  - `-1` = ឱ្យ PyTorch គណនាវិមាត្រនោះ
* - `.flatten()`
  - `(24,)`
  - រួមវិមាត្រទាំងអស់
* - `.transpose(0, 1)`
  - `(3, 2, 4)`
  - ប្ដូរវិមាត្រពីរ។ មិន contiguous។
* - `.permute(2, 0, 1)`
  - `(4, 2, 3)`
  - រៀបវិមាត្រទាំងអស់ឡើងវិញ
* - `.T`
  - —
  - តែសម្រាប់ tensor ≤ 2D ប៉ុណ្ណោះ
* - `.unsqueeze(0)`
  - `(1, 2, 3, 4)`
  - បន្ថែមវិមាត្រទំហំ ១ (ជាទូទៅ batch dim)
* - `.squeeze()`
  - ដកវិមាត្រទំហំ ១ ចេញ
  - ⚠️ អាចដកវិមាត្រដែលអ្នកចង់រក្សា
* - `.expand(3, 2, 3, 4)`
  - `(3, 2, 3, 4)`
  - ពង្រីកវិមាត្រទំហំ ១ **ដោយមិនចម្លង memory**
* - `.repeat(3, 1, 1, 1)`
  - `(3, 2, 3, 4)`
  - ដូច expand ប៉ុន្តែ **ចម្លង memory ពិត**
```

### cat vs stack — ភាពខុសគ្នាសំខាន់

```python
p = torch.ones(2, 3)
q = torch.zeros(2, 3)

torch.cat([p, q], dim=0)     # shape (4, 3)   — ភ្ជាប់តាមវិមាត្រដែលមានស្រាប់
torch.stack([p, q], dim=0)   # shape (2, 2, 3) — បង្កើតវិមាត្រថ្មី
```

- **`cat`** ទាមទារឱ្យ shape ដូចគ្នា **លើកលែងតែវិមាត្រដែលកំពុងភ្ជាប់**។
- **`stack`** ទាមទារឱ្យ shape **ដូចគ្នាទាំងស្រុង**។

````{admonition} squeeze() គ្រោះថ្នាក់ជាងដែលអ្នកគិត
:class: warning
```python
torch.zeros(1, 3, 1, 4).squeeze().shape    # torch.Size([3, 4])
```

វាដកវិមាត្រទំហំ ១ **ទាំងអស់** ចេញ។ បើទិន្នន័យរបស់អ្នកមាន batch size = ១
ដោយចៃដន្យ `squeeze()` នឹងដក batch dimension ចេញ ហើយកូដដែលនៅសល់នឹងខូច។

ប្រើជាក់លាក់ជាង៖

```python
t.squeeze(0)          # ដកតែវិមាត្រទី 0
t.squeeze(-1)         # ដកតែវិមាត្រចុងក្រោយ
```
````

## ភាពជាក់លាក់នៃលេខទសភាគ

កុំប្រើ `==` លើ float tensor។ float32 មានខ្ទង់សំខាន់តែប្រហែល ៧ ខ្ទង់
ដូច្នេះការបូកបន្តិចម្ដងៗបង្កើតកំហុស rounding៖

```python
acc = torch.full((10,), 0.1).sum()      # បូក 0.1 ចំនួន 10 ដង
acc == torch.tensor(1.0)                # tensor([False])  ⚠️
torch.allclose(acc, torch.tensor(1.0))  # True  ✓

diff = torch.tensor(0.3) - torch.tensor(0.1)
diff == torch.tensor(0.2)               # tensor([False])  ⚠️
torch.allclose(diff, torch.tensor(0.2)) # True  ✓
```

```{list-table} ប្រៀបធៀបអនុគមន៍ប្រៀបធៀប
:header-rows: 1
:widths: 30 70

* - ប្រើ
  - ពេលណា
* - `torch.allclose(a, b)`
  - ប្រៀបធៀប tensor ពីរដោយអត់ឱនកំហុស rounding (លំនាំដើម `rtol=1e-5, atol=1e-8`)
* - `torch.isclose(a, b)`
  - ដូចគ្នា ប៉ុន្តែត្រឡប់ tensor bool ម្ដងមួយធាតុ
* - `torch.equal(a, b)`
  - តែពេលអ្នកដឹងថាគ្មានការគណនាលេខទសភាគ (ឧទាហរណ៍ index, mask)
* - `(a - b).abs().max()`
  - ពេលអ្នកចង់ដឹង **ទំហំ** នៃភាពខុសគ្នា
```

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `view size is not compatible with input tensor's size and stride`
  - tensor មិន contiguous
  - `.contiguous().view(...)` ឬ `.reshape(...)`
* - `The size of tensor a (X) must match tensor b (Y)`
  - broadcasting បរាជ័យ
  - ពិនិត្យ shape — មើល {doc}`shapes-and-broadcasting`
* - `Can't call numpy() on Tensor that requires grad`
  - tensor ជាប់ autograd graph
  - `t.detach().numpy()`
* - `can't convert cuda:0 device type tensor to numpy`
  - tensor នៅលើ GPU
  - `t.detach().cpu().numpy()`
* - `expected scalar type Long but found Float`
  - index/label dtype ខុស
  - `.long()` សម្រាប់ index
* - `only integers, slices, ellipsis ... are valid indices`
  - index ជា float
  - `int(i)` ឬ `.long()`
* - `index X is out of bounds for dimension 0 with size Y`
  - index លើសទំហំ
  - ពិនិត្យ `t.shape[dim]` មុន index
* - `too many dimensions 'str'`
  - ព្យាយាមដាក់ string ក្នុង tensor
  - tensor ផ្ទុកបានតែលេខ — ប្រើ Python list
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់ដែលត្រូវចាំ
:class: tip
1. **`shape` និង `dtype` គឺជាព័ត៌មានសំខាន់បំផុត** — ពិនិត្យវាមុនពេលអ្វីផ្សេងទៀត។
2. **`dim=N` មានន័យថាវិមាត្រទី N បាត់បង់** មិនមែន "តាមវិមាត្រទី N" ទេ។
3. **`reshape()` ល្អជាង `view()`** សម្រាប់កូដប្រចាំថ្ងៃ។
4. **`cat` ភ្ជាប់ · `stack` បង្កើតវិមាត្រថ្មី**។
5. **ប្រើ `torch.allclose()` មិនមែន `==`** សម្រាប់ float។
6. **`*_like()` រក្សា shape/dtype/device** — មានប្រយោជន៍ពេលប្ដូរ device។
7. **Views ចែក memory** — កែ view គឺកែ tensor ដើម។
```

## មេរៀនបន្ទាប់

ឥឡូវអ្នកដឹងពីរបៀបបង្កើត និងគណនាលើ tensor ហើយ។ ជំហានបន្ទាប់គឺយល់ពី
របៀបដែល shape ធ្វើអន្តរកម្មគ្នា៖ {doc}`shapes-and-broadcasting`។
