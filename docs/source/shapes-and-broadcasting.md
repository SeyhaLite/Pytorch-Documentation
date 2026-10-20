# Shapes, Dimensions និង Broadcasting

បញ្ហា shape គឺជាមូលហេតុនៃកំហុសភាគច្រើនដែលអ្នកចាប់ផ្ដើមជួប។ មេរៀននេះបង្រៀន
អ្នកឱ្យ **គិតជា shape** និងយល់ពី broadcasting — លក្ខណៈពិសេសដែលមានអានុភាពខ្លាំង
ប៉ុន្តែអាចបង្កើតកំហុសដែល **មិនបង្ហាញសារ error ណាមួយឡើយ**។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ៣ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## ១. វិមាត្រមានន័យយ៉ាងដូចម្តេច?

វិមាត្រនីមួយៗនៃ tensor ជាទូទៅតំណាងឱ្យអ្វីមួយជាក់លាក់។ ការចាំទម្រង់ទូទៅទាំងនេះ
ជួយអ្នកអានកូដបានលឿន៖

```{list-table} ទម្រង់ tensor ទូទៅ
:header-rows: 1
:widths: 26 26 48

* - ទម្រង់
  - វិមាត្រ
  - ប្រើសម្រាប់
* - `()`
  - scalar
  - loss, metric, តម្លៃតែមួយ
* - `(N,)`
  - vector
  - label ក្នុងមួយសំណាក, prediction មួយ
* - `(N, F)`
  - matrix
  - tabular data — N សំណាក ជាមួយ F features
* - `(N, C, H, W)`
  - 4D — **NCHW**
  - batch នៃរូបភាព (លំនាំដើមរបស់ PyTorch)
* - `(N, H, W, C)`
  - 4D — **NHWC**
  - batch នៃរូបភាព (ទម្រង់ TensorFlow / `channels_last`)
* - `(N, L, D)`
  - 3D
  - batch នៃលំដាប់ (N batch, L tokens, D embedding)
* - `(N, H, L, L)`
  - 4D
  - attention weights (H heads)
* - `(out, in)`
  - 2D
  - weight matrix របស់ `nn.Linear` ⚠️ បញ្ច្រាសពីដែលអ្នករំពឹង
* - `(out, in, kH, kW)`
  - 4D
  - weight របស់ `nn.Conv2d`
```

````{admonition} nn.Linear រក្សាទុក weight ជា (out_features, in_features)
:class: warning
```python
layer = nn.Linear(in_features=10, out_features=5)
layer.weight.shape    # torch.Size([5, 10])  <- បញ្ច្រាស!
layer.bias.shape      # torch.Size([5])
```

ខាងក្នុង PyTorch គណនា `y = x @ W.T + b`។ បើអ្នកព្យាយាមផ្ទុក weight
ពី framework ផ្សេង (Keras, JAX) អ្នកអាចត្រូវការ `.T`។
````

```{literalinclude} examples/shapes_1.py
:language: python
:caption: shapes_1.py — ✅ បានសាកល្បងលើ CPU (shape, stride, views)
```

## ២. ច្បាប់ Broadcasting

**Broadcasting** អនុញ្ញាតឱ្យ PyTorch ធ្វើប្រតិបត្តិការលើ tensor ដែលមាន shape ខុសគ្នា
ដោយ "ពង្រីក" វិមាត្រទំហំ ១ ដោយស្វ័យប្រវត្តិ — ដោយ **មិនចម្លង memory**។

### ច្បាប់ទាំង ៣

អាន shape ពី **វិមាត្រចុងក្រោយទៅមុខ**៖

1. **បើចំនួនវិមាត្រមិនស្មើគ្នា** — បន្ថែមវិមាត្រទំហំ `1` នៅខាង **មុខ** របស់ tensor ខ្លីជាង។
2. **សម្រាប់វិមាត្រនីមួយៗ** — ទំហំត្រូវតែ៖ (ក) ស្មើគ្នា ឬ (ខ) មួយក្នុងចំណោមវាស្មើ `1`។
3. **ទំហំ ១ ត្រូវបានពង្រីក** ដោយ repeat ដើម្បីត្រូវនឹងទំហំម្ខាងទៀត។

### ឧទាហរណ៍ដើរតាមជំហាន

```text
  (4, 3)     <- tensor A
+    (3,)    <- tensor B

ជំហាន ១: B ខ្លីជាង ដូច្នេះបន្ថែមវិមាត្រមុខ  ->  (1, 3)
ជំហាន ២: ប្រៀបធៀបវិមាត្រចុងក្រោយ៖ 3 vs 3  ✓  /  វិមាត្រមុខ៖ 4 vs 1  ✓ (មាន 1)
ជំហាន ៣: ពង្រីក B ទៅ (4, 3) ដោយចម្លងជួរដដែល ៤ ដង (គ្មានការចម្លង memory ពិត)

លទ្ធផល: (4, 3)
```

```{list-table} ករណីជាច្រើន
:header-rows: 1
:widths: 24 24 20 32

* - Shape A
  - Shape B
  - លទ្ធផល
  - ការពន្យល់
* - `(4, 3)`
  - `(3,)`
  - `(4, 3)` ✓
  - B → `(1,3)` → `(4,3)`
* - `(4, 3)`
  - `(4, 1)`
  - `(4, 3)` ✓
  - B ពង្រីកតាមឈរ
* - `(8, 1, 6, 1)`
  - `(7, 1, 5)`
  - `(8, 7, 6, 5)` ✓
  - B → `(1,7,1,5)` ហើយពង្រីកទាំងពីរ
* - `(2, 3)`
  - `(3, 2)`
  - ❌ RuntimeError
  - 3≠2 និងគ្មាន 1 ដើម្បីពង្រីក
* - `(3,)`
  - `(4,)`
  - ❌ RuntimeError
  - 3≠4 ហើយគ្មាន 1
* - `(5, 1)`
  - `(1, 4)`
  - `(5, 4)` ✓
  - outer product — ទម្រង់មានប្រយោជន៍ខ្លាំង
```

### ពិនិត្យមុនគណនា

បើអ្នកចង់ដឹងថា shape ពីរអាច broadcast បានឬទេ **ដោយមិនបង្កើត tensor**៖

```python
torch.broadcast_shapes((4, 3), (3,))       # torch.Size([4, 3])
torch.broadcast_shapes((2, 3), (3, 2))     # RuntimeError
```

```{literalinclude} examples/shapes_2.py
:language: python
:caption: shapes_2.py — ✅ បានសាកល្បងលើ CPU (ច្បាប់, ការប្រើប្រាស់ពិត, silent bug)
```

## ៣. Broadcasting ក្នុងការអនុវត្ត

ទម្រង់ដែលអ្នកនឹងប្រើញឹកញាប់បំផុត៖

```{list-table}
:header-rows: 1
:widths: 34 24 42

* - គោលបំណង
  - កូដ
  - Shape ការវិវត្ត
* - Standardize តាម feature
  - `(x - mean) / std`
  - `(N,F) - (F,)` → `(N,F)`
* - ដកមធ្យមភាគជួរ
  - `x - x.mean(1, keepdim=True)`
  - `(N,F) - (N,1)` → `(N,F)`
* - គុណនឹង class weight
  - `logits * weights`
  - `(N,C) * (C,)` → `(N,C)`
* - Pairwise distance
  - `(a[:,None,:] - b[None,:,:]).pow(2).sum(-1)`
  - `(N,1,D) - (1,M,D)` → `(N,M,D)` → `(N,M)`
* - One-hot encoding
  - `arange(C) == labels[:,None]`
  - `(C,) == (N,1)` → `(N,C)`
* - Mask padding
  - `logits.masked_fill(mask==0, -inf)`
  - `(N,L,C)` ជាមួយ `(N,1,L)` → `(N,L,C)`
```

### ករណីសិក្សា៖ Softmax មានស្ថិរភាព

```python
# ខុស — exp() overflow ជាមួយ logits ធំ
naive = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)

# ត្រឹមត្រូវ — ដក max ជាមុន (លទ្ធផលគណិតវិទ្យាដូចគ្នា ប៉ុន្តែស្ថិរជាង)
shifted = logits - logits.max(dim=1, keepdim=True).values
stable = torch.exp(shifted) / torch.exp(shifted).sum(dim=1, keepdim=True)

# ល្អបំផុត — ប្រើអនុគមន៍ដែលមានស្រាប់ (វាធ្វើការដក max ខាងក្នុង)
official = torch.softmax(logits, dim=1)
```

ហេតុអ្វីត្រូវដក max? ព្រោះ `exp(1000)` = `inf` ក្នុង float32 ប៉ុន្តែ
`exp(1000 - 1000) = exp(0) = 1`។ ការដកថេរមួយចេញពី logits ទាំងអស់មិនផ្លាស់ប្ដូរ
លទ្ធផល softmax ទេ ព្រោះវាលុបចោលគ្នានៅភាគយក និងភាគបែង។

```{admonition} ប្រើ torch.softmax() ជានិច្ច
:class: tip
កុំសរសេរ softmax ដោយដៃក្នុងកូដផលិតកម្ម។ `torch.softmax()` និង `nn.CrossEntropyLoss()`
បានដោះស្រាយស្ថិរភាពលេខរួចហើយ។

`nn.CrossEntropyLoss` ថែមទាំងរួមបញ្ចូល log-softmax ផងដែរ — ដូច្នេះផ្ដល់ **logits**
ទៅវា មិនមែន probabilities ទេ។ បើអ្នកផ្ដល់ softmax រួចហើយ អ្នកនឹងទទួលបាន
"double softmax" ដែលធ្វើឱ្យ gradient ខូច។ មើល {doc}`loss-and-optimizers`។
```

```{literalinclude} examples/shapes_3.py
:language: python
:caption: shapes_3.py — ✅ បានសាកល្បងលើ CPU (មធ្យមភាគតាមក្រុម, softmax, one-hot)
```

## ៤. Silent Bug — គ្រោះថ្នាក់ពិតនៃ Broadcasting

Broadcasting គឺជាលក្ខណៈពិសេសដ៏មានអានុភាព ប៉ុន្តែវាអាច **លាក់កំហុស**។
PyTorch នឹងមិនបញ្ចេញ error ទេ — វាគ្រាន់តែផ្ដល់ចម្លើយខុស។

### ឧទាហរណ៍ពិត

```python
losses  = torch.tensor([1.0, 2.0, 3.0, 4.0])    # shape (4,)
weights = torch.tensor([0.1, 0.2, 0.3, 0.4])    # shape (4,)

# ចេតនា៖ គុណ loss នីមួយៗជាមួយ weight របស់វា
correct = losses * weights                       # shape (4,)  ✓

# កំហុស៖ reshape ខុសដោយចៃដន្យ
wrong = losses.reshape(4, 1) * weights.reshape(1, 4)   # shape (4, 4)  ⚠️
```

`wrong` គឺជា **matrix 4×4** នៃផលគុណទាំងអស់ — មិនមែនអ្វីដែលអ្នកចង់បានទេ។
ប៉ុន្តែ PyTorch មិនបញ្ចេញ error ណាមួយឡើយ។

### វិធីការពារ

````{admonition} ទម្លាប់ការពារ ៥ យ៉ាង
:class: important
**១. Assert shape បន្ទាប់ពីប្រតិបត្តិការសំខាន់ៗ** ក្នុងអំឡុងពេលអភិវឌ្ឍ៖

```python
assert result.shape == expected_shape, f"{result.shape} != {expected_shape}"
```

**២. ប្រើ `keepdim=True`** ពេលធ្វើ reduction ដែលអ្នកនឹង broadcast វិញ។

**៣. សរសេរ shape ជា comment** នៅលើបន្ទាត់នីមួយៗក្នុង forward pass។

**៤. ពិនិត្យ `torch.broadcast_shapes()`** មុនធ្វើប្រតិបត្តិការដែលស្មុគស្មាញ។

**៥. បើលទ្ធផលមាន shape ធំជាងដែលអ្នករំពឹង** — សង្ស័យ broadcasting ភ្លាម។
````

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 36 28 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1`
  - broadcasting បរាជ័យ
  - ពិនិត្យ shape ទាំងពីរ។ តើអ្នកត្រូវការ `unsqueeze()` ទេ?
* - `dimension specified as 1 but tensor has no dimension 1`
  - `dim=` លើស ndim
  - ពិនិត្យ `t.ndim` ជាមុន
* - `shape '[-1, 4]' is invalid for input of size 10`
  - ចំនួនធាតុមិនចែកដាច់
  - `reshape` ទាមទារ `numel` ចែកដាច់
* - លទ្ធផលមាន shape ធំជាងដែលរំពឹង (គ្មាន error)
  - silent broadcasting
  - បន្ថែម `assert` លើ shape
* - `non-empty 3D or 4D (batch mode) tensor expected for input`
  - `BatchNorm` ត្រូវការ batch dim
  - `t.unsqueeze(0)` បើមានសំណាកតែមួយ
* - accuracy = 0 ឬ ~1/C ជានិច្ច
  - label និង prediction មាន shape ខុសគ្នា
  - ពិនិត្យ `preds.shape == labels.shape`
```

### Debug shape ដោយទម្លាប់

បន្ថែមអនុគមន៍នេះទៅក្នុងកូដរបស់អ្នក អំឡុងពេលអភិវឌ្ឍ៖

```python
def show(name: str, t: torch.Tensor) -> None:
    """បង្ហាញ shape/dtype/device ជារបៀបងាយអាន។"""
    print(f"{name:<22} shape={str(tuple(t.shape)):<16} "
          f"dtype={str(t.dtype):<16} dev={t.device}")

show("input", x)
show("after conv", model.features(x))
show("logits", logits)
```

ឬប្រើ forward hook ដើម្បីចាប់យក shape គ្រប់ជាន់ដោយមិនកែ `forward()`៖

```python
handles = []
for name, module in model.named_modules():
    if name and not list(module.children()):      # leaf modules តែប៉ុណ្ណោះ
        handles.append(module.register_forward_hook(
            lambda m, i, o, n=name: print(f"{n:<20} -> {tuple(o.shape)}")
        ))

out = model(x)
for h in handles:
    h.remove()                                     # កុំភ្លេចដកចេញ!
```

ឧទាហរណ៍ពេញលេញនៃបច្ចេកទេស hook នេះមាននៅក្នុង {doc}`neural-networks`។

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **អាន shape ពីស្ដាំទៅឆ្វេង** នៅពេលគិតអំពី broadcasting។
2. **វិមាត្រទំហំ ១ អាចពង្រីកបាន** — វិមាត្រផ្សេងទៀតត្រូវតែត្រូវគ្នាពិតប្រាកដ។
3. **`dim=N` = វិមាត្រទី N បាត់បង់**។ ប្រើ `keepdim=True` ដើម្បីរក្សាវាជា ១។
4. **Broadcasting អាចលាក់កំហុស** — assert shape ជាទម្លាប់។
5. **`unsqueeze(0)`** បន្ថែម batch dimension សម្រាប់សំណាកតែមួយ។
6. **stride កំណត់ល្បឿន** — ប្រតិបត្តិការតាមវិមាត្រ contiguous លឿនជាង។ មើល {doc}`performance`។
```

## មេរៀនបន្ទាប់

{doc}`devices` — របៀបផ្លាស់ទី tensor និង model រវាង CPU និង GPU។
