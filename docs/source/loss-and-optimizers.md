# Loss Functions និង Optimizers

Loss function ប្រាប់ម៉ូដែលថា **វាខុសប៉ុន្មាន**។ Optimizer ប្រាប់វាថា **ត្រូវកែយ៉ាងម៉េច**។
ការជ្រើសរើសខុសគឺជាមូលហេតុទីមួយដែលម៉ូដែលមិនរៀន។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ៣ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## ១. Loss Functions

### ជ្រើសរើស loss ត្រឹមត្រូវ

```{list-table}
:header-rows: 1
:widths: 26 28 22 24

* - ប្រភេទបញ្ហា
  - Loss function
  - Target dtype
  - Model output
* - Multi-class classification (១ ថ្នាក់ក្នុងមួយសំណាក)
  - `nn.CrossEntropyLoss`
  - `int64` (class index)
  - **logits** `(N, C)`
* - Binary classification
  - `nn.BCEWithLogitsLoss`
  - `float32` (0.0 / 1.0)
  - **logits** `(N, 1)` ឬ `(N,)`
* - Multi-label (ច្រើនថ្នាក់ក្នុងមួយសំណាក)
  - `nn.BCEWithLogitsLoss`
  - `float32` `(N, C)`
  - **logits** `(N, C)`
* - Regression
  - `nn.MSELoss` (L2)
  - `float32`
  - តម្លៃ `(N, out)`
* - Regression ជាមួយ outliers
  - `nn.L1Loss` ឬ `nn.SmoothL1Loss`
  - `float32`
  - តម្លៃ `(N, out)`
* - Counting / អាំងតង់ស៊ីតេ
  - `nn.PoissonNLLLoss`
  - `float32`
  - log-rate
* - Segmentation
  - `nn.CrossEntropyLoss` ឬ `nn.DiceLoss` (custom)
  - `int64` `(N, H, W)`
  - `(N, C, H, W)`
```

```{literalinclude} examples/loss_and_optimizers_1.py
:language: python
:caption: loss_and_optimizers_1.py — ✅ បានសាកល្បងលើ CPU
```

### CrossEntropyLoss គឺជា ៣ អនុគមន៍ក្នុងមួយ

```text
logits ──► log_softmax ──► nll_loss ──► scalar loss
```

នេះមានន័យថា **អ្នកមិនត្រូវហៅ softmax មុនទេ**។ បើអ្នកធ្វើ អ្នកនឹងទទួលបាន
"double softmax" ដែលបំផ្លាញ gradient៖

```python
# ❌ ខុស
loss = nn.CrossEntropyLoss()(torch.softmax(logits, dim=1), target)

# ✅ ត្រូវ
loss = nn.CrossEntropyLoss()(logits, target)
```

ការរួមបញ្ចូលនេះក៏ផ្ដល់ **ស្ថិរភាពលេខ** ផងដែរ — PyTorch ប្រើ log-sum-exp trick
ខាងក្នុង ដូច្នេះវាមិន overflow ជាមួយ logits ធំទេ។

````{admonition} Target ត្រូវតែជា class index មិនមែន one-hot
:class: danger
```python
logits = torch.randn(4, 3)          # (batch=4, classes=3)

# ✅ ត្រូវ — int64 class index, shape (N,)
target = torch.tensor([0, 2, 1, 1])
nn.CrossEntropyLoss()(logits, target)

# ❌ ខុស — one-hot float
target_oh = torch.nn.functional.one_hot(target, 3).float()
nn.CrossEntropyLoss()(logits, target_oh)   # ផ្ដល់លទ្ធផលខុស ដោយគ្មាន error!
```

កំណែ one-hot មិនបង្ហាញ error ទេ (PyTorch ទទួលយក soft targets) ប៉ុន្តែលទ្ធផល
មានន័យខុសគ្នាទាំងស្រុង។ **ពិនិត្យ `target.dtype` ជានិច្ច។**
````

### Class weights និង label smoothing

សម្រាប់ទិន្នន័យដែលមិនស្មើគ្នា (imbalanced)៖

```python
counts = torch.bincount(labels).float()
weights = counts.sum() / (len(counts) * counts)     # inverse frequency
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

```{list-table} ជម្រើសរបស់ CrossEntropyLoss
:header-rows: 1
:widths: 26 18 56

* - ប៉ារ៉ាម៉ែត្រ
  - លំនាំដើម
  - ប្រើពេល
* - `weight`
  - `None`
  - ថ្នាក់មិនស្មើគ្នា។ Tensor float ទំហំ `(C,)`
* - `ignore_index`
  - `-100`
  - សំណាកដែលមិនត្រូវរាប់ក្នុង loss (ឧ. padding tokens)
* - `label_smoothing`
  - `0.0`
  - ការពារ overconfidence។ តម្លៃធម្មតា 0.05–0.1
* - `reduction`
  - `'mean'`
  - `'none'` ផ្ដល់ loss ក្នុងមួយសំណាក (សម្រាប់ weighting ផ្ទាល់ខ្លួន)
```

### Reduction options

```python
pred, true = torch.randn(4, 2), torch.randn(4, 2)
nn.MSELoss(reduction="mean")(pred, true)   # scalar — លំនាំដើម
nn.MSELoss(reduction="sum")(pred, true)    # scalar — ផលបូក
nn.MSELoss(reduction="none")(pred, true)   # shape (4, 2) — ម្ដងមួយធាតុ
```

ប្រើ `reduction="none"` ពេលអ្នកត្រូវការ weighting ផ្ទាល់ខ្លួន (ឧទាហរណ៍ focal loss)៖

```python
per_sample = nn.MSELoss(reduction="none")(pred, true).mean(dim=1)
loss = (per_sample * sample_weights).mean()
```

## ២. Optimizers

```{literalinclude} examples/loss_and_optimizers_2.py
:language: python
:caption: loss_and_optimizers_2.py — ✅ បានសាកល្បងលើ CPU (ប្រៀបធៀប optimizer ៦ ប្រភេទ)
```

### ជ្រើសរើស optimizer

```{list-table}
:header-rows: 1
:widths: 22 24 54

* - Optimizer
  - lr ចាប់ផ្ដើម
  - ប្រើពេល
* - `SGD`
  - 0.01 – 0.1
  - ពេលអ្នកត្រូវការ convergence ល្អបំផុត និងមានពេល tune។ បន្ថែម `momentum=0.9` ជានិច្ច
* - `Adam`
  - 1e-3 – 3e-4
  - លំនាំដើមសម្រាប់ Transformer, RNN, ការពិសោធន៍រហ័ស
* - `AdamW`
  - 1e-3 – 3e-4
  - **ណែនាំជាលំនាំដើម** ពេលប្រើ `weight_decay > 0`
* - `RMSprop`
  - 1e-3 – 1e-2
  - RNN (ជំនួស Adam ក្នុងករណីខ្លះ)
* - `Adagrad`
  - 0.01 – 0.1
  - ទិន្នន័យ sparse ខ្លាំង (ឧ. recommendation)
* - `LBFGS`
  - 1.0
  - បញ្ហា optimization តូច ដែលត្រូវការ convergence ត្រឹមត្រូវបំផុត
```

### Adam vs AdamW

ភាពខុសគ្នាសំខាន់នៅពេលប្រើ `weight_decay`៖

```text
Adam :  grad ← grad + λ·w        (L2 penalty បញ្ចូលក្នុង gradient)
        w ← w - lr · adapt(grad)  <- ការធ្វើស្តង់ដារ adaptive ធ្វើឱ្យ decay ខុសគ្នាតាម parameter

AdamW:  w ← w - lr · adapt(grad) - lr · λ · w   (decoupled — ដកដោយផ្ទាល់)
```

`AdamW` គឺជាការកែត្រឹមត្រូវ។ បើអ្នកប្រើ `weight_decay > 0` **ប្រើ `AdamW`**។

### Parameter groups

ប្រើ learning rate ឬ weight decay ផ្សេងគ្នាសម្រាប់ជាន់ផ្សេងគ្នា៖

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},      # pretrained — lr តូច
    {"params": model.head.parameters(),     "lr": 1e-3},      # ថ្មី — lr ធំ
    {"params": model.norm.parameters(),     "weight_decay": 0.0},  # កុំ decay LayerNorm
])
```

````{admonition} កុំ weight_decay លើ bias និង normalization parameters
:class: tip
ការ decay លើ bias, `LayerNorm.weight` និង `BatchNorm.weight` ជាទូទៅធ្វើឱ្យ
លទ្ធផលអាក្រក់។ បែងចែកវាទៅ param group ដាច់ដោយឡែកដែលមាន `weight_decay=0.0`៖

```text
decay     :  weight matrices (Linear.weight, Conv2d.weight, Embedding.weight)
no decay  :  *.bias, LayerNorm.weight, BatchNorm.weight
```
````

## ៣. Learning Rate Schedulers

```{literalinclude} examples/loss_and_optimizers_3.py
:language: python
:caption: loss_and_optimizers_3.py — ✅ បានសាកល្បងលើ CPU (រួមទាំង warmup + cosine decay)
```

### ជ្រើសរើស scheduler

```{list-table}
:header-rows: 1
:widths: 26 20 54

* - Scheduler
  - step() ពេលណា
  - ប្រើពេល
* - `StepLR(step_size, gamma)`
  - រាល់ epoch
  - សាមញ្ញ។ បន្ថយ lr ជាកាំជណ្ដើរ
* - `MultiStepLR(milestones)`
  - រាល់ epoch
  - អ្នកដឹងច្បាស់ពី epoch ដែលត្រូវបន្ថយ
* - `CosineAnnealingLR(T_max)`
  - រាល់ epoch
  - **លំនាំដើមល្អបំផុត** សម្រាប់ភាគច្រើន
* - `CosineAnnealingWarmRestarts(T_0)`
  - រាល់ epoch
  - snapshot ensembling
* - `OneCycleLR(max_lr, total_steps)`
  - ⚠️ **រាល់ step**
  - ល្បឿនបណ្ដុះបណ្ដាលអតិបរមា (super-convergence)
* - `LinearLR(start_factor, total_iters)`
  - រាល់ epoch
  - warmup តែម្នាក់ឯង
* - `LambdaLR(lr_lambda)`
  - រាល់ epoch
  - កំណត់ curve ផ្ទាល់ខ្លួន
* - `SequentialLR(schedulers, milestones)`
  - រាល់ epoch
  - ភ្ជាប់ warmup + decay
* - `ReduceLROnPlateau(patience)`
  - ⚠️ **ជាមួយ metric**
  - ពេលអ្នកមិនដឹងចំនួន epoch ជាមុន
```

````{admonition} ReduceLROnPlateau ខុសពី scheduler ផ្សេងទាំងអស់
:class: warning
វាទាមទារ metric ជា argument៖

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

for epoch in range(epochs):
    train(...)
    val_loss = evaluate(...)
    scheduler.step(val_loss)        # ← ត្រូវបញ្ជូន metric!
```

បើអ្នកហៅ `scheduler.step()` ដោយគ្មាន argument អ្នកនឹងទទួលបាន៖

```text
TypeError: ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics'
```
````

````{admonition} លំដាប់ត្រឹមត្រូវនៃការហៅ step()
:class: important
```python
for epoch in range(epochs):
    train_one_epoch(...)          # រួមមាន optimizer.step() ខាងក្នុង
    val_loss = evaluate(...)
    scheduler.step()              # ← បន្ទាប់ពី epoch ទាំងមូល
```

កំហុសទូទៅគឺហៅ `scheduler.step()` **មុន** `optimizer.step()`។ ចាប់ពី PyTorch 1.1
វាបង្ហាញព្រមាន៖

```text
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
```
````

## ៤. Gradient Clipping

ការពារ **exploding gradients** ដែលជាបញ្ហាទូទៅក្នុង RNN និង model ជ្រៅ៖

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # ← មុន step()
optimizer.step()
```

```{list-table}
:header-rows: 1
:widths: 30 70

* - អនុគមន៍
  - អ្វីដែលវាធ្វើ
* - `clip_grad_norm_(params, max_norm)`
  - ពង្រីក/បង្រួម gradient ទាំងអស់ដើម្បីឱ្យ **norm សរុប** ≤ `max_norm`។ រក្សាទិសដៅ
* - `clip_grad_value_(params, clip_value)`
  - កាត់ធាតុនីមួយៗទៅ `[-clip_value, clip_value]`។ ប្ដូរទិសដៅ
```

```{admonition} Clipping ត្រូវតែនៅចន្លោះ backward() និង step()
:class: warning
បើអ្នក clip មុន `backward()` គ្មាន gradient ទេ។ បើអ្នក clip បន្ទាប់ពី `step()`
weight ត្រូវបានធ្វើបច្ចុប្បន្នភាពដោយ gradient ដែលមិនបាន clip រួចហើយ។

លំដាប់ត្រឹមត្រូវគឺ៖ `zero_grad()` → `backward()` → `clip_grad_norm_()` → `step()`
```

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `0D or 1D target tensor expected`
  - target shape ខុសសម្រាប់ CrossEntropyLoss
  - logits `(N,C)` ត្រូវការ target `(N,)`
* - `expected scalar type Long but found Float`
  - target ជា float
  - `target.long()` — class index ត្រូវតែជា int64
* - `using a target size different from the input size`
  - pred និង target shape ខុសគ្នា
  - ពិនិត្យ `pred.shape == target.shape`
* - `The size of tensor a (N) must match ...` ក្នុង loss
  - `weight` tensor មានទំហំខុស
  - `weight` ត្រូវតែ `(C,)` ដែល C = ចំនួនថ្នាក់
* - `ValueError: optimizer got an empty parameter list`
  - `model.parameters()` ទទេ
  - ភ្លេច `super().__init__()` ឬប្រើ list ធម្មតាជំនួស `ModuleList`
* - loss = `nan` ពីជំហានដំបូង
  - lr ធំពេក ឬ input មាន inf
  - បន្ថយ lr ១០ដង ពិនិត្យទិន្នន័យ
* - loss មិនថយចុះសោះ
  - ភ្លេច `optimizer.step()` ឬ `zero_grad()`
  - ពិនិត្យវដ្ដ ៥ ជំហាន។ មើល {doc}`debugging`
* - `step() missing 1 required positional argument: 'metrics'`
  - `ReduceLROnPlateau` ត្រូវការ metric
  - `scheduler.step(val_loss)`
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **CrossEntropyLoss ទទួល logits + int64 class index** — មិនមែន softmax មិនមែន one-hot។
2. **BCEWithLogitsLoss ទទួល logits + float32 0/1** — សម្រាប់ binary និង multi-label។
3. **AdamW ជាលំនាំដើមល្អ** លើកលែងតែអ្នកមានហេតុផលប្រើ SGD+momentum។
4. **CosineAnnealingLR ឬ OneCycleLR** សម្រាប់ schedule ភាគច្រើន។
5. **`scheduler.step()` បន្ទាប់ពី epoch** — លើកលែង OneCycleLR ដែលហៅរាល់ step។
6. **`ReduceLROnPlateau.step(val_loss)`** — ត្រូវបញ្ជូន metric។
7. **Clip gradient រវាង backward និង step**។
8. **កុំ weight_decay លើ bias និង normalization weights**។
```

## មេរៀនបន្ទាប់

{doc}`data-and-dataloader` — របៀបបញ្ចូលទិន្នន័យទៅក្នុងម៉ូដែលរបស់អ្នក។
