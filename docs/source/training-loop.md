# Training Loop

មេរៀននេះដាក់អ្វីៗទាំងអស់ចូលគ្នា៖ Dataset, DataLoader, model, loss, optimizer,
scheduler និងការវាយតម្លៃ — ជាវដ្ដបណ្ដុះបណ្ដាលពេញលេញដែលអ្នកអាចប្រើក្នុងគម្រោងពិត។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ២ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
`training_loop_1.py` រត់ប្រហែល 3–6 វិនាទី។
```

## រចនាសម្ព័ន្ធស្តង់ដារ

វដ្ដបណ្ដុះបណ្ដាល PyTorch មានរចនាសម្ព័ន្ធ ៣ ជាន់៖

```text
for epoch in range(epochs):            # ជាន់ក្រៅ
    model.train()
    for batch in train_loader:         # ជាន់កណ្ដាល
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()               # ជាន់ខាងក្នុង — វដ្ដ ៥ ជំហាន
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            ...                        # វាយតម្លៃ ដោយគ្មាន gradient
```

## ឧទាហរណ៍ពេញលេញ

```{literalinclude} examples/training_loop_1.py
:language: python
:caption: training_loop_1.py — ✅ បានសាកល្បងលើ CPU — val accuracy 0.91, val loss 0.178
```

### Output ពីការរត់ពិត

ខាងក្រោមគឺជា output ពិត ចម្លងដោយផ្ទាល់ពីការរត់ `training_loop_1.py`
លើ PyTorch `2.14.1+cpu`, Python 3.11.2 (ម៉ាស៊ីនដែលបង្កើតឯកសារនេះ)៖

```text
device          : cpu
train / val     : 800 / 200
epochs          : 6
batch size      : 32
lr / weight_dec : 0.01 / 0.001
==================================================================
parameters      : 1,732
------------------------------------------------------------------
epoch | train loss  train acc |  val loss  val acc |       lr |   sec
------------------------------------------------------------------
    1 |     1.0883     0.5575 |    0.6199   0.7950 | 0.010000 |  0.05  <- ល្អបំផុត
    2 |     0.3852     0.8600 |    0.3012   0.8650 | 0.009330 |  0.09  <- ល្អបំផុត
    3 |     0.2195     0.9187 |    0.1968   0.9300 | 0.007500 |  0.09  <- ល្អបំផុត
    4 |     0.2061     0.9137 |    0.1972   0.9000 | 0.005000 |  0.10  (មិនប្រសើរ 1/2)
    5 |     0.1556     0.9413 |    0.1784   0.9100 | 0.002500 |  0.09  <- ល្អបំផុត
    6 |     0.1539     0.9413 |    0.1786   0.9200 | 0.000670 |  0.09  (មិនប្រសើរ 1/2)

បានស្ដារ weight នៃ epoch ល្អបំផុត (val_loss = 0.1784)
==================================================================
រយៈពេលសរុប: 0.52s
val loss ចុងក្រោយ: 0.1784
val acc  ចុងក្រោយ: 0.9100
```

ដោយសារ `seed=42` ត្រូវបានកំណត់ លទ្ធផលលេខខាងលើអាចធ្វើឱ្យដូចគ្នាវិញបាន
នៅលើ CPU ជាមួយ PyTorch កំណែដូចគ្នា។ ជួរ `sec` នឹងខុសគ្នាតាម hardware។

សង្កេតពីរបីយ៉ាងពីតារាងនេះ៖

- **train loss ថយចុះជានិច្ច** ប៉ុន្តែ **val loss** ឈប់ប្រសើរនៅ epoch 4 និង 6 —
  នេះជា early-stopping signal ដែលកូដកត់ត្រា។
- **`val acc` ខ្ពស់ជាង `train acc`** ក្នុង epoch ដំបូង ព្រោះ `train()` mode បើក
  Dropout(0.1) ដែលធ្វើឱ្យការបណ្ដុះបណ្ដាលលំបាកជាង។ នេះធម្មតាទេ។
- **learning rate** ធ្វើតាម cosine curve ពី 0.01 ចុះទៅជិត 0។

## ធាតុផ្សំនីមួយៗ

### ១. Reproducibility

```python
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

ហៅវា **មុន** បង្កើត model ឬ DataLoader។ មើល {doc}`reproducibility` សម្រាប់
ការគ្រប់គ្រងពេញលេញ (NumPy, Python `random`, `Generator`, determinism)។

### ២. train() / eval() ឱ្យបានត្រឹមត្រូវ

```python
def train_one_epoch(...):
    model.train()          # ← Dropout សកម្ម, BatchNorm រៀន statistics
    ...

@torch.no_grad()
def evaluate(...):
    model.eval()           # ← Dropout បិទ, BatchNorm ប្រើ running stats
    ...
```

```{admonition} ដាក់ @torch.no_grad() លើ evaluate() ជា decorator
:class: tip
វាងាយភ្លេចជាងការសរសេរ `with torch.no_grad():` ជារាល់ប្លុក។ Decorator
ធានាថា **គ្មានផ្លូវណា** ដែលការវាយតម្លៃអាចបង្កើត gradient បាន។
```

### ៣. zero_grad(set_to_none=True)

```python
optimizer.zero_grad(set_to_none=True)     # ល្អជាង zero_grad() ធម្មតា
```

`set_to_none=True` កំណត់ `.grad = None` ជំនួសការបំពេញសូន្យ។ វា៖
- សន្សំ memory (គ្មាន tensor សូន្យត្រូវរក្សាទុក)
- លឿនជាងបន្តិច (គ្មានការសរសេរសូន្យ)
- ជាលំនាំដើមចាប់ពី PyTorch 2.0

### ៤. Gradient clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

មើល {doc}`loss-and-optimizers` សម្រាប់ការពន្យល់លម្អិត។

### ៥. Scheduler step

```python
for epoch in ...:
    train_one_epoch(...)          # រួមមាន optimizer.step() ច្រើនដង
    val_loss = evaluate(...)
    scheduler.step()              # ← ម្ដងក្នុងមួយ epoch បន្ទាប់ពី optimizer
```

លើកលែងតែ `OneCycleLR` ដែលហៅ **រាល់ step** ជំនួសរាល់ epoch។

### ៦. Early stopping និង best-weight tracking

```python
best_val_loss = float("inf")
best_state = None
bad_epochs = 0

if val_loss < best_val_loss:
    best_val_loss = val_loss
    # ⚠️ clone ជាន់គ្នា — state_dict() ចែក tensor ជាមួយ model
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
else:
    bad_epochs += 1

if bad_epochs >= patience:
    break
```

````{admonition} state_dict() ចែក tensor — ត្រូវ clone
:class: danger
```python
best_state = model.state_dict()      # ❌ គ្រាន់តែជា reference!
```

`state_dict()` ត្រឡប់ dict ដែល **ចង្អុលទៅ tensor ដដែល** នៅក្នុង model។
នៅពេលអ្នកបណ្ដុះបណ្ដាលបន្ត tensor ទាំងនោះផ្លាស់ប្ដូរ ហើយ "best state"
របស់អ្នកក៏ផ្លាស់ប្ដូរតាមដែរ។ អ្នកនឹងស្ដារ weight នៃ epoch **ចុងក្រោយ**
ជំនួស epoch **ល្អបំផុត**។

```python
best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}   # ✅
```

ឬល្អជាង រក្សាទុកទៅ disk ភ្លាមៗ៖

```python
torch.save(model.state_dict(), "best.pt")
```
````

### ៧. ការកត់ត្រា metrics

```python
total_loss += loss.item() * xb.size(0)     # គុណនឹងទំហំ batch
total_samples += xb.size(0)
epoch_loss = total_loss / total_samples    # មធ្យមភាពត្រឹមត្រូវ
```

````{admonition} កុំគណនាមធ្យមភាពនៃមធ្យមភាព batch
:class: warning
```python
# ❌ ខុស — batch ចុងក្រោយអាចតូចជាង
epoch_loss = sum(batch_losses) / len(batch_losses)

# ✅ ត្រូវ — ថ្លឹងតាមចំនួនសំណាក
epoch_loss = sum(loss_i * n_i) / sum(n_i)
```

កំហុសនេះលេចឡើងពេល `drop_last=False` ហើយ batch ចុងក្រោយមានសំណាកតិច។
````

## កំហុសទូទៅក្នុង Training Loop

```{literalinclude} examples/training_loop_2.py
:language: python
:caption: training_loop_2.py — ✅ បានសាកល្បងលើ CPU (បង្ហាញកំហុសទាំង ៥ ដោយចេតនា)
```

```{list-table}
:header-rows: 1
:widths: 30 34 36

* - កំហុស
  - រោគសញ្ញា
  - ដំណោះស្រាយ
* - ភ្លេច `model.eval()` ពេលវាយតម្លៃ
  - metric ប្រែប្រួលរាល់ការរត់លើទិន្នន័យដដែល
  - ហៅ `model.eval()` មុន validation loop
* - ភ្លេច `torch.no_grad()`
  - memory កើនឡើងជានិច្ច ការវាយតម្លៃយឺត
  - ប្រើ `@torch.no_grad()` decorator
* - ភ្លេច `optimizer.zero_grad()`
  - loss លោត ឬមិនថយចុះ
  - បន្ថែមជាជំហានទី ១ ក្នុង iteration
* - រក្សាទុក loss tensor ជំនួស `.item()`
  - memory leak ក្នុង training loop
  - `history.append(loss.item())`
* - មធ្យមភាពនៃមធ្យមភាព batch
  - metric ខុសបន្តិចបន្តួច
  - ថ្លឹងតាមចំនួនសំណាក
* - `scheduler.step()` មុន `optimizer.step()`
  - UserWarning + lr curve ខុស
  - ហៅបន្ទាប់ពី epoch ទាំងមូល
* - ភ្លេចផ្លាស់ទី batch ទៅ device
  - `Expected all tensors to be on the same device`
  - `xb = xb.to(device, non_blocking=True)`
* - `state_dict()` ដោយគ្មាន clone
  - "best model" គឺជា model ចុងក្រោយ
  - `detach().clone()` រាល់ tensor
```

## Checklist ពេលម៉ូដែលមិនរៀន

បើ loss មិនថយចុះ សូមដើរតាមលំដាប់នេះ៖

```{list-table}
:header-rows: 1
:widths: 6 40 54

* - #
  - ពិនិត្យ
  - របៀប
* - ១
  - តើវដ្ដ ៥ ជំហានពេញលេញទេ?
  - `zero_grad` → forward → loss → `backward` → `step`
* - ២
  - តើ gradient ហូរទេ?
  - `sum(p.grad.norm() for p in model.parameters() if p.grad is not None)`
* - ៣
  - តើ optimizer មាន parameter ទាំងអស់ទេ?
  - `sum(len(g['params']) for g in optimizer.param_groups)`
* - ៤
  - តើ loss function ត្រឹមត្រូវទេ?
  - CrossEntropyLoss ទទួល **logits** + `int64` index
* - ៥
  - តើ label ត្រឹមត្រូវទេ?
  - ពិនិត្យសំណាកខ្លះដោយភ្នែក។ `labels.min()`, `labels.max()` ត្រូវនៅក្នុង `[0, C-1]`
* - ៦
  - តើ lr សមហេតុផលទេ?
  - សាកល្បង ១០× តូចជាង និង ១០× ធំជាង
* - ៧
  - តើម៉ូដែលអាច overfit batch តូចមួយទេ?
  - បណ្ដុះបណ្ដាលលើសំណាក ១៦ ដោយគ្មាន shuffle។ loss គួរទៅជិត ០
* - ៨
  - តើទិន្នន័យបាន normalize ទេ?
  - input ដែលមានវិសាលភាពធំ (ឧ. 0–255) ធ្វើឱ្យរៀនយឺតខ្លាំង
* - ៩
  - តើ `model.train()` ត្រូវបានហៅទេ?
  - BatchNorm ក្នុង eval mode ជាមួយ batch ដំបូង មាន running stats = 0/1
* - ១០
  - តើមាន NaN ទេ?
  - `torch.isnan(loss)` — មើល {doc}`debugging`
```

```{admonition} ជំហានទី ៧ គឺមានអានុភាពបំផុត
:class: tip
ការសាកល្បង **overfit លើ batch តូចមួយ** បំបែកបញ្ហាជាពីរផ្នែក៖

- បើម៉ូដែល **អាច** overfit → ម៉ូដែល និង optimizer ត្រឹមត្រូវ។ បញ្ហានៅក្នុងទិន្នន័យ
  ឬ regularization (lr, augmentation, dropout ខ្លាំងពេក)។
- បើម៉ូដែល **មិនអាច** overfit → បញ្ហានៅក្នុងកូដ (gradient មិនហូរ, loss ខុស,
  optimizer មិនមាន parameter)។
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **វដ្ដ ៥ ជំហាន** ជានិច្ច៖ `zero_grad` → forward → loss → `backward` → `step`។
2. **`model.train()` មុនបណ្ដុះបណ្ដាល · `model.eval()` + `no_grad()` មុនវាយតម្លៃ**។
3. **`zero_grad(set_to_none=True)`** សន្សំ memory។
4. **ថ្លឹង metric តាមចំនួនសំណាក** មិនមែនមធ្យមភាពនៃមធ្យមភាព batch។
5. **Clone `state_dict()`** មុនរក្សាទុកជា "best"។
6. **`scheduler.step()` បន្ទាប់ពី epoch** លើកលែង OneCycleLR។
7. **សាកល្បង overfit លើ batch តូច** ជាជំហាន debug ដំបូង។
```

## មេរៀនបន្ទាប់

{doc}`evaluation-and-inference` — វាយតម្លៃម៉ូដែលដែលបានបណ្ដុះបណ្ដាល និងដាក់វាឱ្យប្រើប្រាស់។
