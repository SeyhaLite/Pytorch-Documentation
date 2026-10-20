# Checkpoints

**Checkpoint** ខុសពី model weights។ Model weights គឺជាអ្វីដែលអ្នកត្រូវការសម្រាប់
**inference**។ Checkpoint គឺជា **ស្ថានភាពបណ្ដុះបណ្ដាលទាំងមូល** ដែលអ្នកត្រូវការ
ដើម្បី **បន្តពីកន្លែងដែលអ្នកឈប់**។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ២ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## ហេតុអ្វីត្រូវការ optimizer state?

Optimizer ភាគច្រើនរក្សាទុក **ស្ថានភាពខាងក្នុង** ដែលបានរៀនតាមពេលវេលា៖

```{list-table}
:header-rows: 1
:widths: 24 76

* - Optimizer
  - ស្ថានភាពដែលរក្សាទុក
* - `SGD` (គ្មាន momentum)
  - គ្មាន
* - `SGD(momentum=0.9)`
  - `momentum_buffer` សម្រាប់ parameter នីមួយៗ
* - `Adam` / `AdamW`
  - `exp_avg` (momentum ទី ១) និង `exp_avg_sq` (momentum ទី ២) + `step` count សម្រាប់ bias correction
* - `RMSprop`
  - `square_avg`
* - `Adagrad`
  - `sum` នៃ squared gradients
```

បើអ្នកស្ដារតែ weight ដោយមិនស្ដារ optimizer state ទេ៖
- Adam ចាប់ផ្ដើម moments ពីសូន្យ → **ជំហានដំបូងៗមានទំហំខុសគ្នាខ្លាំង**
- bias correction count ចាប់ពី 0 → effective learning rate ខុស
- ការបណ្ដុះបណ្ដាលអាច diverge ឬយឺតជាងការចាប់ផ្ដើមពីដើម

## ឧទាហរណ៍ពេញលេញ

```{literalinclude} examples/checkpoints_1.py
:language: python
:caption: checkpoints_1.py — ✅ បានសាកល្បងលើ CPU (save → resume → verify optimizer state)
```

### អ្វីដែលត្រូវរក្សាទុក

```python
def save_checkpoint(path, *, epoch, model, optimizer, scheduler=None,
                    scaler=None, best_metric=float("inf"), rng_state=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "torch_version": torch.__version__,      # ← សំខាន់សម្រាប់ debugging
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:                        # mixed precision
        payload["scaler_state_dict"] = scaler.state_dict()
    if rng_state is not None:
        payload["rng_state"] = rng_state
    torch.save(payload, path)
```

```{list-table} ធាតុដែលគួររក្សាទុក
:header-rows: 1
:widths: 28 72

* - ធាតុ
  - ហេតុអ្វី
* - `epoch`
  - ដឹងថាត្រូវបន្តពីណា
* - `model_state_dict`
  - weight
* - `optimizer_state_dict`
  - moments / buffers (មើលខាងលើ)
* - `scheduler_state_dict`
  - ដំណាក់កាលនៃ learning rate curve
* - `scaler_state_dict`
  - GradScaler scale factor (mixed precision)
* - RNG state
  - ដើម្បីឱ្យការបន្តដូចគ្នានឹងការមិនបានឈប់
* - `best_metric`
  - ដើម្បីបន្ត early stopping ត្រឹមត្រូវ
* - `torch_version`
  - ជួយធ្វើរោគវិនិច្ឆ័យពេលផ្ទុកលើកំណែផ្សេង
```

### ការស្ដារ

```python
model = MyModel()                                  # ← architecture ថ្មី
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

ck = torch.load("epoch_2.pt", weights_only=False)  # ← មាន dict/str ដូច្នេះ False
model.load_state_dict(ck["model_state_dict"])
optimizer.load_state_dict(ck["optimizer_state_dict"])
scheduler.load_state_dict(ck["scheduler_state_dict"])
start_epoch = ck["epoch"] + 1
best_metric = ck["best_metric"]
set_rng_state(ck["rng_state"])

for epoch in range(start_epoch, total_epochs):     # ← បន្តពីកន្លែងដែលឈប់
    ...
```

````{admonition} បង្កើត optimizer ដោយ hyperparameters ដូចគ្នា
:class: important
`optimizer.load_state_dict()` ផ្ទុក **state** (moments, buffers) ប៉ុន្តែ
**hyperparameters** (lr, weight_decay, betas) មកពី `param_groups` នៅក្នុង checkpoint។

បើអ្នកបង្កើត optimizer ជាមួយ param groups ខុសគ្នា (ឧ. ចំនួន group ឬ
ចំនួន parameter មិនដូចគ្នា) ការផ្ទុកនឹងបរាជ័យ៖

```text
ValueError: loaded state dict has a different number of parameter groups
```
````

## RNG State

ដើម្បីឱ្យការបន្តដូចគ្នាពិតប្រាកដទៅនឹងការមិនបានឈប់ អ្នកត្រូវចាប់យក
ស្ថានភាព RNG ទាំងអស់៖

```python
def get_rng_state() -> dict:
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
```

````{admonition} RNG state មិនគ្រប់គ្រង Python random និង NumPy ទេ
:class: warning
`torch.get_rng_state()` ចាប់យកតែ PyTorch CPU RNG ប៉ុណ្ណោះ។ បើកូដរបស់អ្នកប្រើ
`random.random()` ឬ `np.random` (ឧទាហរណ៍ក្នុង augmentation) អ្នកត្រូវចាប់យកវាផងដែរ៖

```python
import random
import numpy as np

state = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
}
```

មើល {doc}`reproducibility` សម្រាប់ការពន្យល់ពេញលេញ។
````

## យុទ្ធសាស្ត្ររក្សាទុក

```{list-table}
:header-rows: 1
:widths: 20 26 54

* - ឯកសារ
  - ពេលសរសេរ
  - គោលបំណង
* - `last.pt`
  - រាល់ epoch (សរសេរជាន់)
  - បន្តការបណ្ដុះបណ្ដាលបន្ទាប់ពីការដាច់ចរន្តអគ្គិសនី ឬ crash
* - `best.pt`
  - ពេល metric ប្រសើរឡើង
  - ម៉ូដែលចុងក្រោយសម្រាប់ deployment
* - `epoch_N.pt`
  - រាល់ epoch
  - សម្រាប់វិភាគ curve។ រក្សាទុក N ចុងក្រោយ លុបចាស់ចេញ
```

### រក្សាទុកតែ N ចុងក្រោយ

```python
keep_last = 3
for old in sorted(CKPT_DIR.glob("epoch_*.pt"))[:-keep_last]:
    old.unlink()
```

### កាត់បន្ថយទំហំ

```{list-table}
:header-rows: 1
:widths: 40 60

* - យុទ្ធសាស្ត្រ
  - ការពន្យល់
* - រក្សាទុក optimizer state តែក្នុង `last.pt`
  - `best.pt` ត្រូវការតែ weight សម្រាប់ inference
* - រក្សាទុក weight ជា `bfloat16`
  - កាត់ពាក់កណ្ដាល។ ប៉ះពាល់ដល់ភាពជាក់លាក់បន្តិចបន្តួច
* - លុប checkpoint ចាស់
  - រក្សាទុក 3–5 ចុងក្រោយ
* - ប្រើ `safetensors` សម្រាប់ weight
  - តូចជាង និងលឿនជាងក្នុងការផ្ទុក
```

## កំហុសទូទៅ

```{literalinclude} examples/checkpoints_2.py
:language: python
:caption: checkpoints_2.py — ✅ បានសាកល្បងលើ CPU (កំហុសទាំង ៦ ជាមួយដំណោះស្រាយ)
```

```{list-table}
:header-rows: 1
:widths: 32 30 38

* - សារកំហុស / រោគសញ្ញា
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `Missing key(s) in state_dict: "module.fc1.weight"`
  - រក្សាទុកពី `DataParallel` / DDP
  - រក្សាទុក `model.module.state_dict()`
* - `loaded state dict has a different number of parameter groups`
  - optimizer ត្រូវបានបង្កើតខុស
  - ប្រើ hyperparameters និង param groups ដូចគ្នា
* - `size mismatch for ...`
  - architecture ផ្លាស់ប្ដូរ
  - រក្សាទុក `hparams` ក្នុង checkpoint ហើយផ្ទៀងផ្ទាត់មុនផ្ទុក
* - metric អាក្រក់បន្ទាប់ពីផ្ទុក checkpoint
  - ភ្លេច `model.eval()`
  - `load_state_dict` មិនកែ `.training` flag ទេ
* - `UnpicklingError` / ឯកសារខូច
  - ការសរសេរត្រូវបានកាត់ផ្ដាច់
  - Atomic write: សរសេរ `.tmp` បន្ទាប់មក `os.replace`
* - loss លោតខ្លាំងបន្ទាប់ពី resume
  - ភ្លេចស្ដារ optimizer ឬ scheduler state
  - ស្ដារទាំងអស់ — មើលធាតុដែលគួររក្សាទុកខាងលើ
* - Disk ពេញ
  - រក្សាទុក optimizer state ក្នុងគ្រប់ epoch
  - រក្សាទុក optimizer តែក្នុង `last.pt`
```

### DataParallel prefix

````{admonition} ការពារជាងព្យាបាល
:class: tip
ប្រសិនបើអ្នកប្រើ `nn.DataParallel` ឬ `DistributedDataParallel` សរសេរ៖

```python
raw_model = model.module if hasattr(model, "module") else model
torch.save(raw_model.state_dict(), path)
```

បើអ្នករក្សាទុក `model.state_dict()` ដោយផ្ទាល់ key ទាំងអស់នឹងមាន prefix
`module.` ហើយការផ្ទុកទៅ model ធម្មតានឹងបរាជ័យ។

បើអ្នកត្រូវការសម្អាត checkpoint ដែលមាន prefix រួចហើយ៖

```python
from collections import OrderedDict
cleaned = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
```
````

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **Checkpoint = model + optimizer + scheduler + epoch + RNG + metric**។
2. **Optimizer state សំខាន់ណាស់** — បើមិនដូច្នោះទេ Adam ចាប់ផ្ដើម moments ពីសូន្យ។
3. **`model.module.state_dict()`** បើអ្នកប្រើ DataParallel/DDP។
4. **ចាប់យក RNG state ទាំងអស់** (Python, NumPy, torch, cuda) សម្រាប់ការបន្តពិតប្រាកដ។
5. **Atomic write** (`os.replace`) ដើម្បីការពារឯកសារខូច។
6. **រក្សាទុក `torch_version` និោោង hyperparameters** សម្រាប់ធ្វើរោគវិនិច្ឆ័យនាពេលអនាគត។
7. **បំបែក `best.pt` (weight តែប៉ុណ្ណោះ) និង `last.pt` (ពេញលេញ)** ដើម្បីសន្សំ disk។
```

## មេរៀនបន្ទាប់

{doc}`reproducibility` — ធ្វើឱ្យការពិសោធន៍របស់អ្នកអាចធ្វើឡើងវិញបាន។
