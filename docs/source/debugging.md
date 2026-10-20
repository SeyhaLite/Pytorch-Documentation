# Debugging និង Common Errors

ទំព័រនេះជាការប្រមូលផ្ដុំ **កំហុស PyTorch ដែលជួបញឹកញាប់បំផុត** ជាមួយសារកំហុសពិត
និងដំណោះស្រាយ។ សូមប្រើវាជាឯកសារយោងពេលអ្នកជួបបញ្ហា។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ៣ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
`debugging_1.py` បង្កើតកំហុស **ដោយចេតនា** ក្នុង `try/except` ដើម្បីបង្ហាញសារពិត។
```

## ១. សារកំហុសពិតពី PyTorch

```{literalinclude} examples/debugging_1.py
:language: python
:caption: debugging_1.py — ✅ បានសាកល្បងលើ CPU (កំហុស ២០+ ត្រូវបានបង្កើត និងចាប់យក)
```

### តារាងស្វែងរកតាមសារកំហុស

```{list-table}
:header-rows: 1
:widths: 40 60

* - សារកំហុស (ឬផ្នែកមួយនៃវា)
  - ទំព័រពាក់ព័ន្ធ
* - `mat1 and mat2 shapes cannot be multiplied`
  - {doc}`shapes-and-broadcasting` · {doc}`neural-networks`
* - `The size of tensor a (X) must match the size of tensor b (Y)`
  - {doc}`shapes-and-broadcasting`
* - `view size is not compatible with input tensor's size and stride`
  - {doc}`tensors`
* - `expected scalar type Long but found Float`
  - {doc}`loss-and-optimizers` · {doc}`neural-networks`
* - `Expected all tensors to be on the same device`
  - {doc}`devices`
* - `grad can be implicitly created only for scalar outputs`
  - {doc}`autograd`
* - `Trying to backward through the graph a second time`
  - {doc}`autograd`
* - `a leaf Variable that requires grad is being used in an in-place operation`
  - {doc}`autograd`
* - `one of the variables needed for gradient computation has been modified by an inplace operation`
  - {doc}`autograd` · {doc}`tensors`
* - `cannot assign module before Module.__init__() call`
  - {doc}`neural-networks`
* - `optimizer got an empty parameter list`
  - {doc}`neural-networks` · {doc}`loss-and-optimizers`
* - `Error(s) in loading state_dict`
  - {doc}`saving-and-loading` · {doc}`checkpoints`
* - `Attempting to deserialize object on a CUDA device`
  - {doc}`saving-and-loading` · {doc}`devices`
* - `stack expects each tensor to be equal size`
  - {doc}`data-and-dataloader`
* - `DataLoader worker (pid X) is killed by signal`
  - {doc}`data-and-dataloader`
* - `ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics'`
  - {doc}`loss-and-optimizers`
* - `CUDA out of memory`
  - {doc}`performance` · {doc}`mixed-precision`
* - `too many dimensions 'str'`
  - {doc}`saving-and-loading` · {doc}`tensors`
* - `non-empty 3D or 4D (batch mode) tensor expected`
  - {doc}`neural-networks`
* - `Inference tensors cannot be saved for backward`
  - {doc}`evaluation-and-inference`
```

## ២. ការធ្វើរោគវិនិច្ឆ័យ NaN និង Inf

```{literalinclude} examples/debugging_2.py
:language: python
:caption: debugging_2.py — ✅ បានសាកល្បងលើ CPU (gradient health, hooks, anomaly detection)
```

### មូលហេតុទូទៅនៃ NaN

```{list-table}
:header-rows: 1
:widths: 34 66

* - មូលហេតុ
  - ការពន្យល់ និងដំណោះស្រាយ
* - Learning rate ធំពេក
  - Weight ផ្ទុះ ហើយ activation ទៅជា inf។ បន្ថយ lr ១០×
* - `log(0)` ឬ `log(negative)`
  - បន្ថែម epsilon៖ `torch.log(x + 1e-8)`
* - ចែកនឹងសូន្យ
  - `x / y.clamp(min=1e-8)` ឬ `torch.where(y == 0, ...)`
* - `exp()` overflow
  - ប្រើ log-space។ `logsumexp` ជំនួស `exp().sum()`
* - Softmax ដោយដៃលើ logits ធំ
  - ប្រើ `torch.softmax()` ឬ `F.log_softmax()` ដែលមានស្ថិរភាពលេខ
* - ទិន្នន័យមាន NaN ស្រាប់
  - ត្រួតពិនិត្យ៖ `torch.isnan(dataset_tensor).any()`
* - fp16 overflow ដោយគ្មាន GradScaler
  - មើល {doc}`mixed-precision`
* - Gradient explosion
  - `clip_grad_norm_(model.parameters(), 1.0)`
* - Weight initialization ខុស
  - មើល {doc}`neural-networks` ផ្នែក initialization
```

### ឧបករណ៍ត្រួតពិនិត្យ

បន្ថែមទៅក្នុង training loop របស់អ្នក៖

```python
def check_tensor(name: str, t: torch.Tensor) -> bool:
    has_nan = bool(torch.isnan(t).any())
    has_inf = bool(torch.isinf(t).any())
    if has_nan or has_inf:
        finite = t[torch.isfinite(t)]
        print(f"✗ {name}: NaN={has_nan} Inf={has_inf} "
              f"min={finite.min().item() if finite.numel() else float('nan'):.3e} "
              f"max={finite.max().item() if finite.numel() else float('nan'):.3e}")
        return False
    return True
```

ប្រើវានៅកន្លែងសំខាន់ៗ៖

```python
for step, (xb, yb) in enumerate(loader):
    check_tensor("input", xb)
    logits = model(xb)
    check_tensor("logits", logits)
    loss = loss_fn(logits, yb)
    if not check_tensor("loss", loss):
        print(f"NaN នៅ step {step}")
        break
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    check_tensor("grad_norm", grad_norm)
    optimizer.step()
```

### torch.autograd.set_detect_anomaly()

```python
with torch.autograd.set_detect_anomaly(True):
    loss = compute_loss(...)
    loss.backward()          # បើ NaN កើតឡើង នឹងបង្ហាញ forward op ដែលបង្កើតវា
```

```{admonition} detect_anomaly ធ្វើឱ្យយឺតខ្លាំង
:class: warning
វាត្រួតពិនិត្យរាល់ backward operation ហើយកត់ត្រា stack trace។ ប្រើ **តែពេល debug**
ហើយបិទវានៅក្នុងការរត់ធម្មតា។ កុំទុកវាបើកក្នុង production។
```

## ៣. Gradient Health Checks

### Gradient មានបញ្ហាបែបណា?

```{list-table}
:header-rows: 1
:widths: 26 32 42

* - រោគសញ្ញា
  - ឈ្មោះ
  - ដំណោះស្រាយ
* - `.grad is None`
  - Parameter មិនជាប់ graph
  - ពិនិត្យថាវាត្រូវបានប្រើក្នុង `forward()`។ ពិនិត្យ `ModuleList`
* - `grad == 0` គ្រប់ទីកន្លែង
  - Gradient មិនហូរ
  - ភ្លេច `loss.backward()` ឬ `.detach()` នៅកន្លែងខុស
* - `grad` ធំខ្លាំង (> 1e3)
  - Exploding gradients
  - `clip_grad_norm_`, បន្ថយ lr, បន្ថែម normalization
* - `grad` តូចខ្លាំង (< 1e-7)
  - Vanishing gradients
  - ReLU ជំនួស sigmoid/tanh, residual connections, LayerNorm
* - `grad` មាន NaN
  - Numerical instability
  - មើលផ្នែក NaN ខាងលើ
* - `grad` ដូចគ្នារាល់ step
  - Gradient ប្រមូលផ្ដុំ ឬមិនបាន zero
  - ពិនិត្យ `optimizer.zero_grad()`
```

### ឧបករណ៍វិភាគ gradient

```python
def report_gradients(model):
    stats = {"n": 0, "zero": 0, "nan": 0, "exploded": 0, "norms": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"  ⚠ {name}: grad is None")
            continue
        g = p.grad
        norm = g.norm().item()
        stats["n"] += 1
        stats["norms"].append(norm)
        if norm == 0:
            stats["zero"] += 1
        if torch.isnan(g).any():
            stats["nan"] += 1
        if norm > 1e3:
            stats["exploded"] += 1
    if stats["norms"]:
        print(f"  norm range: {min(stats['norms']):.3e} .. {max(stats['norms']):.3e}")
    print(f"  params={stats['n']} zero={stats['zero']} "
          f"nan={stats['nan']} exploded={stats['exploded']}")
    return stats
```

## ៤. Forward Hooks ដើម្បីតាមដាន Activation

នេះគឺជាឧបករណ៍មានអានុភាពបំផុតសម្រាប់ស្វែងរកថាតើជាន់ណាបង្កបញ្ហា —
**ដោយមិនកែ `forward()`**៖

```python
activations = {}


def register_stats_hook(name):
    def hook(module, inputs, output):
        if torch.is_tensor(output):
            activations[name] = {
                "shape": tuple(output.shape),
                "mean": output.mean().item(),
                "std": output.std().item(),
                "dead_pct": float((output == 0).float().mean().item() * 100),
            }
    return hook


handles = []
for name, module in model.named_modules():
    if name and not list(module.children()):      # leaf modules តែប៉ុណ្ណោះ
        handles.append(module.register_forward_hook(register_stats_hook(name)))

with torch.no_grad():
    model(sample_batch)

for h in handles:
    h.remove()                                    # ⚠️ ចាំបាច់ — បើមិនដូច្នោះទេ memory leak
```

### ការបកស្រាយលទ្ធផល

```{list-table}
:header-rows: 1
:widths: 34 66

* - សញ្ញា
  - អត្ថន័យ
* - `std` តូចខ្លាំង (< 0.01) ហើយថយចុះតាមជាន់
  - Vanishing activations។ ពិនិត្យ initialization និង normalization
* - `std` ធំខ្លាំង (> 10) ហើយកើនឡើងតាមជាន់
  - Exploding activations។ បន្ថែម normalization ឬ residual connections
* - `dead_pct` ខ្ពស់ (> 50%) ជាមួយ ReLU
  - **Dying ReLU** — neuron ស្លាប់។ ពិចារណា LeakyReLU, GELU ឬបន្ថយ lr
* - `mean` ឆ្ងាយពី 0 ហើយកើនឡើងតាមជាន់
  - Internal covariate shift។ បន្ថែម BatchNorm/LayerNorm
* - `shape` មិនដូចដែលអ្នករំពឹង
  - កំហុស shape។ មើល {doc}`shapes-and-broadcasting`
```

## ៥. Backward Hooks

សម្រាប់តាមដាន gradient ជាក់លាក់ជាន់៖

```python
def grad_hook(name):
    def hook(grad):
        print(f"{name}: grad norm = {grad.norm().item():.3e}")
        return grad                     # អាចកែ gradient បានផង
    return hook

model.fc1.weight.register_hook(grad_hook("fc1.weight"))
```

ឬសម្រាប់ module ទាំងមូល៖

```python
model.fc1.register_full_backward_hook(
    lambda m, gin, gout: print(f"{m}: in-grad norm = {gin[0].norm().item():.3e}")
)
```

```{admonition} កុំកែ tensor ក្នុង hook ដោយ in-place
:class: danger
Hook អាចត្រឡប់ gradient ដែលបានកែ ប៉ុន្តែការកែដោយ in-place
(ឧ. `grad.mul_(0.5)`) នឹងបង្ក error។ ត្រឡប់ tensor ថ្មីវិញ៖ `return grad * 0.5`។
```

## ៦. Debugging Workflow ដែលណែនាំ

```{list-table}
:header-rows: 1
:widths: 8 34 58

* - ជំហាន
  - សកម្មភាព
  - ហេតុអ្វី
* - ១
  - ធ្វើឱ្យបញ្ហាតូចបំផុត
  - ទិន្នន័យ ៣២ សំណាក, model តូច, គ្មាន augmentation, គ្មាន regularization
* - ២
  - តើវា overfit batch តូចបានទេ?
  - បើទេ → បញ្ហានៅក្នុងកូដ (gradient, loss, optimizer)។ បើបាន → បញ្ហានៅក្នុងទិន្នន័យ ឬ hyperparameters
* - ៣
  - ពិនិត្យ gradient flow
  - `report_gradients()` ខាងលើ
* - ៤
  - ពិនិត្យ activation
  - Forward hooks
* - ៥
  - ពិនិត្យទិន្នន័យដោយភ្នែក
  - បង្ហាញសំណាកខ្លះ + label របស់វា។ តើ label ត្រូវទេ? តើ normalization ត្រូវទេ?
* - ៦
  - ពិនិត្យ loss ដំបូង
  - សម្រាប់ classification ជាមួយ C ថ្នាក់ loss ដំបូងគួរ ≈ `ln(C)`។ បើខុសខ្លាំង → loss function ឬ label ខុស
* - ៧
  - សាកល្បង lr ច្រើនតម្លៃ
  - ១០× តូចជាង និង ១០× ធំជាង
* - ៨
  - បើនៅតែមិនដោះស្រាយបាន
  - ប្រើ `torch.autograd.set_detect_anomaly(True)` និង `gradcheck`
```

### ត្រួតពិនិត្យ loss ដំបូង

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - បញ្ហា
  - Loss ដំបូងរំពឹង
  - បើខុសខ្លាំង
* - Multi-class, C ថ្នាក់
  - `ln(C)` (ឧ. C=10 → 2.30)
  - label ខុស ឬ loss function ខុស
* - Binary
  - `ln(2)` ≈ 0.693
  - ដូចគ្នា
* - MSE លើទិន្នន័យ normalized
  - ≈ 1.0
  - ទិន្នន័យមិនបាន normalize ឬ shape ខុស
```

## ៧. ឧបករណ៍ Debug ជាក់ស្ដែង

```{list-table}
:header-rows: 1
:widths: 26 74

* - ឧបករណ៍
  - ការប្រើប្រាស់
* - `python -m pdb script.py`
  - Debugger ស្តង់ដារ។ `b 42` សម្រាប់ breakpoint, `p x.shape` សម្រាប់ពិនិត្យ
* - VS Code debugger
  - Breakpoints + variable inspection។ ល្អបំផុតសម្រាប់ PyTorch
* - `torch.set_printoptions(precision=3, sci_mode=False)`
  - ធ្វើឱ្យ tensor ងាយអាន
* - `print(model)`
  - មើល architecture និងឈ្មោះជាន់
* - `for n, p in model.named_parameters()`
  - មើលឈ្មោះ + shape នៃ parameter ទាំងអស់
* - `torch.fx.symbolic_trace(model)`
  - មើល computation graph ជាអត្ថបទ
* - `torch.profiler`
  - រក bottleneck ល្បឿន។ មើល {doc}`performance`
* - `torch.autograd.gradcheck`
  - ផ្ទៀងផ្ទាត់ gradient ផ្ទាល់ខ្លួន (ត្រូវការ float64)
* - `TORCH_SHOW_CPP_STACKTRACES=1`
  - Environment variable ដែលបង្ហាញ C++ stack trace ក្នុង error
* - `CUDA_LAUNCH_BLOCKING=1`
  - ធ្វើឱ្យ CUDA synchronous ដើម្បីឱ្យ error ចង្អុលទៅកន្លែងត្រឹមត្រូវ
```

````{admonition} CUDA_LAUNCH_BLOCKING=1 ពេល error GPU ចង្អុលទៅកន្លែងខុស
:class: tip
ដោយសារ CUDA asynchronous សារ error ជាញឹកញាប់ចង្អុលទៅបន្ទាត់ **ក្រោយ** កន្លែង
ដែលកំហុសពិតបានកើតឡើង។ រត់ជាមួយ៖

```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
```

វាយឺតជាង ប៉ុន្តែ traceback នឹងត្រឹមត្រូវ។
````

## ៨. ឧទាហរណ៍នៃ Traceback ពេញលេញ

```{literalinclude} examples/debugging_3.py
:language: python
:caption: debugging_3.py — ✅ បានសាកល្បង (សម្គាល់ជា `# EXPECTED FAILURE` — test runner ផ្ទៀងផ្ទាត់ថាវាបរាជ័យដោយសារមូលហេតុត្រឹមត្រូវ)
```

ឯកសារនេះ **មិនបានចាប់យកកំហុសទេ** — វាបញ្ចប់ដោយ traceback ពិត។ នេះជាអ្វីដែល
អ្នកនឹងឃើញនៅក្នុង terminal៖

```text
Traceback (most recent call last):
  File ".../debugging_3.py", line 24, in <module>
    out = model(batch)
  File ".../torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File ".../torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File ".../torch/nn/modules/linear.py", line 138, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x12 and 10x32)
```

### របៀបអាន Traceback

```{list-table}
:header-rows: 1
:widths: 30 70

* - ផ្នែក
  - អ្វីដែលត្រូវមើល
* - បន្ទាត់ដំបូង `Traceback (most recent call last):`
  - ការហៅចាប់ផ្ដើមពីខាងលើ ហើយកំហុសនៅខាងក្រោម
* - បន្ទាត់ដែលមានឯកសារ **របស់អ្នក**
  - នេះជាកន្លែងដែលអ្នកត្រូវកែ។ បន្ទាត់ដែលចង្អុលទៅ `torch/nn/...` គ្រាន់តែជាផ្លូវឆ្លងកាត់
* - បន្ទាត់ចុងក្រោយ (`RuntimeError: ...`)
  - សារកំហុសពិត។ នៅទីនេះ `(8x12 and 10x32)` ប្រាប់អ្នកពិតប្រាកដថា shape អ្វីខុសគ្នា
```

ក្នុងឧទាហរណ៍ខាងលើ `(8x12 and 10x32)` មានន័យថា៖ input មាន 12 features
ប៉ុន្តែ `nn.Linear(10, 32)` រំពឹង 10។ ដំណោះស្រាយ៖ កែ `nn.Linear(12, 32)`
ឬកែទិន្នន័យឱ្យមាន 10 features។

## ៩. Reproducing bugs

បើអ្នកត្រូវរាយការណ៍ bug ទៅ PyTorch សូមផ្ដល់៖

```python
import platform
import sys

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Python version : {sys.version.split()[0]}")
print(f"OS             : {platform.system()} {platform.release()}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version   : {torch.version.cuda}")
    print(f"cuDNN version  : {torch.backends.cudnn.version()}")
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
```

រួមទាំង **កូដតូចបំផុតដែលបង្កើតកំហុស** (minimal reproducible example)។
Bug report ដែលមាន MRE ត្រូវបានដោះស្រាយលឿនជាងច្រើនដង។

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **អានបន្ទាត់ចុងក្រោយនៃ traceback មុន** — វាប្រាប់ shape ជាក់លាក់ដែលខុសគ្នា។
2. **ធ្វើឱ្យបញ្ហាតូចបំផុត** — សាកល្បង overfit លើ batch តូចមួយ។
3. **ត្រួតពិនិត្យ NaN ជាប្រចាំ** ក្នុង training loop។
4. **Forward hooks** ជាឧបករណ៍ល្អបំផុតសម្រាប់ស្វែងរកជាន់ដែលមានបញ្ហា។
5. **`report_gradients()`** ដើម្បីរក vanishing/exploding gradients។
6. **`CUDA_LAUNCH_BLOCKING=1`** ពេល GPU error ចង្អុលទៅកន្លែងខុស។
7. **ត្រួតពិនិត្យ loss ដំបូង** — គួរ ≈ `ln(C)` សម្រាប់ classification។
8. **បើក `detect_anomaly` តែពេល debug** — វាយឺតខ្លាំង។
```
