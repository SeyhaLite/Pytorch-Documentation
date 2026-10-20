# Mixed Precision Training

**Mixed precision** គឺជាការប្រើប្រាស់ទាំង float32 និង float16/bfloat16 ក្នុងពេលតែមួយ៖
ប្រតិបត្តិការធំៗ (matmul, convolution) ក្នុង precision ទាបដើម្បីល្បឿន និង memory
ហើយប្រតិបត្តិការរំញោច (loss, softmax, normalization) ក្នុង float32 ដើម្បីស្ថិរភាព។

```{admonition} តម្រូវការ
:class: note
🖥️ `mixed_precision_1.py` ដំណើរការលើ CPU (ប្រើ `device_type="cpu"` ជាមួយ bfloat16)។
🔒 `mixed_precision_2.py` ត្រូវការ CUDA GPU។
```

## អត្ថប្រយោជន៍

```{list-table}
:header-rows: 1
:widths: 28 72

* - អត្ថប្រយោជន៍
  - ការពន្យល់
* - **លឿនជាង**
  - GPU មាន tensor cores ដែលធ្វើ fp16/bf16 matmul លឿនជាង float32 ច្រើនដង
* - **Memory តិចជាង**
  - Activation ទំហំពាក់កណ្ដាល → batch ធំជាងបាន ឬ model ធំជាងបាន
* - **ភាពជាក់លាក់ប្រហាក់ប្រហែលដដែល**
  - Weight master copy នៅតែ float32 ដូច្នេះ gradient ត្រូវបានប្រមូលផ្ដុំដោយជាក់លាក់
```

```{admonition} ឯកសារនេះមិនអះអាងលេខល្បឿនសកលណាមួយទេ
:class: warning
អត្ថប្រយោជន៍ពិតអាស្រ័យខ្លាំងលើ GPU, model architecture, batch size និង PyTorch version។
លើ **CPU** mixed precision អាចមិនជួយអ្វីសោះ ឬអាចធ្វើឱ្យយឺតជាង បើ CPU មិនមាន
AVX512-BF16។

ឧទាហរណ៍ `mixed_precision_1.py` វាស់ល្បឿនលើម៉ាស៊ីនដែលរត់វា ហើយបោះពុម្ពលទ្ធផល។
សូមរត់វាដោយខ្លួនឯងដើម្បីទទួលបានលេខសម្រាប់ hardware របស់អ្នក។
```

## float16 vs bfloat16

```{literalinclude} examples/mixed_precision_1.py
:language: python
:caption: mixed_precision_1.py — ✅ បានសាកល្បងលើ CPU (device_type="cpu", dtype=bfloat16)
```

### ភាពខុសគ្នាសំខាន់

```{list-table}
:header-rows: 1
:widths: 20 22 22 36

* - លក្ខណៈ
  - `float32`
  - `float16`
  - `bfloat16`
* - Bits សរុប
  - 32
  - 16
  - 16
* - Exponent bits
  - 8
  - 5
  - 8
* - Mantissa (ភាពជាក់លាក់) bits
  - 23
  - 10
  - 7
* - តម្លៃអតិបរមា
  - ~3.4e38
  - **~65504**
  - ~3.4e38
* - តម្លៃតូចបំផុត > 0
  - ~1.2e-38
  - ~6.1e-5
  - ~1.2e-38
* - ត្រូវការ `GradScaler`?
  - ទេ
  - **បាទ/ចាស**
  - ទេ
* - Hardware
  - គ្រប់ទីកន្លែង
  - GPU ទាំងអស់
  - GPU Ampere+ (A100, RTX 30/40), TPU, CPU ដែលមាន AVX512-BF16
```

### ហេតុអ្វី float16 ត្រូវការ GradScaler?

Gradient ក្នុង neural networks ជាទូទៅតូចណាស់ (ឧ. `1e-7` ទៅ `1e-3`)។
float16 មានតម្លៃតូចបំផុតដែលអាចតំណាងបានប្រហែល `6e-5` (normal) ឬ `6e-8` (subnormal)។
Gradient ដែលតូចជាងនេះក្លាយជា **សូន្យ** (underflow)។

`GradScaler` ដោះស្រាយដោយ៖

```text
1. គុណ loss ដោយ scale factor ធំ (ឧ. 65536) មុន backward()
   -> gradient កើនឡើងដូចគ្នា ដូច្នេះមិន underflow
2. ធ្វើ unscale gradient វិញមុន optimizer.step()
3. បើរកឃើញ inf/NaN ក្នុង gradient -> រំលង step នោះ ហើយបន្ថយ scale
4. បើគ្មាន inf ក្នុង interval មួយ -> បង្កើន scale
```

**bfloat16 មិនត្រូវការ GradScaler** ព្រោះវាមាន exponent bits ដូច float32
ដូច្នេះដែនតម្លៃដូចគ្នា — គ្មាន underflow ទេ។ វាគ្រាន់តែមានភាពជាក់លាក់ទាបជាង។

### ការបង្ហាញ underflow ពិតប្រាកដ

ពីឧទាហរណ៍ដែលបានសាកល្បង៖

```python
tiny = torch.tensor(1e-8, dtype=torch.float32)
tiny.to(torch.float16)     # tensor(0.)        <- underflow!
tiny.to(torch.bfloat16)    # tensor(1.0000e-08) <- នៅរក្សាបាន

big = torch.tensor(70000.0)
big.to(torch.float16)      # tensor(inf)       <- overflow!
big.to(torch.bfloat16)     # tensor(70144.)    <- នៅត្រឹមត្រូវ
```

## ការប្រើប្រាស់

### របៀបស្តង់ដារ

```python
from torch.amp import GradScaler, autocast

use_amp = torch.cuda.is_available()
device_type = "cuda" if use_amp else "cpu"
amp_dtype = torch.float16 if use_amp else torch.bfloat16

scaler = GradScaler(device=device_type, enabled=use_amp)

for xb, yb in loader:
    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type=device_type, dtype=amp_dtype):
        logits = model(xb)
        loss = loss_fn(logits, yb)

    scaler.scale(loss).backward()      # គុណ loss ឡើង បន្ទាប់មក backward
    scaler.step(optimizer)             # unscale + optimizer.step() (រំលងបើមាន inf)
    scaler.update()                    # កែតម្រូវ scale factor
```

```{admonition} autocast ប្ដូរប្រតិបត្តិការ មិនមែន parameter
:class: important
Weight របស់ model នៅតែ **float32** (master copy)។ `autocast` គ្រាន់តែប្ដូរ
ការគណនាជាក់លាក់ទៅ precision ទាបដោយស្វ័យប្រវត្តិ។

នេះមានន័យថា `optimizer.step()` ធ្វើបច្ចុប្បន្នភាព float32 weights
ហើយ gradient ត្រូវបានប្រមូលផ្ដុំក្នុង float32។ ភាពជាក់លាក់មិនបាត់បង់ទេ។
```

### អ្វីដែល autocast ប្ដូរ

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - ប្រភេទប្រតិបត្តិការ
  - Precision
  - ឧទាហរណ៍
* - ប្រតិបត្តិការធ្ងន់ ដែលស្ថិរ
  - **ទាប** (fp16/bf16)
  - `matmul`, `conv2d`, `linear`, `bmm`, `addmm`
* - Reductions
  - float32
  - `sum`, `mean`, `norm`, `prod`
* - Loss functions
  - float32
  - `cross_entropy`, `mse_loss`, `nll_loss`, `binary_cross_entropy_with_logits`
* - ប្រតិបត្តិការរំញោច
  - float32
  - `softmax`, `log_softmax`, `layer_norm`, `group_norm`, `log`, `exp`, `pow`
```

បញ្ជីពេញលេញមាននៅក្នុងឯកសារផ្លូវការ PyTorch សម្រាប់
[autocast op eligibility](https://docs.pytorch.org/docs/stable/amp.html#autocast-op-eligibility)។

## Gradient Clipping ជាមួយ GradScaler

ត្រូវ unscale **មុន** clip៖

```python
scaler.scale(loss).backward()

# ⚠️ ត្រូវ unscale ជាមុន បើមិនដូច្នោះទេ norm នឹងខុស
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

បើអ្នក clip លើ scaled gradients norm នឹងធំជាងពិត ហើយ clipping នឹង
កាត់ gradient ខ្លាំងពេក។

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស / រោគសញ្ញា
  - មូលហេតុ
  - ដំណោះស្រាយ
* - loss ក្លាយជា NaN ភ្លាមៗ
  - fp16 overflow ដោយគ្មាន GradScaler
  - ប្រើ `GradScaler(enabled=True)` ឬប្ដូរទៅ bfloat16
* - GradScaler scale ធ្លាក់ចុះដល់ 0
  - gradient មាន inf ជានិច្ច
  - ពិនិត្យ lr, ទិន្នន័យ និង model សម្រាប់ NaN ជាមុន
* - `Expected torch.float16 or torch.bfloat16 but got torch.float32`
  - ភាពមិនត្រូវគ្នានៃ dtype ក្នុង autocast region
  - ធានាថា input និង model នៅ device/dtype ត្រឹមត្រូវ
* - ល្បឿនមិនប្រសើរឡើងលើ CPU
  - CPU គ្មាន AVX512-BF16 ឬ overhead ច្រើនជាងផលចំណេញ
  - ប្រើ float32 លើ CPU លុះត្រាតែបានវាស់ថាជួយ
* - លទ្ធផលខុសពី float32 ខ្លាំង
  - Model រំញោចចំពោះ precision (ឧ. RNN ជ្រៅ)
  - ប្រើ bfloat16 ឬកំណត់ autocast តែលើផ្នែកខ្លះ
* - `GradScaler is enabled but device is CPU`
  - បានបើក scaler លើ CPU
  - `GradScaler(device="cpu", enabled=False)` ឬកំណត់តាម device
* - BatchNorm ផ្ដល់លទ្ធផលខុស
  - BatchNorm ត្រូវការ float32 statistics
  - autocast ដោះស្រាយវាដោយស្វ័យប្រវត្តិ — កុំ `.half()` ដោយដៃ
```

### កុំប្ដូរ parameter ទៅ fp16 ដោយដៃ

```python
# ❌ ខុស — បំផ្លាញ master weights
model = model.half()

# ✅ ត្រូវ — ឱ្យ autocast គ្រប់គ្រង
with autocast(device_type="cuda", dtype=torch.float16):
    out = model(x)
```

`model.half()` ប្ដូរ **weight ពិត** ទៅ float16 ដែលមានន្តៃថា optimizer update
ដែលតូចជាង precision នៃ float16 នឹងត្រូវបានបង្គត់ទៅសូន្យ។ ការបណ្ដុះបណ្ដាល
នឹងឈប់រៀន។

លើកលែងតែអ្នកកំពុងធ្វើ **pure fp16 inference** (មិនមាន training)។

## សម្រាប់ GPU

```{literalinclude} examples/mixed_precision_2.py
:language: python
:caption: mixed_precision_2.py — 🔒 ត្រូវការ CUDA GPU · ⚠️ មិនទាន់បានសាកល្បងដោយអ្នកនិពន្ធ (គ្មាន GPU ក្នុង environment បង្កើតឯកសារ)
```

ឧទាហរណ៍នេះប្រៀបធៀប float32, float16+GradScaler និង bfloat16 លើ MLP ជ្រៅ។
វាបោះពុម្ព loss, រយៈពេល, peak memory និង scale factor សម្រាប់ការកំណត់នីមួយៗ។

### ត្រួតពិនិត្យថា GPU របស់អ្នកគាំទ្រអ្វី

```python
major, minor = torch.cuda.get_device_capability(0)
print(f"compute capability: {major}.{minor}")
supports_bf16 = (major, minor) >= (8, 0)     # Ampere ឬខ្ពស់ជាង
print(f"bfloat16 hardware support: {supports_bf16}")
```

```{list-table}
:header-rows: 1
:widths: 30 70

* - Compute capability
  - ការណែនាំ
* - < 7.0 (Pascal និងចាស់ជាង)
  - គ្មាន tensor cores — mixed precision មានផលតិចតួច
* - 7.0 – 7.5 (Volta, Turing)
  - ប្រើ **float16 + GradScaler** (គ្មាន bfloat16 hardware)
* - ≥ 8.0 (Ampere, Ada, Hopper)
  - ប្រើ **bfloat16** — សាមញ្ញជាង មិនត្រូវការ GradScaler
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **`autocast` ប្ដូរប្រតិបត្តិការ មិនមែន parameter** — master weights នៅ float32។
2. **float16 ត្រូវការ `GradScaler`** · **bfloat16 មិនត្រូវការ**។
3. **ប្រើ bfloat16 បើ hardware គាំទ្រ** (compute capability ≥ 8.0) — សាមញ្ញជាង និងស្ថិរជាង។
4. **លើ CPU ប្រើ bfloat16** ប៉ុន្តែត្រួតពិនិត្យថាវាពិតជាជួយ។
5. **`scaler.unscale_(optimizer)` មុន gradient clipping**។
6. **កុំ `model.half()` ដោយដៃ** សម្រាប់ training។
7. **Loss, softmax, normalization នៅ float32** ដោយស្វ័យប្រវត្តិ។
8. **វាស់លើ hardware របស់អ្នក** — លេខល្បឿនមិនអាចផ្ទេរបានទេ។
```

## មេរៀនបន្ទាប់

{doc}`distributed-training` — បណ្ដុះបណ្ដាលលើ GPU ច្រើន និងម៉ាស៊ីនច្រើន។
