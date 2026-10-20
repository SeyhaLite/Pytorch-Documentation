# Performance Optimization

មុនធ្វើឱ្យកូដលឿនជាង អ្នកត្រូវ **វាស់** សិន។ ការស្មានថាអ្វីយឺតគឺជាមូលហេតុទីមួយ
នៃការខ្ជះខ្ជាយពេលវេលាក្នុងការធ្វើឱ្យប្រសើរឡើង។

```{admonition} តម្រូវការ
:class: note
🖥️ `performance_1.py` ដំណើរការលើ CPU។ 🔒 `performance_2.py` ត្រូវការ CUDA GPU
ហើយនឹងត្រូវរំលងដោយស្វ័យប្រវត្តិបើគ្មាន។
```

```{admonition} លេខក្នុងមេរៀននេះមិនមែនជា benchmark ជាសកលទេ
:class: danger
រាល់រយៈពេលទាំងអស់ដែលបង្ហាញនៅទីនេះត្រូវបាន **វាស់លើម៉ាស៊ីនដែលបង្កើតឯកសារនេះ**
(PyTorch `2.14.1+cpu`)។ ពួកវាបង្ហាញ **ទិសដៅ** នៃការប្រែប្រួល មិនមែនតម្លៃដាច់ខាតទេ។

លេខរបស់អ្នកនឹងខុសគ្នាអាស្រ័យលើ៖ CPU/GPU model, ចំនួន core, ប្រេកង់ RAM,
ចំនួន thread, កំណែ PyTorch, កំណែ driver, សីតុណ្ហភាពម៉ាស៊ីន និងអ្វីផ្សេងដែលកំពុងរត់។

**សូមវាស់លើម៉ាស៊ីនរបស់អ្នកផ្ទាល់ជានិច្ច។**
```

## ១. វាស់ឱ្យបានត្រឹមត្រូវ

```python
import time


def measure(fn, reps: int = 5) -> float:
    """វាស់រយៈពេលល្អបំផុតគិតជាវិនាទី ដោយមាន warmup។"""
    fn()                                  # warmup: JIT, cache, thread pool
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best
```

```{list-table}
:header-rows: 1
:widths: 30 70

* - ច្បាប់
  - ហេតុអ្វី
* - ធ្វើ warmup មុនវាស់
  - ការហៅដំបូងរួមបញ្ចូលការបង្កើត thread pool, memory allocation និង JIT compilation
* - យកលទ្ធផល **តូចបំផុត** មិនមែនមធ្យមភាគ
  - មធ្យមភាគត្រូវបានលំអៀងដោយការរំខាន (OS scheduler, process ផ្សេង)
* - ប្រើ `time.perf_counter()`
  - ភាពជាក់លាក់ខ្ពស់បំផុតដែលមាន
* - `torch.cuda.synchronize()` មុន និងក្រោយ (GPU)
  - CUDA ដំណើរការ asynchronous
* - វាស់ក្នុងលក្ខខណ្ឌដូច production
  - រួមទាំង data loading មិនមែនតែ forward pass
```

## ២. Vectorization — ការកែលម្អធំបំផុត

```{literalinclude} examples/performance_1.py
:language: python
:caption: performance_1.py — ✅ បានសាកល្បងលើ CPU
```

### បញ្ហាជាមួយ Python loop

```python
# ❌ យឺតខ្លាំង
total = 0.0
for i in range(big.shape[0]):
    total += big[i].sum().item()

# ✅ លឿនជាងច្រើនរយដង
total = big.sum().item()
```

ហេតុអ្វីខុសគ្នាខ្លាំងម្ល៉េះ? ការហៅ `.item()` **រាល់ជួរ** បង្ខំឱ្យ៖
1. Python interpreter ដំណើរការ loop overhead (រាប់ពាន់ដង)
2. ការបម្លែង C++ tensor → Python float (រាប់ពាន់ដង)
3. នៅលើ GPU៖ **host-device synchronization** (រាប់ពាន់ដង!)

វិធាន៖ **បើអ្នកអាចសរសេរវាជាប្រតិបត្តិការ tensor ធ្វើវា។**

### ទម្រង់ជំនួស

```{list-table}
:header-rows: 1
:widths: 40 60

* - ជំនួស loop នេះ
  - ដោយ
* - `for i: total += x[i]`
  - `x.sum()`
* - `for i: if x[i] > 0: out.append(x[i])`
  - `x[x > 0]`
* - `for i: out[i] = f(x[i])`
  - `f(x)` បើ `f` vectorized (ឧ. `torch.relu(x)`)
* - `for i: out[i] = x[i] if cond else 0`
  - `torch.where(cond, x, torch.zeros_like(x))`
* - `for i: out[i] = min(max(x[i], a), b)`
  - `torch.clamp(x, a, b)`
* - `for j: for i: out[i,j] = x[i] * y[j]`
  - `x[:, None] * y[None, :]` (broadcasting)
```

## ៣. dtype និង Memory

```{list-table}
:header-rows: 1
:widths: 18 18 64

* - dtype
  - Bytes/ធាតុ
  - ប្រើពេល
* - `torch.float64`
  - 8
  - ភាពជាក់លាក់ខ្ពស់ (scientific, `gradcheck`)។ យឺត និងប្រើ memory ២×
* - `torch.float32`
  - 4
  - **លំនាំដើម** សម្រាប់ការបណ្ដុះបណ្ដាលភាគច្រើន
* - `torch.bfloat16`
  - 2
  - Mixed precision លើ GPU Ampere+ និង CPU ដែលមាន AVX512-BF16
* - `torch.float16`
  - 2
  - Mixed precision លើ GPU ចាស់ជាង។ ត្រូវការ `GradScaler`
* - `torch.int8`
  - 1
  - Quantized inference
```

### ពិនិត្យការប្រើប្រាស់ memory

```python
# ទំហំ tensor មួយ
mb = t.numel() * t.element_size() / 1e6

# នៅលើ GPU
torch.cuda.memory_allocated() / 1e6      # MB ដែល tensor កំពុងប្រើ
torch.cuda.max_memory_allocated() / 1e6  # កំពូលដែលបានឈានដល់
torch.cuda.memory_reserved() / 1e6       # caching allocator កាន់កាប់
```

## ៤. Contiguity និង Memory Layout

Tensor ដែល **contiguous** មានធាតុរៀងជាបន្តបន្ទាប់ក្នុង memory។ ការចូលដល់ពួកវា
លឿនជាងព្រោះ CPU cache line ផ្ទុកធាតុជាប់គ្នា។

```python
m = torch.randn(1000, 1000)
m.is_contiguous()        # True
m.T.is_contiguous()      # False — stride បញ្ច្រាស

# ប្រតិបត្តិការតាមវិមាត្រ contiguous លឿនជាង
m.sum(dim=1)             # លឿន (អានជាបន្តបន្ទាប់)
m.T.sum(dim=0)           # យឺតជាង (លោតក្នុង memory)
```

### channels_last សម្រាប់ CNN

```python
model = model.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)
```

NCHW (លំនាំដើម) រក្សាទុក channel ទាំងអស់នៃ pixel មួយនៅឆ្ងាយពីគ្នា។
NHWC (`channels_last`) រក្សាទុកវានៅជាប់គ្នា ដែលសមនឹងរបៀបដែល cuDNN kernel
ចូលចិត្តអាន។ លទ្ធផលអាចល្អជាងគួរឱ្យកត់សម្គាល់លើ GPU ទំនើប — ប៉ុន្តែ **វាស់ដោយខ្លួនឯង**
ព្រោះវាអាចអាក្រក់ជាងលើ architecture ខ្លះ។

````{admonition} Tensor ត្រូវតែ contiguous ដើម្បី view
:class: note
`view()` ទាមទារ contiguous tensor។ បើមិនដូច្នោះទេ៖

```text
RuntimeError: view size is not compatible with input tensor's size and stride
```

ដំណោះស្រាយ៖ `.contiguous().view(...)` ឬប្រើ `.reshape(...)`។
ប៉ុន្តែចាំថា `.contiguous()` **ចម្លងទិន្នន័យ** — ធ្វើម្ដងគត់ មិនមែនរាល់ iteration។
មើល {doc}`tensors`។
````

## ៥. Data Loading ជា Bottleneck

ក្នុងគម្រោងពិតភាគច្រើន **data loading យឺតជាងការគណនា**។ សញ្ញា៖
GPU utilization ទាប ប៉ុន្តែ CPU ខ្ពស់។

```{list-table}
:header-rows: 1
:widths: 30 70

* - ការកែលម្អ
  - ការពន្យល់
* - `num_workers=4`–`8`
  - ផ្ទុកទិន្នន័យក្នុង process ផ្សេង។ ចាប់ផ្ដើមពី 4 ហើយបង្កើនដល់ចំនួន CPU cores
* - `pin_memory=True`
  - ប្រើ DMA សម្រាប់ការផ្លាស់ទី CPU→GPU (ត្រូវការ GPU)
* - `non_blocking=True`
  - អនុញ្ញាតឱ្យការផ្លាស់ទីត្រួតគ្នាជាមួយការគណនា (ត្រូវការ pinned memory)
* - `persistent_workers=True`
  - កុំបង្កើត workers ឡើងវិញរាល់ epoch
* - `prefetch_factor=4`
  - worker នីមួយៗផ្ទុក batches ច្រើនជាមុន
* - រក្សាទុកជាទម្រង់ binary
  - ប្ដូរពីរូបភាព JPEG ទៅ `.npy`/`webdataset` បើការ decode ជា bottleneck
* - ធ្វើ transform លើ batch
  - មើល {doc}`data-and-dataloader`
```

### របៀបដឹងថា data loading ជា bottleneck

```python
import time

# ១. វាស់ data loading តែម្នាក់ឯង
t0 = time.perf_counter()
for batch in loader:
    pass
load_time = time.perf_counter() - t0

# ២. វាស់ training loop ពេញលេញ
t0 = time.perf_counter()
for batch in loader:
    train_step(batch)
total_time = time.perf_counter() - t0

print(f"data loading: {load_time:.2f}s ({load_time / total_time * 100:.0f}% នៃសរុប)")
```

បើ data loading គិតជាច្រើនភាគរយនៃពេលសរុប → បង្កើន `num_workers`
ឬធ្វើឱ្យ `__getitem__` សាមញ្ញជាង។

## ៦. Autograd Overhead

```python
# ❌ ក្នុង inference — បង្កើត graph ដែលមិនចាំបាច់
out = model(x)

# ✅
with torch.no_grad():
    out = model(x)

# ✅✅ លឿនជាងបន្តិចសម្រាប់ inference សុទ្ធ
with torch.inference_mode():
    out = model(x)
```

## ៧. torch.compile

```python
model = torch.compile(model)                 # default mode
model = torch.compile(model, mode="reduce-overhead")   # CUDA graphs
model = torch.compile(model, mode="max-autotune")      # សាកល្បង kernel ច្រើន
model = torch.compile(model, dynamic=True)             # អនុញ្ញាតឱ្យ shape ប្រែប្រួល
```

មើល {doc}`evaluation-and-inference` សម្រាប់ modes និងដែនកំណត់។

## ៨. Profiling — រក Bottleneck ពិតប្រាកដ

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()                           # ← ចាំបាច់
```

ការវិភាគលទ្ធផល៖

```python
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
```

ឬមើលក្នុង TensorBoard៖

```bash
pip install torch-tb-profiler
tensorboard --logdir=./log/profiler
```

```{list-table} អ្វីដែលត្រូវមើលក្នុង profile
:header-rows: 1
:widths: 32 68

* - សញ្ញា
  - អត្ថន័យ
* - `aten::copy_` ខ្ពស់
  - ការផ្លាស់ទីទិន្នន័យច្រើនពេក (CPU↔GPU ឬ dtype)
* - `aten::item` / `aten::_local_scalar_dense` ខ្ពស់
  - ការហៅ `.item()` ក្នុង hot loop → synchronization
* - CPU time >> GPU time
  - data loading ឬ Python overhead ជា bottleneck
* - GPU ទំនេរច្រើន
  - batch តូចពេក ឬ model តូចពេកសម្រាប់ GPU
* - `cudaLaunchKernel` ច្រើនខ្លាំង
  - kernel តូចៗច្រើនពេក → `torch.compile` ឬ fuse operations
```

## ៩. GPU-specific

```{literalinclude} examples/performance_2.py
:language: python
:caption: performance_2.py — 🔒 ត្រូវការ CUDA GPU · ⚠️ មិនទាន់បានសាកល្បងដោយអ្នកនិពន្ធ (គ្មាន GPU ក្នុង environment បង្កើតឯកសារ)
```

### គន្លឹះសំខាន់ៗ

```{list-table}
:header-rows: 1
:widths: 32 68

* - បច្ចេកទេស
  - ការពន្យល់
* - ជៀសវាង `.item()` ក្នុង hot loop
  - រាល់ការហៅបង្ខំឱ្យ CPU រង់ចាំ GPU (synchronization)
* - Batch ធំជាង
  - GPU មានប្រសិទ្ធភាពជាងជាមួយការងារច្រើនក្នុងពេលតែមួយ
* - `channels_last` សម្រាប់ CNN
  - សមនឹង cuDNN kernel
* - TF32 (`set_float32_matmul_precision("high")`)
  - លឿនជាងលើ GPU Ampere+ ដោយភាពជាក់លាក់ថយចុះបន្តិច
* - Gradient accumulation ជំនួស batch ធំ
  - បើ GPU memory មិនគ្រប់គ្រាន់
* - `empty_cache()` មិនមែនជាដំណោះស្រាយ OOM
  - វាគ្រាន់តែប្រគល់ memory ទៅ OS។ បញ្ហាពិតគឺ tensor របស់អ្នកធំពេក
```

### TF32 និង float32 matmul precision

```python
torch.set_float32_matmul_precision("highest")   # float32 ពេញលេញ (លំនាំដើម)
torch.set_float32_matmul_precision("high")      # TF32 — លឿនជាង ភាពជាក់លាក់ថយបន្តិច
torch.set_float32_matmul_precision("medium")    # bfloat16 — លឿនបំផុត
```

## ១០. ការកែលម្អតាមលំដាប់អាទិភាព

```{list-table}
:header-rows: 1
:widths: 8 44 48

* - #
  - ធ្វើអ្វីមុន
  - ហេតុអ្វី
* - ១
  - វាស់។ រក bottleneck ដោយ profiler
  - ការកែលម្អដែលមិនមានការវាស់គឺជាការស្មាន
* - ២
  - Vectorize — ដក Python loops ចេញ
  - ជាទូទៅផ្ដល់ការកែលម្អ ១០–១០០០×
* - ៣
  - កែ data loading (`num_workers`, `pin_memory`)
  - ជា bottleneck ញឹកញាប់បំផុតក្នុងគម្រោងពិត
* - ៤
  - ប្រើ dtype តូចជាង (mixed precision)
  - មើល {doc}`mixed-precision`
* - ៥
  - `torch.compile()`
  - ការកែលម្អដោយស្វ័យប្រវត្តិ ដោយមិនកែកូដ
* - ៦
  - ប្ដូរ architecture ឬ algorithm
  - ការផ្លាស់ប្ដូរធំបំផុត ប៉ុន្តែត្រូវការការងារច្រើនបំផុត
```

```{admonition} កុំធ្វើឱ្យប្រសើរឡើងមុនពេលវាស់
:class: important
"ការកែលម្អដំបូង" ដែលមនុស្សភាគច្រើនធ្វើ (ឧ. ការប្រើ in-place operations,
ការជៀសវាងការបង្កើត tensor) ជាទូទៅផ្ដល់ផលតិចតួចបំផុត។ ការវាស់ជាមុន
អាចបង្ហាញថា ៩០% នៃពេលវេលាចំណាយលើ data loading — ដែលការកែលម្អគណនា
មិនជួយអ្វីទាំងអស់។
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **វាស់មុនកែលម្អ** — ប្រើ `torch.profiler`។
2. **Vectorize** — ដក Python loops និង `.item()` ចេញពី hot loops។
3. **Data loading ជា bottleneck ញឹកញាប់បំផុត** — បង្កើន `num_workers`។
4. **`torch.no_grad()`/`inference_mode()`** សម្រាប់ inference។
5. **dtype តូចជាង = memory តិចជាង និងលឿនជាង** — ប៉ុន្តែត្រួតពិនិត្យភាពជាក់លាក់។
6. **Contiguous access លឿនជាង** — ជៀសវាង `.contiguous()` រាល់ iteration។
7. **`torch.compile()`** ជាជំហានចុងក្រោយដែលមានផលច្រើន។
8. **លេខល្បឿនមិនអាចផ្ទេររវាងម៉ាស៊ីនបានទេ** — វាស់លើ hardware របស់អ្នក។
```

## មេរៀនបន្ទាប់

{doc}`mixed-precision` — ប្រើ float16/bfloat16 ដើម្បីបង្កើនល្បឿន និងសន្សំ memory។
