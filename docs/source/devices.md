# CPU និង GPU៖ Devices

Tensor និង model ទាំងអស់រស់នៅលើ **device** ជាក់លាក់មួយ។ ការយល់ពី device គឺជា
គន្លឹះសម្រាប់ប្រើ GPU បានត្រឹមត្រូវ និងជៀសវាងកំហុសដែលជួបញឹកញាប់បំផុតមួយរបស់ PyTorch។

```{admonition} តម្រូវការ
:class: note
🖥️ ឧទាហរណ៍ក្នុងមេរៀននេះដំណើរការលើ CPU ហើយនឹងរំលងផ្នែក GPU ដោយស្វ័យប្រវត្តិ
បើគ្មាន។ វាបានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## Devices ទាំងបីរបស់ PyTorch

```{list-table}
:header-rows: 1
:widths: 18 22 60

* - Device string
  - Hardware
  - ការពិនិត្យ
* - `cpu`
  - គ្រប់ម៉ាស៊ីន
  - តែងតែមាន។ លំនាំដើម។
* - `cuda` / `cuda:0`
  - NVIDIA GPU
  - `torch.cuda.is_available()`
* - `mps`
  - Apple Silicon (M1–M4)
  - `torch.backends.mps.is_available()`
```

```{admonition} XPU, ROCm, Vulkan និង backends ផ្សេងទៀត
:class: note
PyTorch ក៏គាំទ្រ Intel GPU (`xpu`), AMD GPU (ROCm ដែលបង្ហាញខ្លួនជា `cuda`) និង
backends ពិសេសផ្សេងទៀត។ ឯកសារនេះផ្ដោតលើ CPU និង CUDA ព្រោះវាជាអ្វីដែលអ្នកប្រើភាគច្រើនជួប។
```

## រូបមន្តជ្រើសរើស device

នេះគឺជាទម្រង់ស្តង់ដារដែលអ្នកគួរប្រើក្នុងគ្រប់គម្រោង៖

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

ឬខ្លីជាង (PyTorch 2.4+)៖

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

```{admonition} កុំសរសេរ cuda:0 ដោយផ្ទាល់ក្នុងកូដ
:class: warning
បើអ្នកសរសេរ `.to("cuda:0")` កូដរបស់អ្នកនឹងខូចភ្លាមៗនៅលើម៉ាស៊ីនគ្មាន GPU។
ប្រើ `torch.device` object ដែលបានគណនាម្ដងគត់ ហើយបញ្ជូនវាជុំវិញ។
នេះធ្វើឱ្យកូដដំណើរការលើម៉ាស៊ីនណាក៏បាន ដោយមិនកែអ្វីទាំងអស់។
```

## ឧទាហរណ៍ពេញលេញ

```{literalinclude} examples/devices_1.py
:language: python
:caption: devices_1.py — ✅ បានសាកល្បងលើ CPU (ផ្នែក GPU នឹងរំលងដោយស្វ័័យប្រវត្តិ)
```

## ច្បាប់សំខាន់ៗ

### ១. ប្រតិបត្តិការទាំងអស់ត្រូវការ tensor នៅលើ device ដូចគ្នា

```python
a = torch.ones(3, device="cuda")
b = torch.ones(3, device="cpu")
a + b        # RuntimeError!
```

សារកំហុសពិត៖

```text
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:0 and cpu!
```

ដំណោះស្រាយ៖ ផ្លាស់ទីឱ្យទៅ device តែមួយ **មុន** ធ្វើប្រតិបត្តិការ។

### ២. Model និង input ត្រូវតែនៅ device ដូចគ្នា

```python
model = model.to(device)                 # ផ្លាស់ទី parameter ទាំងអស់
for xb, yb in loader:
    xb = xb.to(device, non_blocking=True)  # ផ្លាស់ទីរាល់ batch
    yb = yb.to(device, non_blocking=True)
    out = model(xb)
```

````{admonition} .to() មិនកែ tensor ដើមទេ
:class: danger
`Tensor.to()` ត្រឡប់ tensor **ថ្មី**។ វាមិនកែ tensor ដើមនៅនឹងកន្លែងទេ៖

```python
x = torch.randn(3)             # cpu
x.to("cuda")                   # ⚠️ ត្រឡប់ tensor ថ្មី ប៉ុន្តែអ្នកទម្លាក់វាចោល!
print(x.device)                # cpu — មិនបានផ្លាស់ទីទេ

x = x.to("cuda")               # ✓ assign វិញ
```

**លើកលែងតែ `nn.Module.to()`** ដែលកែ module នៅនឹងកន្លែង ហើយត្រឡប់ខ្លួនវាវិញ។
ដូច្នេះ `model.to(device)` ដំណើរការដោយមិនចាំបាច់ assign វិញ
(ប៉ុន្តែការ assign វិញក៏មិនបង្កគ្រោះថ្នាក់ដែរ)។
````

### ៣. Output របស់ model នៅលើ device ដូច input

បើ input នៅលើ `cuda:0` output ក៏នៅ `cuda:0` ដែរ។ អ្នកត្រូវផ្លាស់ទីវាមក CPU វិញ
មុនធ្វើអ្វីដែល CPU-only៖

```python
loss.item()                    # ✓ ដំណើរការ (ប្ដូរតម្លៃតែមួយមក CPU)
preds.cpu().numpy()            # ✓ ត្រូវ .cpu() ជាមុន
preds.numpy()                  # ✗ TypeError លើ GPU tensor
losses.append(loss)            # ✗ រក្សាទុក GPU tensor -> memory leak!
losses.append(loss.item())     # ✓ រក្សាទុក Python float
```

## ព័ត៌មានអំពី GPU

```{list-table}
:header-rows: 1
:widths: 46 54

* - ពាក្យបញ្ជា
  - អ្វីដែលវាប្រាប់
* - `torch.cuda.device_count()`
  - ចំនួន GPU ដែលអាចប្រើបាន
* - `torch.cuda.get_device_name(0)`
  - ឈ្មោះ GPU (ឧ. "NVIDIA GeForce RTX 4090")
* - `torch.cuda.get_device_capability(0)`
  - Compute capability (ឧ. `(8, 9)` = Ada)។ ប្រើដើម្បីដឹងថាវាគាំទ្រ bfloat16 ឬទេ
* - `torch.cuda.memory_allocated()`
  - bytes ដែល tensor កំពុងប្រើ
* - `torch.cuda.memory_reserved()`
  - bytes ដែល caching allocator កាន់កាប់ (ធំជាង ឬស្មើ allocated)
* - `torch.cuda.max_memory_allocated()`
  - កំពូលដែលបានឈានដល់ — ប្រើសម្រាប់វាស់ memory
* - `torch.cuda.empty_cache()`
  - ប្រគល់ memory ដែលមិនប្រើទៅ OS។ **មិនមែន**ជាវិធីដោះស្រាយ OOM ទេ
* - `torch.cuda.synchronize()`
  - រង់ចាំឱ្យប្រតិបត្តិការ GPU ទាំងអស់បញ្ចប់ — ចាំបាច់មុនវាស់ពេល
```

## វាស់ពេលលើ GPU — កំហុសទូទៅ

CUDA ដំណើរការ **ដោយ asynchronous**៖ `model(x)` ត្រឡប់ភ្លាមៗ ប៉ុន្តែការគណនានៅមិនទាន់ចប់ទេ។

```python
import time

# ❌ ខុស — វាស់តែពេលដាក់បញ្ជាទៅ GPU មិនមែនពេលគណនាពិត
t0 = time.perf_counter()
out = model(x)
print(time.perf_counter() - t0)       # ជិតសូន្យ!

# ✓ ត្រូវ — synchronize មុន និងក្រោយ
torch.cuda.synchronize()
t0 = time.perf_counter()
out = model(x)
torch.cuda.synchronize()
print(time.perf_counter() - t0)
```

ឬល្អជាង ប្រើ `torch.cuda.Event`៖

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
out = model(x)
end.record()
torch.cuda.synchronize()
print(f"{start.elapsed_time(end):.2f} ms")     # milliseconds
```

មើល {doc}`performance` សម្រាប់បច្ចេកទេស benchmark ពេញលេញ។

## pin_memory និង non_blocking

សម្រាប់ការផ្លាស់ទីទិន្នន័យ CPU → GPU លឿនជាង៖

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,          # ដាក់ទិន្នន័យក្នុង pinned (page-locked) memory
)

for xb, yb in loader:
    xb = xb.to(device, non_blocking=True)   # អនុញ្ញាតឱ្យផ្លាស់ទីត្រួតគ្នាជាមួយការគណនា
    yb = yb.to(device, non_blocking=True)
```

- **`pin_memory=True`** ប្រើ DMA ដែលលឿនជាង ប៉ុន្តែប្រើ RAM ច្រើនជាង។
- **`non_blocking=True`** មានប្រសិទ្ធភាព **តែ** ពេលប្រភពជា pinned memory។
- ទាំងពីរគ្មានប្រយោជន៍ទេបើអ្នកបណ្ដុះបណ្ដាលលើ CPU។

## Multi-GPU សង្ខេប

```{list-table}
:header-rows: 1
:widths: 26 30 44

* - វិធី
  - ស្ថានភាព
  - ចំណាំ
* - `nn.DataParallel`
  - ⚠️ មិនណែនាំទៀតទេ
  - សាមញ្ញ ប៉ុន្តែយឺត និងមានបញ្ហា GIL។ ប្រើតែសម្រាប់ prototype ប៉ុណ្ណោះ។
* - `nn.parallel.DistributedDataParallel`
  - ✅ ស្តង់ដារ
  - លឿនជាងច្រើន។ ត្រូវការ process មួយក្នុងមួយ GPU។ មើល {doc}`distributed-training`។
* - Device mesh / FSDP
  - ✅ សម្រាប់ model ធំ
  - បំបែក parameter រវាង GPU។ សម្រាប់ model ដែលមិនអាចសមក្នុង GPU តែមួយ។
```

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `Expected all tensors to be on the same device`
  - tensor នៅ device ផ្សេងគ្នា
  - `.to(device)` ទាំង input និង model
* - `CUDA error: out of memory`
  - GPU memory ពេញ
  - បន្ថយ batch size, ប្រើ gradient accumulation ឬ mixed precision
* - `can't convert cuda:0 device type tensor to numpy`
  - `.numpy()` លើ GPU tensor
  - `.detach().cpu().numpy()`
* - `Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`
  - ផ្ទុក GPU checkpoint លើម៉ាស៊ីនគ្មាន GPU
  - `torch.load(path, map_location="cpu")`
* - `Torch not compiled with CUDA enabled`
  - បានដំឡើង wheel CPU
  - ដំឡើងឡើងវិញជាមួយ `--index-url .../cu128`។ មើល {doc}`installation`
* - ល្បឿនមិនប្រសើរឡើងទោះបីប្រើ GPU
  - data loading គឺជា bottleneck
  - បង្កើន `num_workers`, ប្រើ `pin_memory=True`។ មើល {doc}`performance`
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **គណនា `device` ម្ដងគត់** នៅដើមកម្មវិធី ហើយបញ្ជូនវាជុំវិញ។
2. **`model.to(device)` មុន `optimizer`** — optimizer យក reference ទៅ parameter ដដែល។
3. **ផ្លាស់ទី input រាល់ batch** ជាមួយ `.to(device, non_blocking=True)`។
4. **យកលទ្ធផលមក CPU វិញ** មុន `.numpy()`, រក្សាទុក ឬបង្ហាញ។
5. **ប្រើ `.item()` សម្រាប់ logging** — កុំរក្សាទុក GPU tensor ក្នុង list។
6. **`torch.cuda.synchronize()` មុនវាស់ពេល** ជានិច្ច។
```

## មេរៀនបន្ទាប់

{doc}`autograd` — របៀបដែល PyTorch គណនា gradient ដោយស្វ័យប្រវត្តិ។
