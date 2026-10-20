# Random Seeds និង Reproducibility

**Reproducibility** មានន័យថា ការរត់កូដដូចគ្នាពីរដងផ្ដល់លទ្ធផលដូចគ្នា។ នេះសំខាន់
សម្រាប់ការ debug, ការប្រៀបធៀបការពិសោធន៍ និងការស្រាវជ្រាវដែលអាចទុកចិត្តបាន។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍បានសាកល្បងលើ PyTorch `2.14.1+cpu` + NumPy `2.4.6`។
```

## ប្រភពភាពចៃដន្យទាំងអស់

ភាពចៃដន្យនៅក្នុងគម្រោង ML មកពី **ប្រភពច្រើនជាងមួយ**។ ការកំណត់ seed តែលើ
`torch` មិនគ្រប់គ្រាន់ទេ៖

```{list-table}
:header-rows: 1
:widths: 30 34 36

* - ប្រភព
  - ប៉ះពាល់អ្វី
  - វិធីគ្រប់គ្រង
* - Python `random`
  - augmentation ផ្ទាល់ខ្លួន, sampling
  - `random.seed(s)`
* - NumPy `np.random`
  - augmentation ដែលប្រើ numpy
  - `np.random.seed(s)`
* - PyTorch CPU RNG
  - `torch.randn`, weight init, Dropout
  - `torch.manual_seed(s)`
* - PyTorch CUDA RNG
  - ដូចគ្នានៅលើ GPU
  - `torch.cuda.manual_seed_all(s)`
* - DataLoader shuffle
  - លំដាប់ batch
  - `generator=` ប៉ារ៉ាម៉ែត្រ
* - DataLoader workers
  - augmentation ក្នុង worker
  - `worker_init_fn=`
* - `random_split()`
  - train/val split
  - `generator=` ប៉ារ៉ាម៉ែត្រ
* - cuDNN algorithm selection
  - ល្បឿន និងលទ្ធផលលេខ
  - `torch.backends.cudnn.deterministic`
* - Atomic GPU operations
  - gradient reduction
  - `torch.use_deterministic_algorithms(True)`
```

### រូបមន្តស្តង់ដារ

```python
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """កំណត់ seed សម្រាប់គ្រប់ប្រភពភាពចៃដន្យ។

    ហៅនៅដើមកម្មវិធី មុនបង្កើត model ឬ DataLoader។
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # សម្រាប់ determinism ពេញលេញ (មើលផ្នែកខាងក្រោម)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

```{literalinclude} examples/reproducibility_1.py
:language: python
:caption: reproducibility_1.py — ✅ បានសាកល្បងលើ CPU
```

## torch.Generator — ការគ្រប់គ្រងដាច់ដោយឡែក

`torch.Generator` អនុញ្ញាតឱ្យអ្នកផ្ដល់ seed ដាច់ដោយឡែកសម្រាប់ប្រតិបត្តិការជាក់លាក់
ដោយ **មិនប៉ះពាល់ global RNG**៖

```python
g = torch.Generator().manual_seed(7)
x = torch.randn(5, generator=g)      # deterministic
y = torch.randn(5)                   # ប្រើ global RNG — មិនប៉ះពាល់ដោយ g
```

នេះមានប្រយោជន៍ខ្លាំងសម្រាប់៖

```{list-table}
:header-rows: 1
:widths: 30 70

* - ការប្រើប្រាស់
  - ហេតុអ្វី
* - `DataLoader(generator=g)`
  - shuffle ដូចគ្នារាល់ការរត់ ទោះបីមានការហៅ random ផ្សេងនៅចន្លោះ
* - `random_split(ds, [...], generator=g)`
  - train/val split ដូចគ្នាជានិច្ច
* - បង្កើតទិន្នន័យសាកល្បង
  - មិនរំខាន seed នៃការពិសោធន៍
* - augmentation ក្នុង Dataset
  - អាចគ្រប់គ្រងដាច់ដោយឡែកក្នុងមួយ worker
```

```{admonition} Generator ដូចគ្នា → លំដាប់ដូចគ្នា
:class: note
Generator ពីរដែលមាន seed ដូចគ្នាផលិតលំដាប់លេខដូចគ្នា។ ប៉ុន្តែបើអ្នកប្រើ
**Generator តែមួយ** សម្រាប់ប្រតិបត្តិការច្រើន លទ្ធផលនៃប្រតិបត្តិការទីពីរ
អាស្រ័យលើអ្វីដែលបានកើតឡើងមុន។ នេះជាមូលហេតុដែល DataLoader គួរមាន
Generator ផ្ទាល់ខ្លួន។
```

## DataLoader Reproducibility

### shuffle

```python
g = torch.Generator().manual_seed(42)
loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
```

បើអ្នកមិនផ្ដល់ `generator=` ទេ DataLoader ប្រើ global torch RNG។ នេះមានន័យថា
លំដាប់ batch **អាស្រ័យលើអ្វីដែលបានរត់មុនវា**។

### num_workers > 0

```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

```{admonition} Worker ទទួល torch seed ដោយស្វ័យប្រវត្តិ ប៉ុន្តែមិនមែន NumPy
:class: warning
PyTorch ផ្ដល់ seed ដាច់ដោយឡែកទៅ `torch` RNG ក្នុង worker នីមួយៗ។
ប៉ុន្តែ **NumPy និង Python `random` មិនត្រូវបាន seed** ដូច្នេះ augmentation
ដែលប្រើ `np.random` នឹងផលិតលទ្ធផល **ដូចគ្នាក្នុងគ្រប់ worker** —
ដែលធ្វើឱ្យ augmentation របស់អ្នកមិនមានប្រសិទ្ធភាព។

`worker_init_fn` ដោះស្រាយបញ្ហានេះ។
```

## Deterministic Algorithms

ការកំណត់ seed គ្រប់គ្រងតែ **លេខចៃដន្យ**។ វាមិនគ្រប់គ្រង **លំដាប់ប្រតិបត្តិការ** ទេ។
ប្រតិបត្តិការមួយចំនួនលើ GPU មិនមានកំណែ deterministic ទេ។

```python
# ត្រូវកំណត់ environment variable មុន import torch (សម្រាប់ CUDA >= 10.2)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.use_deterministic_algorithms(True)       # ឬ warn_only=True សម្រាប់ការពិសោធន៍
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```{list-table}
:header-rows: 1
:widths: 26 24 50

* - ការកំណត់
  - ឥទ្ធិពល
  - ចំណាំ
* - `use_deterministic_algorithms(True)`
  - ល្បឿនថយចុះ
  - បង្ខំ algorithm deterministic។ បរាជ័យបើគ្មានកំណែ deterministic
* - `use_deterministic_algorithms(True, warn_only=True)`
  - ល្បឿនថយចុះ
  - បញ្ចេញ **ព្រមាន** ជំនួស error — ល្អសម្រាប់ស្វែងរកប្រភព
* - `cudnn.deterministic = True`
  - ល្បឿនថយចុះ
  - cuDNN ជ្រើសរើស algorithm deterministic
* - `cudnn.benchmark = False`
  - ល្បឿនថយចុះបន្តិច
  - បិទការស្វែងរក algorithm លឿនបំផុតដោយវាស់ពេលជាក់ស្ដែង
```

### ប្រតិបត្តិការដែលមិន deterministic

```{list-table}
:header-rows: 1
:widths: 40 60

* - ប្រតិបត្តិការ
  - ហេតុអ្វី
* - `index_add_`, `index_put_(accumulate=True)`
  - atomic additions លើ GPU មានលំដាប់មិនទៀងទាត់
* - `scatter_add_`
  - ដូចគ្នា
* - `nn.Embedding` backward (ជាមួយ sparse grad)
  - atomic accumulation
* - `nn.MaxPool*d` backward
  - ties អាចត្រូវបានដោះស្រាយខុសគ្នា
* - `nn.CTCLoss` backward
  - អាស្រ័យលើ implementation
* - `nn.functional.interpolate` មួយចំនួន
  - antialiasing មិន deterministic
* - `torch.bincount` លើ GPU ជាមួយ `minlength=None`
  - លំដាប់ accumulation
```

## អ្វីដែល seed មិនអាចគ្រប់គ្រងបាន

````{admonition} ដែនកំណត់ពិតប្រាកដនៃ reproducibility
:class: danger
សូម្បីតែជាមួយ seed ទាំងអស់ អ្នកអាចមិនទទួលបានលទ្ធផល **ដូចគ្នាបីត** ទេ។

**១. Floating point មិន associative**

```text
(a + b) + c  ≠  a + (b + c)   ក្នុង float32
```

ប្រតិបត្តិការដែលបូកធាតុច្រើនតាមលំដាប់ផ្សេងគ្នា (ឧ. multi-thread reduction)
ផ្ដល់លទ្ធផលខុសគ្នាបន្តិចបន្តួច។

**២. Multi-GPU / DDP**

លំដាប់ gradient reduction អាចប្រែប្រួល។ មើល {doc}`distributed-training`។

**៣. Hardware និងកំណែ software ផ្សេងគ្នា**

GPU ខុសគ្នា, cuDNN version ខុសគ្នា ឬ PyTorch version ខុសគ្នា អាចផ្ដល់លទ្ធផលខុសគ្នា។

**៤. cuDNN benchmark mode**

`torch.backends.cudnn.benchmark = True` ជ្រើសរើស algorithm ដោយ **វាស់ពេលជាក់ស្ដែង**
ដូច្នេះវាអាចជ្រើសរើសខុសគ្នារាល់ការរត់។

**ការណែនាំជាក់ស្ដែង៖** កំណត់ seed ហើយរាយការណ៍វានៅក្នុងអត្ថបទ។
បើអ្នកត្រូវការភាពជាក់លាក់ពេញលេញ ប្រើ `use_deterministic_algorithms(True)`
លើ CPU ឬ GPU តែមួយ។
````

## ការកត់ត្រាសម្រាប់ការពិសោធន៍

Reproducibility មិនមែនគ្រាន់តែជា seed ទេ — អ្នកត្រូវកត់ត្រា **បរិបទទាំងមូល**៖

```python
import platform
import subprocess
import sys

import torch


def log_environment() -> dict:
    """ចាប់យកព័ត៌មានបរិបទសម្រាប់ការពិសោធន៍។"""
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        info["git_commit"] = "unknown"
        info["git_dirty"] = None
    return info
```

```{admonition} git_dirty សំខាន់ណាស់
:class: tip
បើ working tree របស់អ្នកមានការផ្លាស់ប្ដូរដែលមិនទាន់ commit ការរត់នោះ
**មិនអាចធ្វើឡើងវិញបានទេ** ទោះបីអ្នកដឹង commit hash ក៏ដោយ។
កត់ត្រា `git_dirty=True` ដើម្បីដឹងថាលទ្ធផលមិនអាចទុកចិត្តបានពេញលេញ។
```

## រូបមន្តពេញលេញ

```python
def reproducible_run(seed: int, config: dict, train_fn) -> dict:
    seed_everything(seed)
    env = log_environment()
    env.update({"seed": seed, **config})

    result = train_fn(config)

    env.update(result)
    with open(f"runs/{seed}_{config['name']}.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, default=str)
    return result
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **`seed_everything()`** នៅដើមកម្មវិធី — Python, NumPy, torch, CUDA។
2. **បញ្ជាក់ `generator=`** សម្រាប់ DataLoader shuffle និង `random_split`។
3. **`worker_init_fn=`** ពេល `num_workers > 0` — NumPy មិនត្រូវបាន seed ដោយស្វ័យប្រវត្តិទេ។
4. **`torch.use_deterministic_algorithms(True)`** សម្រាប់ភាពជាក់លាក់ពេញលេញ — ប៉ុន្តែយឺតជាង។
5. **`CUBLAS_WORKSPACE_CONFIG=":4096:8"`** ត្រូវកំណត់ **មុន** `import torch`។
6. **កត់ត្រា seed + git commit + versions + device** ជាមួយលទ្ធផលរាល់ការពិសោធន៍។
7. **Reproducibility ពេញលេញមិនអាចធ្វើបានលើ multi-GPU** — រាយការណ៍ variance ជំនួសវិញ។
```

## មេរៀនបន្ទាប់

{doc}`performance` — ធ្វើឱ្យកូដរបស់អ្នករត់លឿនជាង។
