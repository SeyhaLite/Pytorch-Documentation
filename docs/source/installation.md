# ដំឡើង PyTorch និងរៀបចំ Environment

```{admonition} តម្រូវការ
:class: note
- **Python** 3.9 ឬខ្ពស់ជាង (ណែនាំ 3.11 ឬ 3.12)
- **pip** ឬ **conda**
- 🖥️ CPU គ្រប់គ្រាន់សម្រាប់រាល់មេរៀនក្នុងឯកសារនេះ។ GPU មានប្រយោជន៍តែសម្រាប់ {doc}`mixed-precision` និង {doc}`distributed-training` ប៉ុណ្ណោះ។
```

មេរៀននេះបង្ហាញពីរបៀបដំឡើង PyTorch ឱ្យត្រឹមត្រូវ ជៀសវាងបញ្ហាទូទៅ
ដូចជាកំណែ CUDA មិនត្រូវគ្នា ឬការដំឡើងលើ system Python។

## ជំហានទី ១៖ បង្កើត Virtual Environment

**កុំដំឡើង PyTorch លើ system Python របស់អ្នក។** Virtual environment ការពារ
ការប៉ះទង្គិចកំណែរវាងគម្រោង។

```{code-block} bash
:caption: Linux / macOS
# បង្កើត environment ឈ្មោះ "pt"
python3 -m venv .venv-pt

# ធ្វើឱ្យវាសកម្ម
source .venv-pt/bin/activate

# ផ្ទៀងផ្ទាត់ថា pip ចង្អុលទៅ environment ត្រឹមត្រូវ
which python
which pip
```

```{code-block} powershell
:caption: Windows (PowerShell)
py -3.12 -m venv .venv-pt
.venv-pt\Scripts\Activate.ps1

# បើ PowerShell រារាំង script:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```{code-block} bash
:caption: ជម្រើស conda
conda create -n pt python=3.12 -y
conda activate pt
```

```{admonition} របៀបដឹងថា environment សកម្មហើយ
:class: tip
នៅពេលដែល virtual environment សកម្ម អ្នកនឹងឃើញ `(.venv-pt)` នៅដើម prompt របស់អ្នក។
ពិនិត្យដោយ `which python` (Linux/macOS) ឬ `where python` (Windows) —
វាត្រូវចង្អុលទៅផ្លូវក្នុង `.venv-pt/` មិនមែន `/usr/bin/python` ទេ។
```

## ជំហានទី ២៖ ជ្រើសរើសកំណែ PyTorch ត្រឹមត្រូវ

PyTorch ផ្ដល់ wheel ផ្សេងគ្នាសម្រាប់ CPU និង CUDA។ **ការជ្រើសរើសខុសគឺជាបញ្ហាទូទៅបំផុត**។

```{list-table} ជម្រើសដំឡើង
:header-rows: 1
:widths: 22 46 32

* - ស្ថានភាព
  - ពាក្យបញ្ជា
  - ទំហំ download ប្រហាក់ប្រហែល
* - 🖥️ CPU តែប៉ុណ្ណោះ
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - ~200 MB
* - 🎮 NVIDIA GPU + CUDA 12.8
  - `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  - ~2.5 GB
* - 🎮 NVIDIA GPU + CUDA 12.6
  - `pip install torch --index-url https://download.pytorch.org/whl/cu126`
  - ~2.5 GB
* - 🍎 Apple Silicon (M1/M2/M3/M4)
  - `pip install torch`
  - ~70 MB (MPS backend មានស្រាប់)
```

```{admonition} ចំណាំសំខាន់អំពី CUDA
:class: warning
`--index-url .../cu128` **មិនមែន**ជាអ្វីដែលត្រូវដំឡើង NVIDIA driver របស់អ្នកទេ។
PyTorch wheel មាន CUDA runtime របស់វាផ្ទាល់។ អ្វីដែលអ្នកត្រូវការគឺ៖

1. **NVIDIA driver** ដែលបានដំឡើងរួច (ពិនិត្យដោយ `nvidia-smi`)
2. Driver នោះគាំទ្រ CUDA version ដែល wheel ត្រូវការ

`nvidia-smi` បង្ហាញ "CUDA Version" ដែលជា **កំណែអតិបរមា** ដែល driver របស់អ្នកគាំទ្រ
មិនមែនកំណែដែលបានដំឡើងទេ។ បើវាបង្ហាញ `12.4` អ្នកអាចប្រើ wheel `cu121` បាន
ប៉ុន្តែមិនអាចប្រើ `cu128` ទេ។
```

### ពិនិត្យ driver GPU (បើមាន)

```{code-block} bash
nvidia-smi
```

Output គំរូ (ឧទាហរណ៍សម្រាប់បង្ហាញទម្រង់ មិនមែនលទ្ធផលដែលបានវាស់ទេ)៖

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.16              Driver Version: 570.86.16      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On   | 00000000:01:00.0  On   |                  Off |
+-----------------------------------------+------------------------+----------------------+
```

ក្នុងឧទាហរណ៍នេះ `CUDA Version: 12.8` មានន័យថា driver គាំទ្ររហូតដល់ CUDA 12.8។

## ជំហានទី ៣៖ ដំឡើង

```{code-block} bash
:caption: CPU (ណែនាំសម្រាប់ការសិក្សា)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NumPy មានប្រយោជន៍សម្រាប់ការផ្លាស់ប្ដូរទិន្នន័យទៅ/ពី tensor
pip install numpy
```

```{code-block} bash
:caption: ជាមួយ GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```{code-block} bash
:caption: Apple Silicon
pip install torch torchvision torchaudio
```

```{admonition} តើត្រូវដំឡើង torchvision និង torchaudio ដែរឬទេ?
:class: tip
- **`torch`** — គ្រប់គ្រាន់សម្រាប់រាល់មេរៀនក្នុងឯកសារនេះ។
- **`torchvision`** — សម្រាប់រូបភាព (datasets, transforms, pretrained models)។
- **`torchaudio`** — សម្រាប់សំឡេង។

ឯកសារនេះ **មិនពឹងលើ torchvision ឬ torchaudio ទេ** ដើម្បីឱ្យអ្នកអាចរៀន
ដោយដំឡើងតិចបំផុត។ កំណែទាំងបីត្រូវតែត្រូវគ្នា (ឧ. torch 2.14 + torchvision 0.24)។
```

## ជំហានទី ៤៖ ផ្ទៀងផ្ទាត់ការដំឡើង

ឯកសារខាងក្រោមគឺជាឧទាហរណ៍ពិតដែលបានសាកល្បងរួចហើយ៖

```{literalinclude} examples/installation_1.py
:language: python
:caption: installation_1.py — ✅ បានសាកល្បងលើ CPU (PyTorch 2.14.1+cpu, Python 3.11.2)
```

រត់វា៖

```{code-block} bash
python docs/source/examples/installation_1.py
```

Output ដែលអ្នកនឹងឃើញលើម៉ាស៊ីនគ្មាន GPU (ចម្លងពីការរត់ពិត)៖

```text
==========================================================
PyTorch Environment Check
==========================================================
Python        : 3.11.2
PyTorch       : 2.14.1+cpu
Platform      : Linux 6.1.0 (x86_64)
CUDA (build)  : None
CUDA available: False
  (គ្មាន GPU — ឧទាហរណ៍ទាំងអស់ក្នុងឯកសារនេះដំណើរការលើ CPU បាន)
MPS available : False
CPU threads   : 8
----------------------------------------------------------
Smoke test:  x @ x.T
tensor([[ 5., 11.],
        [11., 25.]])
==========================================================
OK — PyTorch ដំណើរការត្រឹមត្រូវ
```

### ពិនិត្យរហ័សមួយបន្ទាត់

```{code-block} bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

បើឃើញកំណែ និង `True`/`False` ដោយគ្មាន error ការដំឡើងជោគជ័យហើយ។

## ដំណោះស្រាយបញ្ហាទូទៅ

```{list-table}
:header-rows: 1
:widths: 36 64

* - បញ្ហា
  - ដំណោះស្រាយ
* - `ModuleNotFoundError: No module named 'torch'`
  - Environment មិនសកម្ម ឬបានដំឡើងទៅ environment ខុស។ រត់ `which python` ហើយផ្ទៀងផ្ទាត់ថាវានៅក្នុង `.venv-pt/`។
* - `torch.cuda.is_available()` ត្រឡប់ `False` ទោះបីមាន GPU
  - អ្នកបានដំឡើង wheel CPU។ ដំឡើងឡើងវិញជាមួយ `--index-url .../cu128`។ ពិនិត្យ `nvidia-smi` ផង។
* - `ImportError: libtorch_cuda.so: cannot open shared object file`
  - wheel CUDA ខូច ឬមិនពេញលេញ។ `pip uninstall torch` ហើយដំឡើងឡើងវិញ។
* - `GLIBC_2.28 not found`
  - OS ចាស់ពេកសម្រាប់ wheel ថ្មី។ ប្រើ PyTorch កំណែចាស់ជាង ឬ conda។
* - `pip install` យឺតខ្លាំង / download 2 GB+
  - អ្នកកំពុងដំឡើងកំណែ CUDA។ ប្រើ wheel CPU បើអ្នកមិនមាន GPU។
* - `Failed to initialize NumPy`
  - `pip install numpy` — torch 2.x ប្រើ NumPy សម្រាប់ `.numpy()`។
* - `DefaultCPUAllocator: not enough memory`
  - tensor ធំពេកសម្រាប់ RAM របស់អ្នក។ បន្ថយ batch size។
* - កំណែ `torch` និង `torchvision` មិនត្រូវគ្នា
  - ដំឡើងទាំងពីរក្នុងពាក្យបញ្ជាតែមួយ ពី `--index-url` ដូចគ្នា។
```

### ដំឡើងឡើងវិញដោយសុវត្ថិភាព

```{code-block} bash
# ១. មើលអ្វីដែលបានដំឡើង
pip list | grep -i torch

# ២. ដកចេញទាំងអស់
pip uninstall -y torch torchvision torchaudio

# ៣. សម្អាត cache (សន្សំ disk)
pip cache purge

# ៤. ដំឡើងឡើងវិញពី index ត្រឹមត្រូវ
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ៥. ផ្ទៀងផ្ទាត់
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## កំណត់ត្រាកំណែសម្រាប់គម្រោងពិត

បង្កើត `requirements.txt` ដែលបញ្ជាក់កំណែជាក់លាក់ ដើម្បីឱ្យមិត្តរួមក្រុមទទួលបាន
environment ដូចគ្នា៖

```{code-block} text
:caption: requirements.txt
# CPU only
torch==2.14.1
numpy>=2.0,<3.0

# សម្រាប់ CUDA 12.8 បញ្ជាក់ index ផងដែរ៖
# --extra-index-url https://download.pytorch.org/whl/cu128
# torch==2.14.1+cu128
```

```{code-block} bash
pip install -r requirements.txt

# បង្កើត requirements.txt ពី environment បច្ចុប្បន្ន
pip freeze > requirements.txt
```

```{admonition} ហេតុអ្វីត្រូវកំណត់កំណែ?
:class: important
PyTorch ប្ដូរ API និងលំនាំដើមរវាងកំណែធំៗ។ ឧទាហរណ៍ លំនាំដើមរបស់
`torch.load(weights_only=...)` បានផ្លាស់ប្ដូរពី `False` ទៅ `True` នៅ PyTorch 2.6។
បើអ្នកមិនកំណត់កំណែទេ កូដរបស់អ្នកអាចខូចនៅពេលណាមួយដោយមិនដឹងខ្លួន។
មើល {doc}`saving-and-loading` សម្រាប់ព័ត៌មានលម្អិត។
```

## ជំហានបន្ទាប់

អ្នកបានដំឡើង PyTorch រួចរាល់ហើយ។ ឥឡូវសូមទៅ {doc}`getting-started`
ដើម្បីសរសេរកូដ PyTorch ដំបូងរបស់អ្នក។

```{admonition} Editor ណែនាំ
:class: seealso
**VS Code** ជាមួយ extension `Python` និង `Pylance` ផ្ដល់ autocomplete សម្រាប់ PyTorch API។
**Jupyter / VS Code Notebooks** មានប្រយោជន៍សម្រាប់ការពិសោធន៍ tensor ម្ដងមួយ cell។
សម្រាប់ការ debug ជ្រៅ ប្រើ debugger ពិត (breakpoints) ជាង `print()` — មើល {doc}`debugging`។
```
