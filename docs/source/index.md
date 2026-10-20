# ឯកសារ PyTorch ជាភាសាខ្មែរ

សៀវភៅណែនាំ PyTorch ជាភាសាខ្មែរ សម្រាប់អ្នកចាប់ផ្ដើមដំបូង រហូតដល់អ្នកសិក្សា Deep Learning កម្រិតមធ្យម។
កូដឧទាហរណ៍សរុប **៤០** — ៣៨ ត្រូវបានដំណើរការពិត និងផ្ទៀងផ្ទាត់លើ PyTorch `2.14.1+cpu`
(២ ទៀតត្រូវការ CUDA GPU ដូច្នេះត្រូវបានរំលងក្នុង environment នេះ និងសម្គាល់ច្បាស់ថាមិនទាន់បាន Run)។

```{admonition} នេះមិនមែនជាការបកប្រែឯកសារ PyTorch ផ្លូវការទាំងស្រុងទេ
:class: important
នេះគឺជាឯកសារសិក្សាដាច់ដោយឡែកជាភាសាខ្មែរ ដែលដកស្រង់ និងយោងតាម
[PyTorch Documentation](https://docs.pytorch.org/docs/stable/) (អាជ្ញាបណ្ណ CC BY 4.0)។
សម្រាប់ព័ត៌មាន API ពេញលេញ សូមមើលឯកសារផ្លូវការ។ មើល {doc}`license` សម្រាប់ Attribution។
```

## ចាប់ផ្ដើមពីទីណា?

បើអ្នក **ថ្មីទាំងស្រុង** សូមអានតាមលំដាប់៖ {doc}`installation` → {doc}`getting-started` →
{doc}`tensors` → {doc}`shapes-and-broadcasting` → {doc}`autograd` → {doc}`neural-networks`។

បើអ្នក **ចេះ Python និង NumPy រួចហើយ** សូមទៅ {doc}`tensors` ដោយផ្ទាល់
ហើយអាន {doc}`autograd` យ៉ាងយកចិត្តទុកដាក់ — នោះគឺជាគោលគំនិតសំខាន់បំផុតរបស់ PyTorch។

បើអ្នក **កំពុងជួបបញ្ហា** សូមទៅ {doc}`debugging` ដែលមានកំហុសទូទៅចំនួន ២០+
ជាមួយសារកំហុសពិត និងដំណោះស្រាយ។

## មាតិកា

```{toctree}
:caption: ការចាប់ផ្ដើម
:maxdepth: 2

installation
getting-started
```

```{toctree}
:caption: មូលដ្ឋានគ្រឹះ Tensor
:maxdepth: 2

tensors
shapes-and-broadcasting
devices
```

```{toctree}
:caption: ការគណនា Gradient
:maxdepth: 2

autograd
```

```{toctree}
:caption: សាងសង់ និងបណ្ដុះបណ្ដាលម៉ូដែល
:maxdepth: 2

neural-networks
loss-and-optimizers
data-and-dataloader
training-loop
evaluation-and-inference
```

```{toctree}
:caption: ការគ្រប់គ្រងម៉ូដែលក្នុងគម្រោងពិត
:maxdepth: 2

saving-and-loading
checkpoints
reproducibility
```

```{toctree}
:caption: កម្រិតខ្ពស់
:maxdepth: 2

performance
mixed-precision
distributed-training
```

```{toctree}
:caption: ឯកសារយោង
:maxdepth: 2

debugging
api-quick-reference
style-guide
license
```

## សញ្ញាសម្គាល់ដែលប្រើក្នុងឯកសារនេះ

```{list-table}
:header-rows: 1
:widths: 20 80

* - សញ្ញា
  - អត្ថន័យ
* - ✅ បានសាកល្បង
  - កូដត្រូវបានដំណើរការពិតដោយ `tests/run_examples.py` និងផ្ទៀងផ្ទាត់ដោយ `assert`។
* - ⚠️ មិនទាន់បានសាកល្បង
  - កូដផ្អែកលើ API ពិត ប៉ុន្តែមិនទាន់បានដំណើរការ (ឧទាហរណ៍ ត្រូវការ GPU ឬ multi-node)។
* - 🔒 ត្រូវការ hardware ជាក់លាក់
  - កូដត្រូវការ CUDA GPU ឬច្រើនម៉ាស៊ីន។ វានឹងរំលងដោយស្វ័យប្រវត្តិបើមិនមាន។
* - 🖥️ CPU គ្រប់គ្រាន់
  - ដំណើរការលើកុំព្យូទ័រធម្មតាដោយគ្មាន GPU។
```

## តារាងស្ថានភាពមាតិកា

```{list-table}
:header-rows: 1
:widths: 26 52 22

* - ទំព័រ
  - ខ្លឹមសារសំខាន់
  - កូដឧទាហរណ៍
* - {doc}`installation`
  - pip/conda, ជ្រើសរើស CPU/CUDA, venv, ការផ្ទៀងផ្ទាត់
  - ១ ✅
* - {doc}`getting-started`
  - tensor ដំបូង, ម៉ូដែលដំបូង, វដ្ដ ៥ ជំហាន
  - ២ ✅
* - {doc}`tensors`
  - ការបង្កើត, លក្ខណៈសម្បត្តិ, indexing, ប្រមាណវិធី, reshape
  - ៥ ✅
* - {doc}`shapes-and-broadcasting`
  - shape/stride, ច្បាប់ broadcasting, silent bugs
  - ៣ ✅
* - {doc}`devices`
  - CPU/GPU/MPS, `.to()`, device mismatch
  - ១ ✅
* - {doc}`autograd`
  - computational graph, backward(), gradient accumulation
  - ៣ ✅
* - {doc}`neural-networks`
  - `nn.Module`, MLP, CNN, layers ទាំងអស់
  - ៣ ✅
* - {doc}`loss-and-optimizers`
  - CrossEntropy/BCE/MSE, SGD/Adam/AdamW, schedulers
  - ៣ ✅
* - {doc}`data-and-dataloader`
  - `Dataset`, `DataLoader`, `collate_fn`, transforms
  - ៣ ✅
* - {doc}`training-loop`
  - វដ្ដពេញលេញ, early stopping, កំហុសទូទៅ
  - ២ ✅
* - {doc}`evaluation-and-inference`
  - metrics, confusion matrix, `inference_mode`, batch inference
  - ២ ✅
* - {doc}`saving-and-loading`
  - `state_dict`, TorchScript, `torch.export`, safetensors, ONNX
  - ២ ✅
* - {doc}`checkpoints`
  - resume training, RNG state, កំហុស checkpoint
  - ២ ✅
* - {doc}`reproducibility`
  - seeds, `Generator`, determinism, ដែនកំណត់
  - ១ ✅
* - {doc}`performance`
  - vectorization, dtype, contiguity, profiler
  - ១ ✅ + ១ 🔒
* - {doc}`mixed-precision`
  - `torch.amp`, float16 vs bfloat16, `GradScaler`
  - ១ ✅ + ១ 🔒
* - {doc}`distributed-training`
  - DDP, `DistributedSampler`, `torchrun`, FSDP
  - ⚠️ គ្មានឧទាហរណ៍ដែលបាន Run (ត្រូវការ multi-GPU)
* - {doc}`debugging`
  - កំហុសទូទៅ ២០+, NaN, gradient health, hooks
  - ៣ ✅
* - {doc}`api-quick-reference`
  - តារាង API យោងរហ័ស (បង្កើត tensor, ប្រមាណវិធី, nn, optim, data, amp)
  - —
* - {doc}`style-guide`
  - មគ្គុទ្ទេសក៍រចនា៖ palette, typography, ធាតុទាំងអស់ក្នុងទំព័រតែមួយ
  - ១ ✅
```

## របៀបសាកល្បងកូដឧទាហរណ៍ដោយខ្លួនឯង

កូដឧទាហរណ៍ទាំងអស់ស្ថិតនៅក្នុងឯកសារ `.py` ដាច់ដោយឡែក ហើយត្រូវបានបញ្ចូលមកទីនេះដោយ
`literalinclude`។ នេះមានន័យថា **អ្វីដែលអ្នកអាន គឺជាអ្វីដែលត្រូវបានដំណើរការពិត**។

```bash
# ដំឡើង PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy

# សាកល្បងឧទាហរណ៍ទាំងអស់
python tests/run_examples.py
```
