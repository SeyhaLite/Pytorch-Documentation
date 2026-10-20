# រក្សាទុក និងផ្ទុក Model

មានវិធីជាច្រើនក្នុងការរក្សាទុកម៉ូដែល PyTorch។ ការជ្រើសរើសខុសអាចធ្វើឱ្យ checkpoint
របស់អ្នក **មិនអាចផ្ទុកបាននៅលើម៉ាស៊ីនផ្សេង** ឬ **បើកចំហទៅនឹងការវាយប្រហារសុវត្ថិភាព**។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ២ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## ជ្រើសរើសវិធីត្រឹមត្រូវ

```{list-table}
:header-rows: 1
:widths: 24 30 46

* - គោលបំណង
  - វិធី
  - ហេតុអ្វី
* - បន្តការបណ្ដុះបណ្ដាល / fine-tune
  - `state_dict()` → `torch.save()`
  - តូច សុវត្ថិភាព មិនអាស្រ័យលើ module path
* - ចែករំលែក weight ជាសាធារណៈ
  - `safetensors`
  - មិនអាចរត់ code បាន → សុវត្ថិភាព
* - Deployment ដោយគ្មាន PyTorch
  - TorchScript ឬ `torch.export`
  - រត់ក្នុង C++ / runtime ផ្សេង
* - ប្រព័ន្ធ inference ខាងក្រៅ
  - ONNX
  - គាំទ្រដោយឧបករណ៍ជាច្រើន
* - បន្តការបណ្ដុះបណ្ដាលពីកន្លែងដែលឈប់
  - Checkpoint ពេញលេញ
  - មើល {doc}`checkpoints`
```

## ១. state_dict — វិធីណែនាំ

```{literalinclude} examples/saving_and_loading_1.py
:language: python
:caption: saving_and_loading_1.py — ✅ បានសាកល្បងលើ CPU
```

### របៀបស្តង់ដារ

```python
# រក្សាទុក
torch.save(model.state_dict(), "model_weights.pth")

# ផ្ទុក
model = MyModel()                                  # ← ត្រូវបង្កើត architecture ជាមុន
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
model.eval()                                       # ← បើសម្រាប់ inference
```

```{admonition} state_dict មានតែ weight មិនមាន architecture
:class: important
ឯកសារ `.pth` មិនដឹងអ្វីអំពី class `MyModel` របស់អ្នកទេ។ វាគ្រាន់តែជា
`OrderedDict` នៃ tensor ដែលមានឈ្មោះ។ ដូច្នេះអ្នក **ត្រូវតែមាននិយមន័យ class
ដូចគ្នា** នៅកន្លែងដែលអ្នកផ្ទុក។

នេះគឺជាលក្ខណៈល្អ — វាធ្វើឱ្យឯកសារតូច និងមិនអាស្រ័យលើ code layout។
ប៉ុន្តែវាក៏មានន័យថា បើអ្នកប្ដូរឈ្មោះ module ក្នុង class នោះ key ទាំងអស់នឹងខុស។
```

### កុំរក្សាទុក model ទាំងមូល

```python
# ❌ កុំធ្វើបែបនេះ
torch.save(model, "model_whole.pth")

# ✅ ធ្វើបែបនេះ
torch.save(model.state_dict(), "model_weights.pth")
```

ការរក្សាទុក model ទាំងមូលប្រើ Python `pickle` ដែលកត់ត្រា **module path** ពេញលេញ។
បើអ្នកផ្លាស់ទីឯកសារ ឬប្ដូរឈ្មោះ class/file ការផ្ទុកនឹងបរាជ័យ៖

```text
ModuleNotFoundError: No module named 'myproject.models'
```

ហើយវាក៏ជា **ហានិភ័យសុវត្ថិភាព** ផងដែរ — មើលផ្នែកខាងក្រោម។

## ២. weights_only និងសុវត្ថិភាព

````{admonition} ឯកសារ .pth អាចរត់ code បាន
:class: danger
`torch.save()` ប្រើ `pickle`។ Pickle អាចបង្កប់ **code ដែលរត់នៅពេលផ្ទុក**។
ឯកសារ `.pth` ពីប្រភពដែលអ្នកមិនទុកចិត្តអាចលួច credential ឬដំឡើង malware
នៅពេលអ្នកហៅ `torch.load()`។

ចាប់ពី **PyTorch 2.6** លំនាំដើមរបស់ `torch.load()` គឺ `weights_only=True`
ដែលអនុញ្ញាតតែ tensor និងប្រភេទសុវត្ថិភាពមូលដ្ឋាន។ បើអ្នកត្រូវការផ្ទុក
object ផ្សេង (str, dict ផ្ទាល់ខ្លួន) អ្នកត្រូវបញ្ជាក់ `weights_only=False`
ដោយដឹងខ្លួន — ហើយតែជាមួយឯកសារដែលអ្នកទុកចិត្តបានប៉ុណ្ណោះ។

```python
# សុវត្ថិភាព — តែ tensor
sd = torch.load("model.pth", weights_only=True)

# មានហានិភ័យ — អនុញ្ញាត object ណាក៏បាន
data = torch.load("unknown_file.pth", weights_only=False)   # ⚠️
```

សម្រាប់ឯកសារពីប្រភពខាងក្រៅ ប្រើ **safetensors** ជំនួសវិញ។
````

## ៣. ការផ្ទុកទៅ device ផ្សេង

```python
# រក្សាទុកលើ GPU ផ្ទុកលើ CPU
sd = torch.load(path, map_location="cpu", weights_only=True)

# ឬផ្ទុកទៅ device គោលដៅដោយផ្ទាល់
model = MyModel().to(device)
model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
```

បើអ្នកភ្លេច `map_location` ហើយ checkpoint ត្រូវបានរក្សាទុកលើ `cuda:0`
ការផ្ទុកលើម៉ាស៊ីនគ្មាន GPU នឹងបរាជ័យ៖

```text
RuntimeError: Attempting to deserialize object on a CUDA device but
torch.cuda.is_available() is False
```

## ៤. strict=False និង Transfer Learning

```python
result = model.load_state_dict(sd, strict=False)
print(result.missing_keys)      # key ដែលមានក្នុង model ប៉ុន្តែគ្មានក្នុង checkpoint
print(result.unexpected_keys)   # key ដែលមានក្នុង checkpoint ប៉ុន្តែគ្មានក្នុង model
```

````{admonition} strict=False មិនអត់ឱនឱ្យ shape mismatch ទេ
:class: warning
វាអត់ឱនឱ្យតែ **key ដែលបាត់ ឬលើស** ប៉ុណ្ណោះ។ បើ tensor មានឈ្មោះដូចគ្នា
ប៉ុន្តែ shape ខុសគ្នា វានៅតែបរាជ័យ៖

```text
RuntimeError: Error(s) in loading state_dict for Net:
	size mismatch for fc1.weight: copying a param with shape torch.Size([16, 8])
	from checkpoint, the shape in current model is torch.Size([32, 8]).
```

ដំណោះស្រាយ — តម្រង weight ដែល shape ត្រូវគ្នាដោយដៃ៖

```python
target_sd = model.state_dict()
compatible = {k: v for k, v in sd.items()
              if k in target_sd and v.shape == target_sd[k].shape}
model.load_state_dict(compatible, strict=False)
```

បច្ចេកទេសនេះមានប្រយោជន៍សម្រាប់ fine-tuning backbone ជាមួយ head ថ្មី។
````

## ៥. ទម្រង់ផ្សេងទៀត

```{literalinclude} examples/saving_and_loading_2.py
:language: python
:caption: saving_and_loading_2.py — ✅ បានសាកល្បងលើ CPU (TorchScript, torch.export, safetensors, ONNX)
```

### TorchScript

```python
# វិធី ១: script — វិភាគ Python code
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

# វិធី ២: trace — កត់ត្រាប្រតិបត្តិការជាក់ស្ដែង
traced = torch.jit.trace(model, example_input)
traced.save("model_traced.pt")

# ផ្ទុកដោយគ្មាននិយមន័យ class
reloaded = torch.jit.load("model_scripted.pt")
```

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - វិធី
  - គាំទ្រ
  - ដែនកំណត់
* - `torch.jit.script`
  - control flow ដែលអាស្រ័យលើ tensor, ច្រើនផ្លូវ
  - Python subset តែប៉ុណ្ណោះ។ កូដខ្លះមិនអាច script បាន
* - `torch.jit.trace`
  - ម៉ូដែលលីនេអ៊ែរសាមញ្ញ
  - **បាត់បង់ control flow** — `if`/`for` ត្រូវបានកត់ត្រាតែផ្លូវដែលបានរត់
```

```{admonition} torch.jit.trace បាត់បង់ control flow
:class: danger
បើ `forward()` របស់អ្នកមាន `if x.shape[0] > 1:` ការ trace ជាមួយ batch ទំហំ ៨
នឹងកត់ត្រាតែផ្លូវ `True`។ នៅពេល inference ជាមួយ batch ទំហំ ១ ម៉ូដែល
នឹងដើរផ្លូវខុស **ដោយគ្មាន error**។

ប្រើ `script` បើមាន control flow។ ប្រើ `trace` តែពេលម៉ូដែលគ្មាន branching។
```

### torch.export (PyTorch 2.x)

```python
ep = torch.export.export(model, (example_input,))
torch.export.save(ep, "model.pt2")

ep2 = torch.export.load("model.pt2")
out = ep2.module()(example_input)
```

`torch.export` ផលិត **ExportedProgram** ដែលមាន graph ពេញលេញ រួមទាំង signature។
វាជាមូលដ្ឋានសម្រាប់ `torch.compile`, edge deployment និង compiler ផ្សេងៗ។

### safetensors

```python
from safetensors.torch import save_file, load_file

save_file(model.state_dict(), "model.safetensors")
sd = load_file("model.safetensors")
```

```{list-table}
:header-rows: 1
:widths: 26 37 37

* - លក្ខណៈ
  - `.pth` (pickle)
  - `.safetensors`
* - អាចរត់ code ពេលផ្ទុក
  - ⚠️ បាន
  - ❌ ទេ — សុវត្ថិភាព
* - Lazy loading (mmap)
  - ❌
  - ✅
* - រក្សាទុក metadata
  - តាមរយៈ dict
  - ✅ មាន header ជាក់លាក់
* - គាំទ្រ object មិនមែន tensor
  - ✅
  - ❌ tensor តែប៉ុណ្ណោះ
```

ត្រូវដំឡើងដោយឡែក៖ `pip install safetensors`។

### ONNX

```python
torch.onnx.export(
    model,
    (example_input,),
    "model.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    dynamo=False,
)
```

`dynamic_axes` សំខាន់ណាស់ — បើអ្នកមិនបញ្ជាក់ទេ ONNX model នឹងជាប់នឹង
batch size ជាក់លាក់ដែលអ្នកបានប្រើពេល export។

## ៦. រក្សាទុក tensor និង metadata

```python
payload = {
    "embeddings": torch.randn(10, 4),
    "label_names": ["cat", "dog", "bird"],       # ← Python list មិនមែន tensor
    "label_to_id": {"cat": 0, "dog": 1, "bird": 2},
    "config": {"dim": 4, "n": 10},
    "epoch": 7,
    "best_val_loss": 0.4231,
}
torch.save(payload, "artifacts.pt")
restored = torch.load("artifacts.pt", weights_only=False)
```

````{admonition} Tensor មិនអាចផ្ទុក string បានទេ
:class: danger
```python
torch.tensor(["cat", "dog", "bird"])
# ValueError: too many dimensions 'str'
```

Tensor គឺជាអារេលេខដែលមានប្រភេទតែមួយ។ សម្រាប់ឈ្មោះថ្នាក់ សូមប្រើ
**Python list** ឬ **dict** ហើយ map ទៅ index។ នេះជាមូលហេតុដែល
`torch.load()` ត្រូវការ `weights_only=False` សម្រាប់ payload បែបនេះ។
````

## ៧. ការសរសេរឯកសារដោយសុវត្ថិភាព (atomic write)

បើកម្មវិធីត្រូវបានសម្លាប់ ឬ disk ពេញកំឡុងពេលសរសេរ ឯកសារនឹងខូច។
សរសេរទៅឯកសារបណ្ដោះអាសន្ន ហើយប្ដូរឈ្មោះ (rename គឺ atomic លើ POSIX)៖

```python
import os
from pathlib import Path

def safe_save(payload, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic — ឯកសារចាស់នៅដដែលរហូតដល់បញ្ចប់
```

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`
  - checkpoint ពី GPU ផ្ទុកលើម៉ាស៊ីនគ្មាន GPU
  - `map_location="cpu"`
* - `ModuleNotFoundError: No module named '...'`
  - បានរក្សាទុក model ទាំងមូល (pickle)
  - រក្សាទុក `state_dict()` វិញ
* - `Error(s) in loading state_dict ... Missing key(s)`
  - architecture ផ្លាស់ប្ដូរ
  - `strict=False` ឬកែ architecture ឱ្យត្រូវ
* - `Error(s) in loading state_dict ... size mismatch`
  - shape ខុសគ្នា
  - តម្រង weight ដែល shape ត្រូវគ្នា (មើលខាងលើ)
* - `UnpicklingError: invalid load key`
  - ឯកសារខូច ឬមិនពេញលេញ
  - សរសេរឡើងវិញដោយ atomic write
* - `weights_only=True` បរាជ័យលើ checkpoint ចាស់
  - checkpoint មាន object មិនមែន tensor
  - ប្រើ `weights_only=False` **តែ** បើទុកចិត្តបាន
* - `Unexpected key(s) ... module.`
  - រក្សាទុកពី `DataParallel`
  - រក្សាទុក `model.module.state_dict()`
* - ឯកសារធំខ្លាំង
  - រក្សាទុក optimizer state ដែលមិនចាំបាច់
  - រក្សាទុកតែ `state_dict` សម្រាប់ inference
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **រក្សាទុក `state_dict()` មិនមែន model ទាំងមូល**។
2. **បង្កើត architecture ជាមុន** បន្ទាប់មក `load_state_dict()`។
3. **`weights_only=True`** ជាលំនាំដើមចាប់ពី PyTorch 2.6 — រក្សាវាទុក។
4. **`map_location="cpu"`** ពេលផ្ទុក GPU checkpoint លើម៉ាស៊ីនគ្មាន GPU។
5. **`model.eval()`** បន្ទាប់ពីផ្ទុក សម្រាប់ inference។
6. **`strict=False` មិនអត់ឱនឱ្យ shape mismatch** — អត់ឱនឱ្យតែ key បាត់/លើស។
7. **safetensors សម្រាប់ចែករំលែកសាធារណៈ** — មិនអាចរត់ code។
8. **Atomic write** (`os.replace`) ដើម្បីការពារឯកសារខូច។
```

## មេរៀនបន្ទាប់

{doc}`checkpoints` — រក្សាទុកស្ថានភាពបណ្ដុះបណ្ដាលទាំងមូល ដើម្បីអាចបន្តបាន។
