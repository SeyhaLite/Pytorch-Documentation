# មគ្គុទ្ទេសក៍រចនា (Style Guide)

ទំព័រនេះបង្ហាញ **ធាតុរចនាទាំងអស់** ដែលឯកសារនេះប្រើ ក្នុងទីតាំងតែមួយ។
វាមានប្រយោជន៍ពីរយ៉ាង៖

1. ជា **ឯកសារយោងរូបភាព** សម្រាប់អ្នកនិពន្ធ — មុនបន្ថែមធាតុថ្មី សូមមើលថាតើ
   វាត្រូវបានគាំទ្រដោយ palette ដែរឬទេ។
2. ជា **គោលដៅសម្រាប់ offline preview** — `tools/make_preview.py` សាងសង់
   ឯកសារ HTML តែមួយពីទំព័រនេះ ដើម្បីឱ្យអាចត្រួតពិនិត្ថ contrast, dark/light mode
   និង responsive layout បានដោយគ្មាន server។

````{admonition} របៀបបង្កើត preview ឡើងវិញ
:class: note
```bash
sphinx-build -b html docs/source docs/_build/html -q -W --keep-going
python tools/make_preview.py --page style-guide.html
python tools/check_contrast.py          # ផ្ទៀងផ្ទាត់ contrast ១០៤ គូ
```
````

## ១. Palette និង Contrast

ពណ៌ទាំងអស់ត្រូវបានកំណត់ជា **CSS custom properties** នៅក្នុង
`_static/css/custom.css` លើ `html[data-theme="dark"]` និង `html[data-theme="light"]`។
កុំ hardcode ពណ៌ក្នុង selector — កែតម្លៃ token វិញ។

រាល់គូ text/background ទាំង **១០៤** ត្រូវបានផ្ទៀងផ្ទាត់ដោយ
`tools/check_contrast.py` ថាឈានដល់ **≥ 4.5:1** (WCAG AA)។

```{list-table} Design tokens — dark mode
:header-rows: 1
:widths: 30 18 52

* - Token
  - តម្លៃ
  - ប្រើសម្រាប់
* - `--kh-page-bg`
  - `#14161a`
  - ផ្ទៃខាងក្រោយទំព័រ
* - `--kh-sidebar-bg`
  - `#1f2328`
  - ផ្ទៃខាងក្រោយ sidebar
* - `--kh-card-bg`
  - `#1c2027`
  - ក្រឡាតារាងសេស, code blocks, cards
* - `--kh-row-alt-bg`
  - `#171a1e`
  - ក្រឡាតារាងគូ (striping)
* - `--kh-chip-bg`
  - `#262b33`
  - ផ្ទៃខាងក្រោយ inline code / kbd / samp
* - `--kh-text`
  - `#d9dde4`
  - អត្ថបទខ្លួន (13.3:1 លើ page bg)
* - `--kh-heading`
  - `#eef1f6`
  - ចំណងជើង (16.0:1)
* - `--kh-muted`
  - `#8b949e`
  - អត្ថបទបន្ទាប់បញ្ចប់, caption (5.3:1 លើ card)
* - `--kh-border`
  - `#343b45`
  - បន្ទាត់ព្រំដែនចម្បង
* - `--kh-link`
  - `#79b8ff`
  - តំណ (8.7:1)
* - `--kh-link-visited`
  - `#b392f0`
  - តំណដែលបានទស្សនា (7.2:1)
```

```{list-table} Design tokens — light mode
:header-rows: 1
:widths: 30 18 52

* - Token
  - តម្លៃ
  - ប្រយោជន៍
* - `--kh-page-bg`
  - `#ffffff`
  - ផ្ទៃខាងក្រោយទំព័រ
* - `--kh-sidebar-bg`
  - `#f3f4f6`
  - ផ្ទៃខាងក្រោយ sidebar
* - `--kh-card-bg`
  - `#f6f8fa`
  - cards, code blocks
* - `--kh-text`
  - `#24292f`
  - អត្ថបទខ្លួន (14.7:1)
* - `--kh-heading`
  - `#1f2328`
  - ចំណងជើង (15.8:1)
* - `--kh-muted`
  - `#57606a`
  - អត្ថបទបន្ទាប់បញ្ចប់ (6.0:1)
* - `--kh-link`
  - `#0b5cad`
  - តំណ (6.7:1)
```

## ២. Typography និងភាសាខ្មែរ

### ពុម្ពអក្សរ

```{list-table} Font stack
:header-rows: 1
:widths: 34 66

* - គោលបំណង
  - Stack
* - អត្ថបទ
  - `Kantumruy Pro`, `Noto Sans Khmer`, `Khmer OS`, `Khmer OS System`, `Noto Serif Khmer`, `Leelawadee UI`, `Khmer OS Battambang`, system-ui, sans-serif
* - កូដ
  - `JetBrains Mono`, `SFMono-Regular`, `Menlo`, `Consolas`, `Liberation Mono`, `Courier New`, បន្ទាប់មកពុម្ពអក្សរខ្មែរ, monospace
```

ពុម្ពអក្សរខ្មែរត្រូវបានផ្ទុកពី Google Fonts CDN ដោយ `@import`។
**Fallback stack ធានាថាអត្ថបទនៅតែអានបាន** បើ CDN មិនអាចទាក់ទងបាន។

### ចំណងជើង

# ចំណងជើងកម្រិត ១ (H1)

## ចំណងជើងកម្រិត ២ (H2)

### ចំណងជើងកម្រិត ៣ (H3)

#### ចំណងជើងកម្រិត ៤ (H4)

អត្ថបទធម្មតាជាភាសាខ្មែរ។ ភាសាខ្មែរមានសញ្ញាខ្ពស់ និងទាបច្រើន ដូច្នេះ
`line-height` ត្រូវបានកំណត់ **1.85–1.9** ដើម្បីកុំឱ្យសញ្ញាជាន់គ្នា។
English text renders in the same stack with the Latin faces first.
លេខ ០ ១ ២ ៣ ៤ ៥ ៦ ៧ ៨ ៩ និងសញ្ញា ៖ ។ ៕ ៘ ៙។

អត្ថបទ **ដិត** និងអត្ថបទ *ទ្រេត* និងអត្ថបទ ***ដិតផងទ្រេតផង***។

## ៣. ធាតុ Inline

### Inline code

អត្ថបទធម្មតា ជាមួយ `torch.nn.Linear` និង `loss.backward()` និង `requires_grad=True` នៅក្នុងប្រយោគ។
តំណដែលមាន inline code៖ [មើល {doc}`autograd`](autograd)។

### Keyboard chips

ចុច {kbd}`Ctrl` + {kbd}`C` ដើម្បីចម្លង · {kbd}`Ctrl` + {kbd}`Shift` + {kbd}`P` សម្រាប់ command palette ·
{kbd}`⌘` + {kbd}`K` លើ macOS · {kbd}`F1` សម្រាប់ជំនួយ។

### GUI labels, file paths, samples

ជ្រើសរើស {guilabel}`File` → {guilabel}`Preferences` → {guilabel}`Theme`។
ឯកសារកំណត់រចនាសម្ព័ន្ធស្ថិតនៅ {file}`docs/source/conf.py` ហើយលទ្ធផលនៅ {file}`docs/_build/html/index.html`។
ចុច {guilabel}`Build` បន្ទាប់មកមើលសារ {samp}`Build finished with {1} warning(s)`។

### តំណ

- តំណខាងក្នុង៖ {doc}`tensors`
- តំណខាងក្រៅ៖ [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)
- តំណដែលមាន inline code៖ [`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html)

### Mark និង highlighted search hits

ពាក្យដែលបានស្វែងរកត្រូវបានបញ្លិចដូច្នេះ៖ ==អត្ថបទដែលបានបញ្លិច== និង <mark>ពាក្យគន្លឹះ</mark> ។

នៅលើទំព័រស្វែងរក លទ្ធផលដែលត្រូវគ្នាប្រើ <span class="highlighted">class="highlighted"</span> ជំនួសវិញ។

## ៤. Code Blocks

### Python ជាមួយ Pygments highlighting

```python
import torch
import torch.nn as nn

# នេះជាមតិយោបល់ជាភាសាខ្មែរ — Pygments comment token
class MLP(nn.Module):
    """Multi-Layer Perceptron សាមញ្ញ។"""

    def __init__(self, in_features: int, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)   # 48 parameters
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.head(x)


torch.manual_seed(42)
model = MLP(in_features=784)
logits = model(torch.randn(32, 784))          # shape (32, 10)
print(f"logits: {tuple(logits.shape)}, dtype={logits.dtype}")
loss = nn.functional.cross_entropy(logits, torch.randint(0, 10, (32,)))
loss.backward()
print(f"loss = {loss.item():.4f}  # លេខទសភាគ 1.234e-05")
```

### Console / bash

```bash
# ដំឡើង PyTorch CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
python tests/run_examples.py            # រត់ឧទាហរណ៍ទាំង ៤០
sphinx-build -b html docs/source docs/_build/html -q -W --keep-going
echo "exit=$?"
```

### លទ្ធផល (text)

```text
==========================================================
PyTorch Environment Check
==========================================================
Python        : 3.11.2
PyTorch       : 2.14.1+cpu
CUDA available: False
----------------------------------------------------------
OK — PyTorch ដំណើរការត្រឹមត្រូវ
```

### Diff

```diff
- optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
+ optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
  for epoch in range(epochs):
+     scheduler.step(val_loss)
```

### ឧទាហរណ៍ពិតពីឯកសារ (literalinclude)

```{literalinclude} examples/getting_started_1.py
:language: python
:caption: getting_started_1.py — ✅ បានសាកល្បងលើ CPU
:lines: 1-40
```

## ៥. តារាង

### តារាងធម្មតា ជាមួយ striping

```{list-table} ការប្រៀបធៀប dtype
:header-rows: 1
:widths: 24 16 16 44

* - dtype
  - Bytes
  - ដែនអតិបរមា
  - ប្រើសម្រាប់
* - `torch.float64`
  - 8
  - 1.8e308
  - ភាពជាក់លាក់ខ្ពស់, `gradcheck`
* - `torch.float32`
  - 4
  - 3.4e38
  - លំនាំដើមសម្រាប់ការបណ្ដុះបណ្ដាល
* - `torch.bfloat16`
  - 2
  - 3.4e38
  - mixed precision លើ Ampere+
* - `torch.float16`
  - 2
  - 65504
  - mixed precision (ត្រូវការ `GradScaler`)
* - `torch.int8`
  - 1
  - 127
  - quantized inference
```

### តារាងវែង (១០ ជួរ) ដើម្បីមើល striping

```{list-table} Loss functions
:header-rows: 1
:widths: 34 22 44

* - API
  - Target
  - ប្រើសម្រាប់
* - `nn.CrossEntropyLoss`
  - int64
  - multi-class (logits)
* - `nn.BCEWithLogitsLoss`
  - float32
  - binary / multi-label
* - `nn.MSELoss`
  - float32
  - regression (L2)
* - `nn.L1Loss`
  - float32
  - regression (MAE)
* - `nn.SmoothL1Loss`
  - float32
  - regression (Huber)
* - `nn.NLLLoss`
  - int64
  - ជាមួយ `LogSoftmax` ដោយដៃ
* - `nn.KLDivLoss`
  - log-probs
  - distribution matching
* - `nn.PoissonNLLLoss`
  - float32
  - count data
* - `nn.CosineEmbeddingLoss`
  - −1 / 1
  - similarity
* - `nn.TripletMarginLoss`
  - float32
  - metric learning
```

### តារាងពីរឈរ (field-list រចនាបថ)

```{eval-rst}
:ប្រភព: `pytorch/pytorch/docs/source <https://github.com/pytorch/pytorch/tree/main/docs/source>`_
:អាជ្ញាបណ្ណខ្លឹមសារ: CC BY 4.0
:អាជ្ញាបណ្ណកូដ: BSD 3-Clause
:កំណែ PyTorch ដែលបានផ្ទៀងផ្ទាត់: 2.14.1+cpu
```

## ៦. Admonitions (គ្រប់ប្រភេទ)

```{note}
**note** — ព័ត៌មានបន្ថែមដែលមានប្រយោជន៍។ អត្ថបទធម្មតាជាភាសាខ្មែរ
ជាមួយ `inline code` និង [តំណ](https://pytorch.org)។
```

```{tip}
**tip** — ដំបូន្មានជាក់ស្ដែង។ ប្រើ `optimizer.zero_grad(set_to_none=True)`
ដើម្បីសន្សំ memory។
```

```{important}
**important** — ចំណុចដែលអ្នកមិនគួររំលង។ `nn.CrossEntropyLoss` ទទួល **logits**
មិនមែន probabilities ទេ។
```

```{warning}
**warning** — អ្វីដែលអាចបង្កបញ្ហា។ កុំហៅ `model.forward(x)` ដោយផ្ទាល់។
```

```{caution}
**caution** — ប្រយ័ត្ន។ `squeeze()` ដកវិមាត្រទំហំ ១ **ទាំងអស់** ចេញ។
```

```{attention}
**attention** — ចាប់អារម្មណ៍។ `inference_mode` tensor មិនអាចប្រើក្នុង autograd បានទេ។
```

```{danger}
**danger** — គ្រោះថ្នាក់។ ឯកសារ `.pth` អាចរត់ code បាន — ប្រើ `weights_only=True`។
```

```{error}
**error** — កំហុស។ `RuntimeError: mat1 and mat2 shapes cannot be multiplied`។
```

```{seealso}
**seealso** — មើលបន្ថែម។ [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)។
```

### Admonition ជាមួយ code block ខាងក្នុង

````{admonition} ឧទាហរណ៍ដែលមាន nested code
:class: tip
Admonition នេះប្រើ fence ៤ backticks ព្រោះវាមាន code block ខាងក្នុង។
បើប្រើ ៣ ទំព័រទាំងមូលនឹងខូច — មើល `CONTRIBUTING.md` ច្បាប់ទី ៦។

```python
x = torch.randn(4, 3)
print(x.mean(dim=1, keepdim=True).shape)   # torch.Size([4, 1])
```

អត្ថបទបន្ទាប់ពី code block នៅតែ render ត្រឹមត្រូវ។
````

### Admonition ជាមួយតារាងខាងក្នុង

````{admonition} ឧទាហរណ៍ដែលមាន nested table
:class: note
```{list-table}
:header-rows: 1

* - ធាតុ
  - តម្លៃ
* - `a`
  - ១
* - `b`
  - ២
```
````

## ៧. Blockquote និង Epigraph

> នេះគឺជា blockquote។ វាមានផ្ទៃខាងក្រោយ card និងបន្ទាត់ឆ្វេង។
> បន្ទាត់ទីពីរនៃ blockquote ដូចគ្នា។
>
> — ការដកស្រង់ជាមួយប្រភព

## ៘. បញ្ជី

### បញ្ជីមានលេខ

1. ធាតុទីមួយ — ជាមួយ `inline code`
2. ធាតុទីពីរ
   1. ធាតុរង
   2. ធាតុរងទីពីរ
3. ធាតុទីបី

### បញ្ជីជាចំណុច

- ធាតុទីមួយ
- ធាតុទីពីរ ជាមួយ [តំណ](https://pytorch.org) និង **អត្ថបទដិត**
  - ធាតុរងជ្រៅ
- ធាតុទីបី

### Definition list

`Tensor`
: អារេពហុវិមាត្រនៃលេខ ដែលមានប្រភេទតែមួយ។ អាចរស់នៅលើ CPU ឬ GPU។

`Gradient`
: Derivative ភាគយកនៃ loss ធៀបនឹង parameter។ ប្រាប់ទិសដៅ និងទំហំនៃការផ្លាស់ប្ដូរ។

`Autograd`
: ប្រព័ន្ធ automatic differentiation របស់ PyTorch។

### Task list

- [x] បង្កើត palette dark + light
- [x] ផ្ទៀងផ្ទាត់ contrast ១០៤ គូ
- [x] សាកល្បងឧទាហរណ៍ ៤០
- [ ] បន្ថែមរូបភាពពន្យល់ broadcasting

## ៙. Footnotes

អត្ថបទដែលមាន footnote នៅទីនេះ[^1] និងមួយទៀត[^note]។

[^1]: នេះគឺជា footnote លេខ។

[^note]: នេះគឺជា footnote ដែលមានឈ្មោះ។ វាអាចមានអត្ថបទវែង
    និងច្រើនបន្ទាត់។

## ៚. Horizontal rule និងការបំបែកផ្នែក

អត្ថបទមុន horizontal rule។

---

អត្ថបទបន្ទាប់ពី horizontal rule។

## ៛. Sidebar និង Navigation

Sidebar (នៅខាងឆ្វេងលើ desktop, ឬបើកដោយប៊ូតុង hamburger លើទូរស័ព្ទ) មាន៖

- ចំណងជើងគម្រោង និងកំណែ
- ប៊ូតុង **ពណ៌ងងឹត / ពណ៌ភ្លឺ** (ធាតុអត្ថបទធម្មតាតែមួយ គ្មាន icon គ្មាន border)
- ប្រអប់ស្វែងរក
- មាតិកាដែលមាន caption groups និង nested items ដល់កម្រិត ៤
- Flyout កំណែ នៅផ្នែកខាងក្រោមឆ្វេង

ប៊ូតុង **មុន / បន្ទាប់** ស្ថិតនៅផ្នែកខាងក្រោមនៃទំព័រ។
Breadcrumbs ស្ថិតនៅផ្នែកខាងលើ ជាមួយតំណ "Edit on GitHub" នៅខាងស្ដាំ។

## ៜ. ការឆ្លើយតប (Responsive)

```{list-table} Breakpoints
:header-rows: 1
:widths: 30 70

* - ទំហំអេក្រង់
  - ឥរិយាបថ
* - `≥ 1400px`
  - មាតិកាអតិបរមា 1100px
* - `769px – 1399px`
  - មាតិកាអតិបរមា 1000px, sidebar នៅជាប់
* - `≤ 768px`
  - sidebar ក្លាយជា overlay, តារាងវែង scroll តាមផ្ដេក, អក្សរតូចជាងបន្តិច, ប៊ូតុងចម្លងបង្ហាញជានិច្ច (គ្មាន hover លើ touch)
```

សាកល្បងដោយប្ដូរទំហំបង្អួច ឬប្រើ developer tools device mode។

## ៝. ភាពអាចចូលដំណើរការបាន (Accessibility)

```{list-table}
:header-rows: 1
:widths: 34 66

* - លក្ខណៈ
  - ការអនុវត្ត
* - Contrast
  - ≥ 4.5:1 សម្រាប់គ្រប់គូ text/background (ផ្ទៀងផ្ទាត់ដោយ `tools/check_contrast.py`)
* - Focus ring
  - `outline: 2px solid var(--kh-link)` លើ `:focus-visible` សម្រាប់តំណ ប៊ូតុង និង input
* - Keyboard navigation
  - ប៊ូតុងទាំងអស់ជា `<button>` ពិត មាន `type`, `aria-label` និង `aria-pressed`
* - Reduced motion
  - `@media (prefers-reduced-motion: reduce)` បិទ transitions ទាំងអស់
* - Touch devices
  - `@media (hover: none)` បង្ហាញប៊ូតុងចម្លងជានិច្ច (គ្មាន hover state)
* - ភាសា
  - `language = "km"` ក្នុង `conf.py` → `<html lang="km">`
* - ពុម្ពអក្សរ
  - Fallback stack ធានាអត្ថបទអានបានបើ CDN មិនអាចទាក់ទង
* - Print
  - `css/print.css` ផ្ដល់ពណ៌ភ្លឺលើក្រដាស លាក់ navigation និងបង្ហាញ URL តំណ
```

## ៞. ធាតុដែលគេប្រើក្នុងឯកសារនេះ

### ឧទាហរណ៍កូដខ្លី

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)          # torch.Size([2, 2])
print(x @ x.T)          # matrix multiply
```

```python
# កូដជាមួយមតិយោបល់ខ្មែរ និងលេខ
for i in range(3):
    loss = (i * 0.5) ** 2   # 0.0, 0.25, 1.0
    print(f"step {i}: {loss:.3f}")
```

### លេខ និងសញ្ញាពិសេស

`0`, `1.5`, `-2.3e-4`, `1e38`, `True`, `False`, `None`, `inf`, `nan`,
`≤`, `≥`, `≠`, `≈`, `×`, `→`, `←`, `·`, `—`, `–`, `…`
