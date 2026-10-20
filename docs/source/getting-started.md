# ចាប់ផ្ដើមជាមួយ PyTorch

មេរៀននេះនាំអ្នកពីសូន្យទៅដល់ **ម៉ូដែលដែលបានបណ្ដុះបណ្ដាល** ក្នុងរយៈពេលប្រហែល ១៥ នាទី។
អ្នកនឹងរៀនពីគោលគំនិតស្នូលទាំង ៣ របស់ PyTorch៖ **Tensor**, **Autograd** និង **`torch.nn`**។

```{admonition} អ្នកត្រូវការអ្វីខ្លះ?
:class: note
- Python ជាមូលដ្ឋាន (list, dict, function, class)
- PyTorch បានដំឡើង — មើល {doc}`installation`
- 🖥️ CPU គ្រប់គ្រាន់ គ្មាន GPU ក៏ដំណើរការដែរ
```

## គំនិតស្នូល៖ PyTorch ធ្វើអ្វី?

PyTorch គឺជាបណ្ណាល័យសម្រាប់សាងសង់ និងបណ្ដុះបណ្ដាល **neural networks**។
វាផ្ដល់អ្វីដែលអ្នកត្រូវការ ៣ យ៉ាង៖

```{list-table}
:header-rows: 1
:widths: 22 78

* - សមាសធាតុ
  - តួនាទី
* - **Tensor**
  - ទម្រង់ទិន្នន័យ។ ដូច NumPy array ប៉ុន្តែអាចរត់លើ GPU បាន។ អ្វីៗទាំងអស់ក្នុង PyTorch គឺជា tensor។
* - **Autograd**
  - ប្រព័ន្ធគណនា derivative ដោយស្វ័យប្រវត្តិ។ នេះជាអ្វីដែលធ្វើឱ្យ neural networks អាចរៀនបាន។
* - **`torch.nn`**
  - ជាន់ (layers) សាងសង់រួចស្រេច — `Linear`, `Conv2d`, `ReLU`, loss functions និង optimizers។
```

អ្វីដែលធ្វើឱ្យ PyTorch ពិសេសគឺ **dynamic graph**៖ អ្នកសរសេរ Python ធម្មតា
ហើយ PyTorch កត់ត្រាប្រមាណវិធីនៅពីក្រោយឆាក។ អ្នកអាចប្រើ `print()`, `pdb`,
`if/else` និង `for` loop ធម្មតាបាន — គ្មាន "compile step" ដែលត្រូវរៀនមុនទេ។

## ផ្នែកទី ១៖ Tensor គឺជាអ្វី?

**Tensor** គឺជាអារេលេខ (array of numbers) ដែលមានវិមាត្រណាមួយ៖

```{list-table}
:header-rows: 1
:widths: 26 24 50

* - ឈ្មោះ
  - វិមាត្រ (ndim)
  - ឧទាហរណ៍
* - Scalar
  - ០
  - `torch.tensor(3.14)` — លេខតែមួយ
* - Vector
  - ១
  - `torch.tensor([1., 2., 3.])` — បន្ទាត់នៃលេខ
* - Matrix
  - ២
  - តារាង (ជួរ × ឈរ)
* - 3D tensor
  - ៣
  - រូបភាពមួយ (channels × កម្ពស់ × ទទឹង)
* - 4D tensor
  - ៤
  - batch នៃរូបភាព (batch × channels × H × W)
```

Tensor ខុសពី NumPy array ត្រង់ ២ ចំណុចសំខាន់៖

1. **វាអាចរស់នៅលើ GPU បាន** — ធ្វើឱ្យការគណនាលឿនជាងច្រើនដង។
2. **វាអាចកត់ត្រា gradient បាន** — ធ្វើឱ្យការបណ្ដុះបណ្ដាល neural network អាចទៅរួច។

### សាកល្បង Tensor ដំបូងរបស់អ្នក

```{literalinclude} examples/getting_started_1.py
:language: python
:caption: getting_started_1.py — ✅ បានសាកល្បងលើ CPU (PyTorch 2.14.1)
```

Output សំខាន់ៗពីការរត់ពិត៖

```text
tensor:
tensor([[1., 2., 3.],
        [4., 5., 6.]])
shape    : (2, 3)
dtype    : torch.float32
device   : cpu
ndim     : 2
```

### ចំណុចដែលត្រូវចាំពីឧទាហរណ៍នេះ

```{list-table}
:header-rows: 1
:widths: 32 68

* - លក្ខណៈ
  - ការពន្យល់
* - `t.shape`
  - ទំហំនៃវិមាត្រនីមួយៗ។ `(2, 3)` = ២ ជួរ ៣ ឈរ។
* - `t.dtype`
  - ប្រភេទធាតុ។ លំនាំដើមគឺ `torch.float32` សម្រាប់លេខទសភាគ និង `torch.int64` សម្រាប់ចំនួនគត់។
* - `t.device`
  - ទីកន្លែងដែល tensor រស់នៅ (`cpu` ឬ `cuda:0`)។
* - `t[0, 1]`
  - ធាតុនៅជួរ ០ ឈរ ១។ Index ចាប់ផ្ដើមពី ០ ដូច Python។
* - `t.sum(dim=0)`
  - បូក **តាមជួរ** (ជួរបាត់បង់) → លទ្ធផលមាន shape `(3,)`។
* - `t.item()`
  - យកតម្លៃ Python ចេញពី scalar tensor។ ប្រើជានិច្ចសម្រាប់ logging។
```

```{admonition} dim= គឺជាចំណុចដែលច្រឡំញឹកញាប់បំផុត
:class: warning
`sum(dim=0)` មានន័យថា **"បាត់បង់វិមាត្រទី ០"** មិនមែន "បូកតាមវិមាត្រទី ០" ទេ។

សម្រាប់ tensor shape `(2, 3)`៖
- `sum(dim=0)` → shape `(3,)` (ជួរទាំង ២ ត្រូវបានបូកបញ្ចូលគ្នា)
- `sum(dim=1)` → shape `(2,)` (ឈរទាំង ៣ ត្រូវបានបូកបញ្ចូលគ្នា)

វិធីចាំ៖ **វិមាត្រដែលអ្នកបញ្ជាក់ គឺវិមាត្រដែលបាត់ពីលទ្ធផល**។
មើល {doc}`shapes-and-broadcasting` សម្រាប់ការពន្យល់លម្អិត។
```

## ផ្នែកទី ២៖ វដ្ដបណ្ដុះបណ្ដាល ៥ ជំហាន

រាល់ការបណ្ដុះបណ្ដាល neural network ក្នុង PyTorch ដើរតាមវដ្ដ ៥ ជំហានដូចគ្នា។
បើអ្នកចាំវដ្ដនេះបាន អ្នកយល់ PyTorch ៨០% ហើយ។

```{list-table}
:header-rows: 1
:widths: 8 30 62

* - ជំហាន
  - កូដ
  - អ្វីដែលកើតឡើង
* - ១
  - `optimizer.zero_grad()`
  - សម្អាត gradient ពីជុំមុន។ **បើភ្លេច gradient នឹងប្រមូលផ្ដុំ ហើយម៉ូដែលនឹងរៀនខុស។**
* - ២
  - `logits = model(x)`
  - **Forward pass** — ទិន្នន័យហូរកាត់ជាន់នីមួយៗ។ PyTorch កត់ត្រាប្រមាណវិធីទាំងអស់ទុកក្នុង graph។
* - ៣
  - `loss = loss_fn(logits, y)`
  - គណនាថាតើការទាយខុសប៉ុន្មាន។ `loss` គឺជា scalar tensor តែមួយ។
* - ៤
  - `loss.backward()`
  - **Backpropagation** — ដើរថយក្រោយតាម graph ដើម្បីគណនា gradient សម្រាប់ parameter នីមួយៗ ហើយរក្សាទុកក្នុង `.grad`។
* - ៥
  - `optimizer.step()`
  - ធ្វើបច្ចុប្បន្នភាព weight ដោយប្រើ gradient ដែលទើបគណនា។
```

### ឧទាហរណ៍ពេញលេញ

```{literalinclude} examples/getting_started_2.py
:language: python
:caption: getting_started_2.py — ✅ បានសាកល្បងលើ CPU (PyTorch 2.14.1) — accuracy > 0.9
```

### អាន output

```text
features shape : (200, 2)
labels   shape : (200,)
labels dtype   : torch.int64
--------------------------------------------------
Sequential(
  (0): Linear(in_features=2, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=2, bias=True)
)
ចំនួន parameters សរុប : 82
epoch  10 | loss = 0.0143
epoch  20 | loss = 0.0031
epoch  30 | loss = 0.0012
--------------------------------------------------
accuracy លើទិន្នន័យទាំងអស់ : 1.000
```

ចំណាំពីរយ៉ាងសំខាន់៖

1. **`labels` មាន `dtype=int64`** — `nn.CrossEntropyLoss` ត្រូវការ class index ជាចំនួនគត់
   មិនមែន one-hot vector ទេ។ នេះជាកំហុសទី ១ ដែលអ្នកចាប់ផ្ដើមជួប។
2. **ចំនួន parameters = 82** — គណនាដោយដៃ៖
   `Linear(2,16)` = 2×16 weight + 16 bias = 48 · `Linear(16,2)` = 16×2 + 2 = 34 · សរុប 82។

### `model.train()` និង `model.eval()` — កុំភ្លេច!

នៅក្នុងឧទាហរណ៍ខាងលើ អ្នកឃើញ `model.train()` នៅដើម epoch និង `model.eval()` មុនវាយតម្លៃ។
វាមិនមែនជាទម្រង់ទេ — វាផ្លាស់ប្ដូរឥរិយាបថនៃជាន់មួយចំនួនពិតប្រាកដ៖

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - ជាន់
  - ក្នុង `train()`
  - ក្នុង `eval()`
* - `nn.Dropout`
  - បិទធាតុដោយចៃដន្យ ហើយពង្រីកអ្វីដែលនៅសល់
  - មិនធ្វើអ្វីទាំងអស់ (pass-through)
* - `nn.BatchNorm1d/2d`
  - ប្រើស្ថិតិនៃ batch បច្ចុប្បន្ន ហើយធ្វើបច្ចុប្បន្នភាព `running_mean`
  - ប្រើ `running_mean`/`running_var` ដែលបានរៀន
* - ជាន់ផ្សេងទៀត (`Linear`, `ReLU` ។ល។)
  - ដូចគ្នា
  - ដូចគ្នា
```

```{admonition} ការភ្លេច model.eval() ធ្វើឱ្យលទ្ធផលវាយតម្លៃមិនត្រឹមត្រូវ
:class: danger
បើអ្នកភ្លេចហៅ `model.eval()` មុនវាយតម្លៃ Dropout នៅតែចៃដន្យបិទធាតុ
ដូច្នេះ **ការវាយតម្លៃលើទិន្នន័យដដែលពីរដង នឹងផ្ដល់លទ្ធផលខុសគ្នា**។
នេះជាមូលហេតុដែល accuracy របស់អ្នក "លោតចុះលោតឡើង" ដោយគ្មានមូលហេតុ។

វិធីចាំ៖ **ប្ដូរ mode ឱ្យត្រូវរាល់ពេលប្ដូរពីការបណ្ដុះបណ្ដាលទៅការវាយតម្លៃ។**
មើល {doc}`training-loop` សម្រាប់ឧទាហរណ៍ពិតនៃកំហុសនេះ។
```

## ផ្នែកទី ៣៖ Gradient គឺជាអ្វី?

**Gradient** គឺជាលេខដែលប្រាប់អ្នកថា៖ *"បើខ្ញុំប្ដូរ weight នេះបន្តិច
តើ loss នឹងប្រែប្រួលប៉ុន្មាន ហើយក្នុងទិសដៅណា?"*

គិតអំពីវាដូចជាការឈរលើភ្នំក្នុងអ័ព្ទ៖
- **Loss** = កម្ពស់របស់អ្នក (អ្នកចង់ទៅកន្លែងទាបបំផុត)
- **Gradient** = ទិសដៅ និងភាពចោតនៃជម្រាក់នៅជើងអ្នក
- **Optimizer** = អ្នកសម្រេចចិត្តថាត្រូវដើរទៅទិសណា និងជំហានធំប៉ុន្មាន
- **Learning rate** = ទំហំជំហាន

**Backpropagation** គ្រាន់តែជាវិធីគណនា gradient សម្រាប់ weight *រាប់លាន* ក្នុងពេលតែមួយ
ដោយដើរថយក្រោយតាម graph ពី loss ទៅ input។ PyTorch ធ្វើវាដោយស្វ័យប្រវត្តិ
នៅពេលអ្នកហៅ `loss.backward()`។

ឧទាហរណ៍សាមញ្ញបំផុត៖ បើ `y = x²` នោះ `dy/dx = 2x`។ បើ `x = 3` gradient = 6។

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)        # tensor(6.)
```

មើល {doc}`autograd` សម្រាប់ការពន្យល់ពេញលេញ។

## អ្វីបន្ទាប់?

អ្នកបានឃើញ tensor, ម៉ូដែល និងវដ្ដបណ្ដុះបណ្ដាលហើយ។ ឥឡូវជ្រមុជជ្រួត៖

```{list-table}
:header-rows: 1
:widths: 30 70

* - បើអ្នកចង់…
  - ទៅ
* - យល់ពី tensor ឱ្យស៊ីជម្រៅ
  - {doc}`tensors`
* - យល់ពី shape និង broadcasting (កំហុសទូទៅបំផុត)
  - {doc}`shapes-and-broadcasting`
* - យល់ពី gradient និង autograd
  - {doc}`autograd`
* - សាងសង់ neural networks ផ្សេងៗ
  - {doc}`neural-networks`
* - សរសេរ training loop ពេញលេញ
  - {doc}`training-loop`
* - Debug កំហុសដែលអ្នកកំពុងជួប
  - {doc}`debugging`
```

```{admonition} របៀបរៀនដែលមានប្រសិទ្ធភាព
:class: tip
**កុំអានតែមួយមុខ** — រត់កូដឧទាហរណ៍នីមួយៗដោយខ្លួនឯង។ ឯកសារនេះរចនាឡើង
ដើម្បីឱ្យអ្នកអាចចម្លងកូដទាំងមូលទៅឯកសារ `.py` ហើយរត់វាបានភ្លាមៗ។

បន្ទាប់មក **ខូចវាដោយចេតនា**៖ ប្ដូរ shape, លុប `zero_grad()`, ដូរ `dim=`។
ការឃើញសារកំហុសពិត និងការយល់ពីមូលហេតុ គឺជាវិធីរៀនលឿនបំផុត។
មើល {doc}`debugging` សម្រាប់កំហុសទូទាំង ២០+ ដែលបានរៀបចំរួចហើយ។
```
