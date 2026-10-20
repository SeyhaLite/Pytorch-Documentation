# Neural Networks និង torch.nn

`torch.nn` ផ្ដល់ជាន់ (layers) សាងសង់រួចស្រេចសម្រាប់ neural networks។ មេរៀននេះបង្ហាញ
របៀបសាងសង់ម៉ូដែលពី MLP សាមញ្ញ រហូតដល់ CNN និង Transformer blocks។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ៣ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## nn.Module — មូលដ្ឋានគ្រឹះ

ម៉ូដែល PyTorch ទាំងអស់ស្នងពី `nn.Module`។ អ្នកត្រូវការតែពីរអនុគមន៍៖

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()            # ⚠️ ចាំបាច់ជានិច្ច — ចុះបញ្ជី parameter
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)
```

````{admonition} super().__init__() គឺជាចាំបាច់ មិនមែនជាទម្រង់ទេ
:class: danger
បើអ្នកភ្លេចវា `nn.Module` មិនបានរៀបចំ internal registry របស់វា។ ការផ្ដល់
`self.fc = nn.Linear(...)` នឹងបរាជ័យ៖

```text
RuntimeError: cannot assign module before Module.__init__() call
```

ហើយបើអ្នកធ្វើវាតាមលំដាប់ខុស (assign មុន `super().__init__()`) ក៏បរាជ័យដែរ។
**សរសេរ `super().__init__()` ជាបន្ទាត់ដំបូងនៃ `__init__` ជានិច្ច។**
````

### ហៅ model(x) មិនមែន model.forward(x)

```python
out = model(x)              # ✓ ត្រឹមត្រូវ
out = model.forward(x)      # ⚠️ ដំណើរការ ប៉ុន្តែរំលង hooks
```

`nn.Module.__call__()` រត់ forward hooks, backward hooks និង logic ផ្សេងទៀត
មុនហៅ `forward()`។ បើអ្នកហៅ `forward()` ដោយផ្ទាល់ hooks ទាំងនោះមិនដំណើរការទេ
ហើយ `torch.compile`, profiler និងឧបករណ៍ជាច្រើននឹងខូច។

## ១. MLP (Multi-Layer Perceptron)

```{literalinclude} examples/neural_networks_1.py
:language: python
:caption: neural_networks_1.py — ✅ បានសាកល្បងលើ CPU
```

### វិធីសរសេរពីរ

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - វិធី
  - សមស្របសម្រាប់
  - ដែនកំណត់
* - `nn.Sequential(...)`
  - ម៉ូដែលលីនេអ៊ែរសាមញ្ញ គ្មាន branching
  - មិនអាចមាន skip connection, ពហុ output ឬ logic ក្នុង forward
* - class ផ្ទាល់ខ្លួន
  - អ្វីៗទាំងអស់ផ្សេងទៀត
  - កូដច្រើនជាងបន្តិច
```

### ការរាប់ parameters

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

សម្រាប់ `nn.Linear(in, out)`៖ ចំនួន parameters = `in × out + out` (weight + bias)។

```{list-table} ការគណនាពីឧទាហរណ៍ខាងលើ
:header-rows: 1
:widths: 30 34 18 18

* - ជាន់
  - ការគណនា
  - weight
  - សរុប
* - `Linear(784, 128)`
  - 784 × 128 + 128
  - 100,352
  - 100,480
* - `Linear(128, 64)`
  - 128 × 64 + 64
  - 8,192
  - 8,256
* - `Linear(64, 10)`
  - 64 × 10 + 10
  - 640
  - 650
* - **សរុប**
  - —
  - —
  - **109,386**
```

## ២. CNN សម្រាប់រូបភាព

```{literalinclude} examples/neural_networks_2.py
:language: python
:caption: neural_networks_2.py — ✅ បានសាកល្បងលើ CPU (រួមទាំងការផ្ទៀងផ្ទាត់រូបមន្ត conv)
```

### រូបមន្ត Conv2d output size

$$\text{out} = \left\lfloor \frac{\text{in} + 2 \cdot \text{padding} - \text{dilation} \cdot (\text{kernel} - 1) - 1}{\text{stride}} + 1 \right\rfloor$$

```{list-table} ករណីដែលប្រើញឹកញាប់
:header-rows: 1
:widths: 30 30 40

* - ការកំណត់
  - លទ្ធផល
  - ប្រើសម្រាប់
* - `kernel=3, stride=1, padding=1`
  - ទំហំដដែល
  - "same" convolution — ស្តង់ដារសម្រាប់ ResNet
* - `kernel=3, stride=2, padding=1`
  - ពាក់កណ្ដាល
  - downsample ដោយមិនប្រើ pooling
* - `kernel=1, stride=1, padding=0`
  - ទំហំដដែល
  - ប្ដូរចំនួន channels តែប៉ុណ្ណោះ
* - `kernel=in_size, padding=0`
  - `1 × 1`
  - global pooling (ឬប្រើ `AdaptiveAvgPool2d(1)`)
```

````{admonition} ប្រើ AdaptiveAvgPool2d ជំនួសការគណនាដោយដៃ
:class: tip
ជំនួសឱ្យការគណនាថា feature map នឹងមានទំហំប៉ុន្មានមុន flatten ប្រើ៖

```python
self.pool = nn.AdaptiveAvgPool2d(1)     # តែងតែផ្ដល់ (N, C, 1, 1)
self.flatten = nn.Flatten()              # -> (N, C)
```

វិធីនេះម៉ូដែលរបស់អ្នក **ទទួល input គ្រប់ទំហំ** ហើយអ្នកមិនចាំបាច់គណនាអ្វីទាំងអស់។
ឧទាហរណ៍ខាងលើបង្ហាញថាវាដំណើរការជាមួយ input 32×32, 64×64 និង 48×48។
````

### តាមដាន shape ដោយ forward hooks

បច្ចេកទេសមានប្រយោជន៍បំផុតសម្រាប់ debug CNN៖

```python
handles = []
for name, module in model.named_modules():
    if name and not list(module.children()):      # leaf modules តែប៉ុណ្ណោះ
        handles.append(module.register_forward_hook(
            lambda m, i, o, n=name: print(f"{n:<30} -> {tuple(o.shape)}")
        ))

with torch.no_grad():
    logits = model(images)

for h in handles:
    h.remove()                                    # ⚠️ កុំភ្លេចដកចេញ
```

```{admonition} ត្រូវដក hooks ចេញជានិច្ច
:class: warning
Hook រក្សា reference ទៅ closure (ដែលចាប់យក `name`, `shapes` dict ។ល។)។
បើអ្នកមិនហៅ `handle.remove()` ទេ reference នោះនឹងនៅជាប់ ហើយបង្កើត
**memory leak**។ នៅក្នុង loop ដែលបង្កើត hook រាល់ iteration វាអាចធ្វើឱ្យ
ម៉ូដែលចាស់ៗមិនត្រូវបានដោះលែងឡើយ។
```

## ៣. ជាន់សំខាន់ៗរបស់ torch.nn

```{literalinclude} examples/neural_networks_3.py
:language: python
:caption: neural_networks_3.py — ✅ បានសាកល្បងលើ CPU (ជាន់នីមួយៗជាមួយ input/output shape)
```

### តារាងយោងជាន់

```{list-table}
:header-rows: 1
:widths: 24 22 54

* - ប្រភេទ
  - ជាន់
  - ចំណាំ
* - **Linear**
  - `nn.Linear(in, out)`
  - weight shape = `(out, in)` ⚠️ បញ្ច្រាស
* -
  - `nn.LazyLinear(out)`
  - ស្វែងរក `in_features` ដោយស្វ័យប្រវត្តិនៅ forward ដំបូង
* -
  - `nn.Bilinear(in1, in2, out)`
  - `y = x1ᵀ A x2 + b`
* - **Convolution**
  - `nn.Conv1d/2d/3d`
  - input shape = `(N, C_in, *)`
* -
  - `nn.ConvTranspose2d`
  - upsample ("deconvolution") ប៉ុន្តែមាន checkerboard artifacts
* -
  - `nn.Unfold` / `nn.Fold`
  - im2col ដោយដៃ
* - **Pooling**
  - `nn.MaxPool2d(k)`
  - យកតម្លៃអតិបរមា
* -
  - `nn.AvgPool2d(k)`
  - យកមធ្យមភាគ
* -
  - `nn.AdaptiveAvgPool2d((1,1))`
  - global pooling ដោយមិនគិតពី input size
* - **Normalization**
  - `nn.BatchNorm1d/2d`
  - normalize តាម batch។ ⚠️ ឥរិយាបថខុសគ្នាក្នុង train/eval
* -
  - `nn.LayerNorm(dim)`
  - normalize តាម features។ ស្តង់ដារសម្រាប់ Transformer
* -
  - `nn.GroupNorm(g, c)`
  - ជម្រើសសម្រាប់ batch តូច (ឧ. detection/segmentation)
* -
  - `nn.InstanceNorm2d`
  - style transfer
* - **Activation**
  - `nn.ReLU`
  - លំនាំដើម សាមញ្ញ។ អាចមាន "dying ReLU"
* -
  - `nn.GELU`
  - ស្តង់ដារ Transformer/BERT
* -
  - `nn.SiLU` (Swish)
  - ស្តង់ដារ ConvNeXt, EfficientNet
* -
  - `nn.LeakyReLU(0.01)`
  - ដោះស្រាយ dying ReLU
* -
  - `nn.Softmax(dim)` / `nn.LogSoftmax(dim)`
  - ⚠️ **កុំ** ប្រើមុន `nn.CrossEntropyLoss`
* - **Regularization**
  - `nn.Dropout(p)`
  - ⚠️ ឥរិយាបថខុសគ្នាក្នុង train/eval
* -
  - `nn.Dropout2d(p)`
  - បិទ channels ទាំងមូល (សម្រាប់ CNN)
* - **Sequence**
  - `nn.Embedding(vocab, dim)`
  - ⚠️ ទាមទារ index ជា `int64`
* -
  - `nn.LSTM` / `nn.GRU`
  - ត្រឡប់ `(output, (h_n, c_n))`។ ប្រើ `batch_first=True`
* -
  - `nn.MultiheadAttention`
  - ត្រឡប់ `(attn_output, attn_weights)`
* -
  - `nn.TransformerEncoderLayer`
  - building block សម្រាប់ Transformer
* - **Reshaping**
  - `nn.Flatten(start, end)`
  - ជំនួស `x.view(x.size(0), -1)` ក្នុង Sequential
* -
  - `nn.Unflatten(dim, shape)`
  - បញ្ច្រាស Flatten
```

## ៤. train() និង eval()

នេះគឺជាចំណុចដែលអ្នកចាប់ផ្ដើមភ្លេចញឹកញាប់បំផុត ហើយវាបង្កើតបញ្ហាដែលពិបាករក។

```python
model.train()      # បើក training behaviour
model.eval()       # បើក evaluation behaviour
```

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - ជាន់
  - `train()`
  - `eval()`
* - `nn.Dropout`
  - បិទធាតុដោយចៃដន្យ ហើយពង្រីកអ្វីដែលនៅសល់ដោយ `1/(1-p)`
  - pass-through (identity)
* - `nn.BatchNorm*`
  - ប្រើស្ថិតិ batch បច្ចុប្បន្ន + ធ្វើបច្ចុប្បន្នភាព `running_mean`/`running_var`
  - ប្រើ `running_mean`/`running_var` ដែលបានរៀន
* - ជាន់ផ្សេងទៀត
  - ដូចគ្នា
  - ដូចគ្នា
```

````{admonition} វិធីត្រឹមត្រូវក្នុង training loop
:class: important
```python
for epoch in range(epochs):
    model.train()                          # <- ម្ដងក្នុងមួយ epoch
    for xb, yb in train_loader:
        ...

    model.eval()                           # <- ប្ដូរមុន validation
    with torch.no_grad():
        for xb, yb in val_loader:
            ...
```

`model.train()` មិនមែន "បើក gradient" ទេ — វាគ្រាន់តែកំណត់ flag `.training`។
Gradient ត្រូវបានគ្រប់គ្រងដោយ `torch.no_grad()` ដាច់ដោយឡែក។
**អ្នកត្រូវការទាំងពីរ** សម្រាប់ validation។
````

### Recursive propagation

`model.train()` កំណត់ flag នេះលើ **module រងទាំងអស់** ដោយ recursive។ អ្នកមិនចាំបាច់
ហៅវាលើជាន់នីមួយៗទេ។ ពិនិត្យដោយ៖

```python
print(model.training)                      # True / False
print([m.training for m in model.modules()])
```

## ៥. Functional API — torch.nn.functional

ជាន់ភាគច្រើនមានកំណែ "functional" ដែលគ្មាន parameter៖

```{list-table}
:header-rows: 1
:widths: 34 34 32

* - Module (មាន state)
  - Functional (គ្មាន state)
  - ប្រើអ្វី?
* - `nn.ReLU()`
  - `F.relu(x)`
  - ណាក៏បាន
* - `nn.Dropout(0.5)`
  - `F.dropout(x, 0.5, training=self.training)`
  - Module — ងាយភ្លេច `training=`
* - `nn.BatchNorm2d(64)`
  - `F.batch_norm(x, mean, var, w, b, ...)`
  - Module — មាន running statistics
* - `nn.Linear(10, 5)`
  - `F.linear(x, w, b)`
  - Module
* - (គ្មាន)
  - `F.cross_entropy(logits, y)`
  - Functional តែប៉ុណ្ណោះ — ស្តង់ដារសម្រាប់ classification
* - `nn.Softmax(dim)`
  - `F.softmax(x, dim=1)`
  - Functional សម្រាប់ inference
* - `nn.NLLLoss`
  - `F.nll_loss(...)`
  - Module — ប៉ុន្តែចូលចិត្ត `cross_entropy` ដែលរួម log-softmax
```

````{admonition} F.dropout ទាមទារ training= ដោយដៃ
:class: warning
```python
F.dropout(x, p=0.5)                          # ⚠️ លំនាំដើម training=False!
F.dropout(x, p=0.5, training=self.training)  # ✓ ត្រឹមត្រូវ
```

បើអ្នកភ្លេច `training=self.training` dropout នឹង **មិនដំណើរការ** ក្នុងការបណ្ដុះបណ្ដាល។
ប្រើ `nn.Dropout` ជំនួស លុះត្រាតែអ្នកមានហេតុផលច្បាស់លាស់។
````

## ៦. ការកំណត់តម្លៃដើម weight (Initialization)

PyTorch ធ្វើ initialization ដោយស្វ័យប្រវត្តិសម្រាប់ជាន់ស្តង់ដារ
(`nn.Linear` ប្រើ Kaiming uniform)។ សម្រាប់ជាន់ផ្ទាល់ខ្លួន៖

```python
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

model.apply(init_weights)          # អនុវត្តលើ module រងទាំងអស់
```

```{list-table} ជ្រើសរើស initialization
:header-rows: 1
:widths: 30 70

* - Activation
  - Initialization
* - `ReLU`, `LeakyReLU`, `SiLU`
  - `kaiming_normal_` (He) ជាមួយ `nonlinearity='relu'`
* - `Tanh`, `Sigmoid`
  - `xavier_uniform_` (Glorot)
* - Output layer (regression)
  - តូច ឬសូន្យ ដើម្បីឱ្យ loss ដំបូងតូច
```

## ៧. ការណែនាំអំពីការរចនា

```{list-table}
:header-rows: 1
:widths: 40 60

* - អ្វីដែលត្រូវធ្វើ
  - ហេតុអ្វី
* - ដាក់ activation ជាជាន់ដាច់ដោយឡែកក្នុង `Sequential`
  - ងាយស្រួល debug ដោយ hooks និងងាយស្រួលផ្លាស់ប្ដូរ
* - ប្រើ `AdaptiveAvgPool2d` ជំនួសការគណនា flatten size
  - ម៉ូដែលដំណើរការជាមួយ input គ្រប់ទំហំ
* - កំណត់ type hints លើ `forward()`
  - ជួយ IDE និងអ្នកអាន
* - ផ្ដល់ logits ចេញពី `forward()` មិនមែន probabilities
  - `nn.CrossEntropyLoss` រួមបញ្ចូល log-softmax ហើយ (ស្ថិរភាពលេខ)
* - ប្រើ `nn.ModuleList`/`ModuleDict` សម្រាប់ជាន់ដែលបង្កើតតាម loop
  - List ធម្មតា **មិន** ចុះបញ្ជី parameters!
```

````{admonition} ModuleList vs list ធម្មតា — កំហុសដែលលាក់ខ្លួន
:class: danger
```python
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4) for _ in range(3)]   # ✗ list ធម្មតា

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

print(sum(p.numel() for p in Bad().parameters()))   # 0  ⚠️
```

Parameter ទាំងអស់ **មិនត្រូវបានចុះបញ្ជី** ដូច្នេះ optimizer មិនដែលធ្វើបច្ចុប្បន្នភាពពួកវាទេ។
ប្រើ `nn.ModuleList`៖

```python
self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
print(sum(p.numel() for p in Good().parameters()))  # 75  ✓
```

បញ្ហានេះមិនបង្ហាញ error ណាមួយឡើយ — ម៉ូដែលគ្រាន់តែមិនរៀន។
````

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `cannot assign module before Module.__init__() call`
  - ភ្លេច ឬដាក់ខុសលំដាប់ `super().__init__()`
  - សរសេរវាជាបន្ទាត់ដំបូង
* - `mat1 and mat2 shapes cannot be multiplied (8x12 and 10x5)`
  - `nn.Linear` ទទួល feature ចំនួនខុស
  - ពិនិត្យ `x.shape[-1]` ធៀប `in_features`
* - `expected 2D or 3D input to Linear`
  - បញ្ចូន 4D tensor
  - `nn.Flatten()` ឬ `.reshape(x.size(0), -1)` ជាមុន
* - `Expected 3D (unbatched) or 4D (batched) input to Conv2d`
  - input ខ្វះ channel dim
  - `x.unsqueeze(0)` (batch) ឬ `unsqueeze(1)` (channel)
* - `non-empty 3D or 4D (batch mode) tensor expected`
  - `BatchNorm1d` ជាមួយ input 1D
  - `x.unsqueeze(0)`
* - `CUDNN_STATUS_NOT_SUPPORTED` ឬ NaN ជាមួយ BatchNorm
  - batch size = 1 ក្នុង training mode
  - ប្រើ `GroupNorm` ឬ `LayerNorm` សម្រាប់ batch តូច
* - optimizer ទទួល parameter list ទទេ
  - ប្រើ list ធម្មតាជំនួស `ModuleList`
  - ប្រើ `nn.ModuleList` / `nn.ModuleDict`
* - លទ្ធផលវាយតម្លៃប្រែប្រួលរាល់ការរត់
  - ភ្លេច `model.eval()`
  - ហៅ `model.eval()` + `torch.no_grad()`
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **`super().__init__()` ជាបន្ទាត់ដំបូង** នៃ `__init__` ជានិច្ច។
2. **ហៅ `model(x)` មិនមែន `model.forward(x)`**។
3. **ប្រើ `nn.ModuleList`** បើអ្នកបង្កើតជាន់តាម loop — list ធម្មតាមិនចុះបញ្ជី parameters។
4. **`forward()` ត្រឡប់ logits** មិនមែន probabilities (សម្រាប់ classification)។
5. **`model.train()` / `model.eval()`** ផ្លាស់ប្ដូរ Dropout និង BatchNorm ប៉ុណ្ណោះ។
6. **`AdaptiveAvgPool2d(1)`** ធ្វើឱ្យម៉ូដែលមិនអាស្រ័យលើ input size។
7. **Forward hooks** ជាឧបករណ៍ debug shape ដ៏ល្អបំផុត — ប៉ុន្តែត្រូវដកចេញ។
```

## មេរៀនបន្ទាប់

{doc}`loss-and-optimizers` — ជ្រើសរើស loss function និង optimizer ត្រឹមត្រូវ។
