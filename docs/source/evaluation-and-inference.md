# Model Evaluation និង Inference

បន្ទាប់ពីបណ្ដុះបណ្ដាល អ្នកត្រូវ **វាយតម្លៃ** និង **ប្រើប្រាស់** ម៉ូដែល។ ទាំងពីរ
ទាមទារការយកចិត្តទុកដាក់ខ្លាំងជាងការបណ្ដុះបណ្ដាល ព្រោះកំហុសនៅទីនេះមិនបង្ហាញ
ជា loss ខ្ពស់ទេ — វាបង្ហាញជា **លេខខុសដែលមើលទៅហាក់ដូចជាត្រឹមត្រូវ**។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ២ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
```

## ១. របៀបវាយតម្លៃត្រឹមត្រូវ

វិធានពីរដែលមិនអាចលើកលែងបាន៖

```python
model.eval()                 # ១. បិទ Dropout, បង្កក BatchNorm statistics
with torch.no_grad():        # ២. បិទ gradient tracking
    logits = model(x)
```

```{list-table}
:header-rows: 1
:widths: 26 74

* - បើអ្នកភ្លេច
  - អ្វីដែលកើតឡើង
* - `model.eval()`
  - Dropout នៅតែច្របល់ → លទ្ធផល **មិនដូចគ្នារាល់ការរត់** លើទិន្នន័យដដែល។ BatchNorm ប្រើស្ថិតិ batch បច្ចុប្បន្ន → លទ្ធផលអាស្រ័យលើ batch composition។
* - `torch.no_grad()`
  - Autograd graph ត្រូវបានរក្សាទុកសម្រាប់រាល់ batch → **memory កើនឡើងជានិច្ច** ហើយការវាយតម្លៃយឺតជាង។ លទ្ធផលលេខដូចគ្នា។
```

## ២. Metrics

PyTorch មិនមាន metrics បណ្ណាល័យទេ (នោះគឺ `torchmetrics` ដែលជាគម្រោងដាច់ដោយឡែក)។
ការគណនាដោយផ្ទាល់គឺសាមញ្ញ ហើយអ្នកគ្រប់គ្រងអ្វីៗទាំងអស់៖

```{literalinclude} examples/evaluation_and_inference_1.py
:language: python
:caption: evaluation_and_inference_1.py — ✅ បានសាកល្បងលើ CPU
```

### Accuracy

```python
preds = logits.argmax(dim=1)                    # (N,)
accuracy = (preds == labels).float().mean().item()
```

### Confusion Matrix

```python
def confusion_matrix(y_true, y_pred, num_classes):
    """cm[i, j] = ចំនួនសំណាកដែលជាការពិត i ហើយត្រូវបានទាយថា j។"""
    idx = y_true * num_classes + y_pred
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
```

បច្ចេកទេស `y_true * C + y_pred` + `bincount` នេះលឿនជាង loop ច្រើនដង
ហើយដំណើរការលើ GPU ផងដែរ។

### Precision / Recall / F1

```python
tp = cm.diagonal().float()          # ការទាយត្រូវក្នុងថ្នាក់នីមួយៗ
pred_pos = cm.sum(dim=0).float()    # សរុបតាមឈរ — ទាយថាជាថ្នាក់ j
true_pos = cm.sum(dim=1).float()    # សរុបតាមជួរ — ការពិតជាថ្នាក់ i

precision = tp / pred_pos.clamp(min=1)
recall = tp / true_pos.clamp(min=1)
f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
```

```{admonition} តើ clamp(min=1) មានប្រយោជន៍អ្វី?
:class: tip
បើម៉ូដែលមិនដែលទាយថ្នាក់មួយ សូន្យ នោះ `pred_pos = 0` ហើយការចែកផ្ដល់ `nan`។
`clamp(min=1)` ធ្វើឱ្យលទ្ធផល = 0 ជំនួសវិញ ដែលជាការបកស្រាយត្រឹមត្រូវ
("precision គ្មានន័យព្រោះគ្មានការទាយ")។
```

### Macro vs Weighted Average

```{list-table}
:header-rows: 1
:widths: 22 30 48

* - មធ្យមភាគ
  - រូបមន្ត
  - ប្រើពេល
* - **Macro**
  - មធ្យមភាគសាមញ្ញតាមថ្នាក់
  - ថ្នាក់តូចៗសំខាន់ដូចថ្នាក់ធំ។ រំញោចចំពោះ class imbalance
* - **Weighted**
  - មធ្យមភាគថ្លឹងតាមចំនួនសំណាក
  - អ្នកចង់បានរូបភាពជាក់ស្ដែងលើទិន្នន័យ។ weighted recall = accuracy
* - **Micro**
  - គណនា TP/FP/FN សរុប បន្ទាប់មកគណនា
  - សម្រាប់ multi-label
```

ឧទាហរណ៍ខាងលើផ្ទៀងផ្ទាត់ថា **weighted recall ស្មើ accuracy** ជានិច្ច —
ការត្រួតពិនិត្យដ៏ល្អសម្រាប់កូដ metric របស់អ្នក។

### Streaming metrics សម្រាប់ទិន្នន័យធំ

កុំរក្សាទុក prediction ទាំងអស់ក្នុង memory។ បូក confusion matrix ជាបន្តបន្ទាប់៖

```python
total_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
for xb, yb in val_loader:
    with torch.no_grad():
        pred = model(xb.to(device)).argmax(1)
    total_cm += confusion_matrix(yb.cpu(), pred.cpu(), num_classes)

accuracy = total_cm.diagonal().sum().item() / total_cm.sum().item()
```

## ៣. Inference

```{literalinclude} examples/evaluation_and_inference_2.py
:language: python
:caption: evaluation_and_inference_2.py — ✅ បានសាកល្បងលើ CPU (រួមទាំងលេខវាស់លើម៉ាស៊ីននេះ)
```

### no_grad() vs inference_mode()

```{list-table}
:header-rows: 1
:widths: 26 20 20 34

* - លក្ខណៈ
  - `no_grad()`
  - `inference_mode()`
  - ចំណាំ
* - បិទ gradient tracking
  - ✅
  - ✅
  - ទាំងពីរ
* - លទ្ធផលអាចប្រើក្នុង autograd បន្ត
  - ✅ (ជាមួយ `.detach().requires_grad_()`)
  - ❌ RuntimeError
  - នេះជាភាពខុសគ្នាសំខាន់
* - លឿនជាង
  - មូលដ្ឋាន
  - ✅ បន្តិច
  - មិនកត់ត្រា version counter
* - ប្រើសម្រាប់
  - validation loop
  - production inference
  - —
```

````{admonition} Inference tensor មិនអាចត្រឡប់ទៅ autograd បានទេ
:class: warning
```python
with torch.inference_mode():
    out = model(x)

out.requires_grad_(True)     # RuntimeError!
out.sum().backward()         # RuntimeError!
```

បើអ្នកត្រូវការ gradient (ឧទាហរណ៍ adversarial examples, gradient-based attribution)
ប្រើ `torch.no_grad()` ឬកុំប្រើ context manager ណាមួយទាំងអស់។
````

### Batch inference

កុំរត់ម៉ូដែលម្ដងមួយសំណាក។ ប្រើ batches៖

```python
all_preds, all_probs = [], []
with torch.inference_mode():
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs)
        all_preds.append(probs.argmax(dim=1))

probs = torch.cat(all_probs, dim=0)      # (N, C)
preds = torch.cat(all_preds, dim=0)      # (N,)
```

បើទិន្នន័យធំជាង RAM អ្នកអាចសរសេរលទ្ធផលទៅ disk ជំនួសការប្រមូលវាក្នុង list។

### Top-k predictions

```python
topk_probs, topk_idx = torch.topk(probs, k=3, dim=1)   # ទាំងពីរ shape (N, 3)
```

## ៤. torch.compile

PyTorch 2.x មាន `torch.compile()` ដែល fuse operations និងកាត់បន្ថយ Python overhead៖

```python
model = torch.compile(model)                    # ប្រើនៅកន្លែងដដែល
out = model(x)                                   # forward ដំបូងយឺត (compile)
```

```{list-table}
:header-rows: 1
:widths: 24 76

* - ប៉ារ៉ាម៉ែត្រ
  - អត្ថន័យ
* - `mode="default"`
  - លំនឹងរវាងពេល compile និងល្បឿន
* - `mode="reduce-overhead"`
  - ប្រើ CUDA graphs។ ល្អសម្រាប់ batch size ថេរ
* - `mode="max-autotune"`
  - សាកល្បង kernel ច្រើន។ ពេល compile យូរខ្លាំង ប៉ុន្តែលឿនបំផុត
* - `dynamic=True`
  - អនុញ្ញាតឱ្យ shape ប្រែប្រួលដោយមិន recompile
* - `fullgraph=True`
  - បង្ខំឱ្យ graph តែមួយ។ បរាជ័យបើមាន graph break
```

```{admonition} កត្តាដែលធ្វើឱ្យ torch.compile មិនមានប្រយោជន៍
:class: note
- **ពេល compile ដំបូងអាចយូរ** (រាប់វិនាទី ឬរាប់នាទី)។ មិនសមស្របសម្រាប់ script ខ្លី។
- **Shape ដែលប្រែប្រួល** បណ្ដាលឱ្យ recompile។ ប្រើ `dynamic=True`។
- **ត្រូវការ backend ដែលគាំទ្រ** (TorchInductor)។ អាចមិនមានលើប្រព័ន្ធខ្លះ។
- **មិនមានប្រយោជន៍ច្រើនលើ CPU** សម្រាប់ម៉ូដែលតូច។

ឯកសារនេះ **មិនអះអាងលេខល្បឿនសកលណាមួយទេ** ព្រោះវាអាស្រ័យខ្លាំងលើ hardware,
កំណែ PyTorch និង architecture។ សូមវាស់លើម៉ាស៊ីនរបស់អ្នកផ្ទាល់។
```

## ៥. ត្រួតពិនិត្យលទ្ធផល inference

មុនដាក់ម៉ូដែលឱ្យប្រើប្រាស់ សូមត្រួតពិនិត្យចំណុចទាំងនេះ៖

```{list-table}
:header-rows: 1
:widths: 32 68

* - ត្រួតពិនិត្យ
  - ហេតុអ្វី
* - `model.training is False`
  - បញ្ជាក់ថា `eval()` បានហៅ
* - `probs.sum(dim=1) ≈ 1.0`
  - softmax ត្រឹមត្រូវ
* - `(probs >= 0).all()`
  - គ្មានតម្លៃអវិជ្ជមាន
* - Prediction distribution មិន collapse
  - `torch.bincount(preds, minlength=C)` — បើថ្នាក់តែមួយទទួល ១០០% មានបញ្ហា
* - លទ្ធផល deterministic លើ input ដដែល
  - រត់ពីរដង ប្រៀបធៀប។ បើខុស → Dropout ឬ randomness មិនបានបិទ
* - ទិន្នន័យ input បាន normalize ដូច training
  - ភាពខុសគ្នានៅទីនេះជាមូលហេតុទូទៅនៃ "ម៉ូដែលល្អក្នុង training ប៉ុន្តែអាក្រក់ក្នុង production"
```

```python
# ការត្រួតពិនិត្យរហ័សមុន deployment
model.eval()
with torch.inference_mode():
    p1 = torch.softmax(model(sample), dim=1)
    p2 = torch.softmax(model(sample), dim=1)
assert torch.equal(p1, p2), "ម៉ូដែលមិន deterministic — ត្រួតពិនិត្យ eval() និង Dropout"
assert torch.allclose(p1.sum(dim=1), torch.ones(p1.shape[0])), "probabilities មិនបូកស្មើ ១"
```

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស / រោគសញ្ញា
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `Inference tensors cannot be saved for backward`
  - ប្រើ inference tensor ក្នុង autograd
  - ប្រើ `torch.no_grad()` ជំនួស `inference_mode()`
* - `a view of an inference tensor cannot be used in autograd`
  - ដូចគ្នា
  - ដូចគ្នា
* - metric ប្រែប្រួលរាល់ការរត់
  - ភ្លេច `model.eval()`
  - ហៅ `model.eval()`
* - memory អស់ពេលវាយតម្លៃ dataset ធំ
  - ភ្លេច `no_grad()` ឬរក្សាទុក tensor ទាំងអស់
  - ប្រើ streaming metrics + `.item()`
* - accuracy production ទាបជាង validation
  - preprocessing ខុសគ្នា
  - ប្រើ transform ដូចគ្នាពិតប្រាកដ
* - `expected scalar type Float but found Half`
  - model ជា fp16 ប៉ុន្តែ input ជា fp32
  - `x = x.to(dtype=next(model.parameters()).dtype)`
* - លទ្ធផលខុសបន្ទាប់ពីផ្ទុក checkpoint
  - ភ្លេច `model.eval()` បន្ទាប់ពី `load_state_dict`
  - `load_state_dict` មិនកែ `.training` flag ទេ
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **`model.eval()` + `torch.no_grad()`** សម្រាប់ validation។
2. **`model.eval()` + `torch.inference_mode()`** សម្រាប់ production inference។
3. **Streaming metrics** (បូក confusion matrix) ជំនួសការរក្សាទុក prediction ទាំងអស់។
4. **Weighted recall = accuracy** — ការត្រួតពិនិត្យដ៏ល្អសម្រាប់កូដ metric។
5. **Batch inference** ជានិច្ច — កុំរត់ម្ដងមួយសំណាក។
6. **ត្រួតពិនិត្យ determinism** មុន deployment។
7. **Preprocessing ត្រូវតែដូច training ពិតប្រាកដ**។
```

## មេរៀនបន្ទាប់

{doc}`saving-and-loading` — រក្សាទុកម៉ូដែលសម្រាប់ប្រើប្រាស់នាពេលអនាគត។
