# តារាងយោង API រហ័ស

តារាងសង្ខេបនៃ API ដែលប្រើញឹកញាប់បំផុត រួមជាមួយតំណទៅកាន់មេរៀនដែលពន្យល់លម្អិត។
ឈ្មោះ API ទាំងអស់រក្សាទុកជាភាសាអង់គ្លេសដូចដើម។

```{admonition} សម្រាប់ API ពេញលេញ
:class: note
ទំព័រនេះគ្របដណ្ដប់តែអ្វីដែលប្រើក្នុងមេរៀនទាំង ១៩ របស់ឯកសារខ្មែរនេះប៉ុណ្ណោះ។
សម្រាប់បញ្ជីពេញលេញ សូមមើល
[PyTorch API Reference ផ្លូវការ](https://docs.pytorch.org/docs/stable/index.html)។
```

## ការបង្កើត Tensor

```{list-table}
:header-rows: 1
:widths: 36 64

* - API
  - ការពន្យល់
* - `torch.tensor(data)`
  - បង្កើតពី Python data (**ចម្លង** ទិន្នន័យ)
* - `torch.from_numpy(arr)`
  - បង្កើតពី NumPy array (**ចែក** memory)
* - `torch.zeros(*shape)` / `torch.ones(*shape)`
  - បំពេញដោយ 0 / 1
* - `torch.empty(*shape)`
  - មិនបានកំណត់តម្លៃ — កុំពឹងលើមាតិកា
* - `torch.full(shape, value)`
  - បំពេញដោយតម្លៃថេរ
* - `torch.eye(n)`
  - identity matrix
* - `torch.arange(start, end, step)`
  - ដូច Python `range`
* - `torch.linspace(start, end, steps)`
  - ចែកចន្លោះស្មើគ្នា
* - `torch.rand(*shape)`
  - uniform `[0, 1)`
* - `torch.randn(*shape)`
  - normal `N(0, 1)`
* - `torch.randint(low, high, shape)`
  - ចំនួនគត់ `[low, high)`
* - `torch.zeros_like(t)` / `ones_like` / `randn_like` / `empty_like`
  - រក្សា shape/dtype/device ដូច `t`
```

{doc}`តារាងពេញលេញ <tensors>`

## លក្ខណៈសម្បត្តិ Tensor

```{list-table}
:header-rows: 1
:widths: 30 70

* - លក្ខណៈ / អនុគមន៍
  - ការពន្យល់
* - `t.shape` / `t.size(dim)`
  - ទំហំវិមាត្រ
* - `t.ndim`
  - ចំនួនវិមាត្រ
* - `t.numel()`
  - ចំនួនធាតុសរុប
* - `t.dtype`
  - ប្រភេទធាតុ (`torch.float32`, `torch.int64` ។ល។)
* - `t.device`
  - `cpu`, `cuda:0`, `mps`
* - `t.stride()`
  - ចម្ងាយរវាងធាតុក្នុងវិមាត្រនីមួយៗ
* - `t.is_contiguous()`
  - តើធាតុរៀងជាបន្តបន្ទាប់ក្នុង memory ឬទេ
* - `t.element_size()`
  - bytes ក្នុងមួយធាតុ
* - `t.requires_grad`
  - តើ autograd កត់ត្រាប្រមាណវិធីលើវាឬទេ
* - `t.grad`
  - gradient ដែលបានគណនា (តែលើ leaf tensor)
* - `t.grad_fn`
  - អនុគមន៍ backward ដែលបានកត់ត្រា (`None` សម្រាប់ leaf)
* - `t.item()`
  - យកតម្លៃ Python ពី scalar tensor
* - `t.tolist()`
  - ប្ដូរទៅ nested Python list
* - `t.numpy()`
  - ប្ដូរទៅ NumPy (ត្រូវការ CPU + គ្មាន grad)
```

## ប្រមាណវិធី

```{list-table}
:header-rows: 1
:widths: 34 66

* - API
  - ការពន្យល់
* - `a + b`, `a - b`, `a * b`, `a / b`, `a ** n`
  - element-wise
* - `a @ b`, `torch.matmul(a, b)`, `a.mm(b)`
  - matrix multiply
* - `torch.bmm(A, B)`
  - batched matrix multiply
* - `torch.dot(a, b)`
  - dot product (1D តែប៉ុណ្ណោះ)
* - `torch.add`, `sub`, `mul`, `div`, `pow`, `neg`, `abs`
  - កំណែអនុគមន៍
* - `torch.exp`, `log`, `log2`, `sqrt`, `sin`, `cos`, `tanh`
  - គណិតវិទ្យា
* - `torch.clamp(t, min, max)`
  - កំណត់ដែនតម្លៃ
* - `torch.round`, `floor`, `ceil`, `sign`
  - បង្គត់
* - `torch.sum(t, dim, keepdim)`
  - បូក
* - `torch.mean`, `std`, `var`, `median`, `prod`
  - ស្ថិតិ
* - `torch.max(t, dim)` / `torch.min(t, dim)`
  - ត្រឡប់ `(values, indices)`
* - `torch.argmax(t, dim)` / `argmin`
  - index នៃតម្លៃខ្ពស់/ទាបបំផុត
* - `torch.topk(t, k, dim)`
  - ត្រឡប់ `(values, indices)` ធំបំផុត k
* - `torch.sort(t, dim)`
  - ត្រឡប់ `(values, indices)`
* - `torch.allclose(a, b)` / `torch.isclose`
  - ប្រៀបធៀបដោយអត់ឱនកំហុស rounding
* - `torch.equal(a, b)`
  - ប្រៀបធៀបពិតប្រាកដ
* - `torch.isnan(t)` / `isinf` / `isfinite`
  - ត្រួតពិនិត្ថតម្លៃខូច
* - `t.add_()`, `mul_()`, `zero_()`, `fill_()`, `clamp_()`
  - in-place (បញ្ចប់ដោយ `_`) — ⚠️ អាចខូច autograd graph
```

{doc}`មើលការពន្យល់លម្អិត <tensors>`

## ការផ្លាស់ប្ដូររូបរាង

```{list-table}
:header-rows: 1
:widths: 34 66

* - API
  - ការពន្យល់
* - `t.view(*shape)`
  - ប្ដូរ shape (ទាមទារ contiguous)
* - `t.reshape(*shape)`
  - ដូច `view` ប៉ុន្តែចម្លងបើចាំបាច់
* - `t.flatten(start, end)`
  - រួមវិមាត្រ
* - `t.unflatten(dim, sizes)`
  - ពន្លាតវិមាត្រ
* - `t.transpose(d0, d1)`
  - ប្ដូរវិមាត្រពីរ
* - `t.permute(*dims)`
  - រៀបវិមាត្រទាំងអស់ឡើងវិញ
* - `t.T` / `t.mT`
  - transpose (2D / batched)
* - `t.squeeze(dim)` / `unsqueeze(dim)`
  - ដក / បន្ថែមវិមាត្រទំហំ 1
* - `t.expand(*shape)`
  - ពង្រីកវិមាត្រទំហំ 1 (**គ្មានការចម្លង memory**)
* - `t.repeat(*sizes)`
  - ចម្លងទិន្នន័យ (**ប្រើ memory**)
* - `t.contiguous()`
  - ចម្លងឱ្យរៀងជាបន្តបន្ទាប់
* - `torch.cat(tensors, dim)`
  - ភ្ជាប់តាមវិមាត្រដែលមាន
* - `torch.stack(tensors, dim)`
  - បង្កើតវិមាត្រថ្មី
* - `torch.split(t, size)` / `chunk(t, n)`
  - បំបែក
* - `torch.flip(t, dims)`
  - ត្រឡប់លំដាប់
* - `torch.roll(t, shifts, dims)`
  - រំកិលធាតុ
* - `torch.broadcast_shapes(*shapes)`
  - គណនាលទ្ធផល shape ដោយមិនបង្កើត tensor
* - `torch.broadcast_tensors(*tensors)`
  - ពង្រីក tensors ទៅ shape រួម
```

{doc}`មើលច្បាប់ broadcasting <shapes-and-broadcasting>`

## Indexing

```{list-table}
:header-rows: 1
:widths: 32 68

* - ទម្រង់
  - ការពន្យល់
* - `t[i]`, `t[i, j]`
  - index ធម្មតា
* - `t[a:b]`, `t[::2]`, `t[-1]`
  - slicing
* - `t[..., k]`
  - `...` = វិមាត្រដែលនៅសល់ទាំងអស់
* - `t[idx_tensor]`
  - advanced indexing (ចម្លងទិន្នន័យ)
* - `t[bool_mask]`
  - boolean masking (ផ្ដល់ 1D tensor)
* - `torch.where(cond, a, b)`
  - ជ្រើសរើសម្ដងមួយធាតុ — រក្សា shape
* - `torch.nonzero(mask)`
  - indices នៃធាតុ `True`
* - `torch.masked_select(t, mask)`
  - ជ្រើសយកធាតុតាម mask
* - `t.masked_fill(mask, value)`
  - ជំនួសធាតុតាម mask (មានកំណែ `_`)
* - `torch.gather(t, dim, index)`
  - ប្រមូលធាតុតាម index tensor
* - `torch.scatter(t, dim, index, src)`
  - ដាក់ធាតុតាម index tensor
* - `torch.index_select(t, dim, index)`
  - ជ្រើសយកតាម index ក្នុងវិមាត្រមួយ
* - `torch.take_along_dim(t, idx, dim)`
  - ជ្រើសយកតាម index ពហុវិមាត្រ
```

## Device និង dtype

```{list-table}
:header-rows: 1
:widths: 38 62

* - API
  - ការពន្យល់
* - `torch.device("cpu" \| "cuda" \| "mps")`
  - object តំណាង device
* - `t.to(device)` / `t.to(dtype)` / `t.to(other_tensor)`
  - ផ្លាស់ទី ឬប្ដូរប្រភេទ (**ត្រឡប់ tensor ថ្មី**)
* - `t.cpu()` / `t.cuda()`
  - ផ្លាស់ទីជាក់លាក់
* - `t.float()` / `double()` / `half()` / `bfloat16()` / `long()` / `bool()`
  - ប្ដូរ dtype
* - `torch.cuda.is_available()`
  - តើមាន NVIDIA GPU ឬទេ
* - `torch.backends.mps.is_available()`
  - តើមាន Apple Silicon GPU ឬទេ
* - `torch.cuda.device_count()`
  - ចំនួន GPU
* - `torch.cuda.get_device_name(i)`
  - ឈ្មោះ GPU
* - `torch.cuda.get_device_capability(i)`
  - compute capability (≥ 8.0 គាំទ្រ bfloat16)
* - `torch.cuda.memory_allocated()` / `max_memory_allocated()` / `memory_reserved()`
  - ការប្រើប្រាស់ GPU memory
* - `torch.cuda.synchronize()`
  - រង់ចាំឱ្យប្រតិបត្តិការ GPU ទាំងអស់បញ្ចប់ (ចាំបាច់មុនវាស់ពេល)
* - `torch.get_device_module(device)`
  - module សម្រាប់ device (`torch.cuda` ឬ `torch.cpu`)
```

{doc}`មើលការណែនាំ device <devices>`

## Autograd

```{list-table}
:header-rows: 1
:widths: 38 62

* - API
  - ការពន្យល់
* - `t.requires_grad_(True)`
  - បើកការកត់ត្រា gradient
* - `t.backward()`
  - គណនា gradient (ទាមទារ scalar ឬ `gradient=` argument)
* - `t.backward(retain_graph=True)`
  - រក្សា graph សម្រាប់ backward ម្ដងទៀត
* - `torch.autograd.grad(outputs, inputs, ...)`
  - យក gradient ដោយមិនរក្សាទុកក្នុង `.grad`
* - `torch.autograd.gradcheck(fn, inputs)`
  - ផ្ទៀងផ្ទាត់ gradient ដោយ finite differences (⚠️ ត្រូវការ float64)
* - `torch.autograd.set_detect_anomaly(True)`
  - រកប្រភព NaN ក្នុង backward (យឺតខ្លាំង — ប្រើតែពេល debug)
* - `t.detach()`
  - កាត់ចេញពី graph (**នៅចែក memory**)
* - `t.detach().clone()`
  - ច្បាប់ចម្លងដាច់ដោយឡែកពិតប្រាកដ
* - `with torch.no_grad():`
  - បិទការកត់ត្រាក្នុង block
* - `with torch.inference_mode():`
  - លឿនជាង `no_grad` ប៉ុន្តែលទ្ធផលមិនអាចប្រើក្នុង autograd
* - `torch.is_grad_enabled()`
  - ត្រួតពិនិត្ថស្ថានភាព
* - `torch.set_grad_enabled(False)`
  - បើក/បិទជាអនុគមន៍
```

{doc}`មើលការពន្យល់ពេញលេញ <autograd>`

## torch.nn

### Modules

```{list-table}
:header-rows: 1
:widths: 32 68

* - Module
  - Input shape → Output shape
* - `nn.Linear(in, out)`
  - `(*, in)` → `(*, out)`
* - `nn.LazyLinear(out)`
  - ដូច Linear ប៉ុន្តែស្វែងរក `in` ដោយស្វ័យប្រវត្តិ
* - `nn.Conv1d(in_c, out_c, k)`
  - `(N, in_c, L)` → `(N, out_c, L')`
* - `nn.Conv2d(in_c, out_c, k)`
  - `(N, in_c, H, W)` → `(N, out_c, H', W')`
* - `nn.ConvTranspose2d`
  - upsample
* - `nn.MaxPool2d(k)` / `AvgPool2d(k)`
  - `(N, C, H, W)` → `(N, C, H/k, W/k)`
* - `nn.AdaptiveAvgPool2d((1,1))`
  - `(N, C, H, W)` → `(N, C, 1, 1)` — មិនអាស្រ័យលើ input size
* - `nn.AdaptiveMaxPool2d`
  - ដូចគ្នា ប៉ុន្តែយក max
* - `nn.BatchNorm1d(C)`
  - `(N, C)` → ដូចគ្នា
* - `nn.BatchNorm2d(C)`
  - `(N, C, H, W)` → ដូចគ្នា
* - `nn.LayerNorm(D)`
  - `(*, D)` → ដូចគ្នា
* - `nn.GroupNorm(G, C)`
  - `(N, C, *)` → ដូចគ្នា
* - `nn.ReLU` / `GELU` / `SiLU` / `LeakyReLU` / `Tanh` / `Sigmoid`
  - ដូចគ្នា → ដូចគ្នា
* - `nn.Softmax(dim)` / `nn.LogSoftmax(dim)`
  - ដូចគ្នា → ដូចគ្នា
* - `nn.Dropout(p)` / `Dropout2d(p)` / `Dropout3d(p)`
  - ដូចគ្នា → ដូចគ្នា
* - `nn.Embedding(vocab, dim)`
  - `(N, L)` int64 → `(N, L, dim)`
* - `nn.LSTM(input, hidden, batch_first=True)`
  - `(N, L, in)` → `(N, L, hidden)` + `(h_n, c_n)`
* - `nn.GRU`
  - ដូច LSTM ប៉ុន្តែគ្មាន `c_n`
* - `nn.MultiheadAttention(embed_dim, heads, batch_first=True)`
  - ត្រឡប់ `(attn_output, attn_weights)`
* - `nn.TransformerEncoderLayer` / `TransformerEncoder`
  - Transformer building blocks
* - `nn.Flatten()` / `nn.Unflatten(dim, shape)`
  - reshape
* - `nn.Sequential(*modules)`
  - ភ្ជាប់ជាន់តាមលំដាប់
* - `nn.ModuleList` / `nn.ModuleDict`
  - ⚠️ ចាំបាច់សម្រាប់ជាន់ដែលបង្កើតតាម loop
* - `nn.Parameter(tensor)`
  - ចុះឈ្មោះ tensor ជា parameter
```

### វិធីសាស្ត្ររបស់ nn.Module

```{list-table}
:header-rows: 1
:widths: 34 66

* - វិធីសាស្ត្រ
  - ការពន្យល់
* - `model(x)`
  - forward pass (⚠️ ប្រើនេះ មិនមែន `model.forward(x)`)
* - `model.train()` / `model.eval()`
  - ប្ដូរឥរិយាបថ Dropout/BatchNorm
* - `model.parameters()`
  - iterator នៃ tensors parameter ទាំងអស់
* - `model.named_parameters()`
  - ដូចគ្នា ជាមួយឈ្មោះ
* - `model.children()` / `named_children()`
  - module រងផ្ទាល់
* - `model.modules()` / `named_modules()`
  - module រងទាំងអស់ (recursive)
* - `model.state_dict()` / `load_state_dict(sd, strict=True)`
  - រក្សាទុក / ផ្ទុក weight
* - `model.to(device)` / `model.cpu()` / `model.cuda()`
  - ផ្លាស់ទី (**កែនៅនឹងកន្លែង** ខុសពី `Tensor.to()`)
* - `model.apply(fn)`
  - អនុវត្តអនុគមន៍លើ module រងនីមួយៗ
* - `model.register_forward_hook(fn)`
  - ចាប់យក activation សម្រាប់ debug
* - `model.register_full_backward_hook(fn)`
  - ចាប់យក gradient
* - `model.zero_grad()`
  - សម្អាត gradient
```

{doc}`មើលការណែនាំ <neural-networks>`

### Loss Functions

```{list-table}
:header-rows: 1
:widths: 34 24 42

* - API
  - Target
  - ប្រើសម្រាប់
* - `nn.CrossEntropyLoss()`
  - int64 `(N,)`
  - multi-class (ទទួល **logits**)
* - `nn.BCEWithLogitsLoss()`
  - float `(N,)` ឬ `(N,C)`
  - binary / multi-label (ទទួល **logits**)
* - `nn.BCELoss()`
  - float
  - ⚠️ ទទួល probabilities — ប្រើ `BCEWithLogitsLoss` ជំនួស
* - `nn.NLLLoss()`
  - int64
  - ជាមួយ `LogSoftmax` ដោយដៃ
* - `nn.MSELoss()`
  - float
  - regression (L2)
* - `nn.L1Loss()`
  - float
  - regression (MAE) — ធន់នឹង outlier
* - `nn.SmoothL1Loss()` / `HuberLoss()`
  - float
  - regression — ល្អបំផុតសម្រាប់ភាគច្រើន
* - `nn.KLDivLoss()`
  - log-probabilities
  - distribution matching
* - `nn.PoissonNLLLoss()`
  - float
  - count data
* - `nn.CosineEmbeddingLoss()`
  - `-1` / `1`
  - similarity
```

ជម្រើសរួម៖ `weight=`, `reduction='mean'|'sum'|'none'`, `ignore_index=`, `label_smoothing=`

### torch.nn.functional

```{list-table}
:header-rows: 1
:widths: 36 64

* - API
  - ការពន្យល់
* - `F.relu(x)` / `gelu` / `silu` / `leaky_relu` / `elu`
  - activations
* - `F.softmax(x, dim)` / `F.log_softmax(x, dim)`
  - normalization ជា probabilities
* - `F.cross_entropy(logits, target)`
  - loss ដោយគ្មាន module
* - `F.mse_loss` / `l1_loss` / `binary_cross_entropy_with_logits`
  - losses
* - `F.dropout(x, p, training=...)`
  - ⚠️ ត្រូវបញ្ជាក់ `training=` ដោយដៃ
* - `F.one_hot(idx, num_classes)`
  - int64 → one-hot
* - `F.normalize(x, dim)`
  - L2 normalize
* - `F.pad(x, pad, mode)`
  - padding
* - `F.interpolate(x, size, mode)`
  - resize / upsample
* - `F.scaled_dot_product_attention(q, k, v)`
  - attention ដែល fuse រួច (លឿនជាង)
* - `F.embedding(input, weight)`
  - lookup
* - `F.linear(x, weight, bias)`
  - fully connected
* - `F.conv2d(x, weight, bias, stride, padding)`
  - convolution
```

## torch.optim

```{list-table}
:header-rows: 1
:widths: 32 68

* - API
  - ការពន្យល់
* - `optim.SGD(params, lr, momentum, weight_decay)`
  - stochastic gradient descent
* - `optim.Adam(params, lr, betas, eps, weight_decay)`
  - Adam
* - `optim.AdamW(params, lr, weight_decay)`
  - Adam ជាមួយ decoupled weight decay — **ណែនាំ**
* - `optim.RMSprop(params, lr, alpha)`
  - RMSprop
* - `optim.Adagrad(params, lr)`
  - Adagrad
* - `optim.Adamax` / `optim.ASGD` / `optim.Rprop` / `optim.LBFGS`
  - ជម្រើសផ្សេងទៀត
* - `optimizer.zero_grad(set_to_none=True)`
  - សម្អាត gradient
* - `optimizer.step()`
  - ធ្វើបច្ចុប្បន្នភាព parameter
* - `optimizer.state_dict()` / `load_state_dict(sd)`
  - រក្សាទុក / ផ្ទុក optimizer state
* - `optimizer.param_groups`
  - list នៃ groups ជាមួយ hyperparameters
* - `optimizer.add_param_group(dict)`
  - បន្ថែម parameter ថ្មីបន្ទាប់ពីបង្កើត
```

### torch.optim.lr_scheduler

```{list-table}
:header-rows: 1
:widths: 36 64

* - API
  - ការហៅ step()
* - `StepLR(opt, step_size, gamma)`
  - រាល់ epoch
* - `MultiStepLR(opt, milestones, gamma)`
  - រាល់ epoch
* - `ExponentialLR(opt, gamma)`
  - រាល់ epoch
* - `CosineAnnealingLR(opt, T_max, eta_min)`
  - រាល់ epoch
* - `CosineAnnealingWarmRestarts(opt, T_0, T_mult)`
  - រាល់ epoch
* - `ReduceLROnPlateau(opt, mode, factor, patience)`
  - ⚠️ **ជាមួយ metric** — `step(val_loss)`
* - `OneCycleLR(opt, max_lr, total_steps)`
  - ⚠️ **រាល់ step**
* - `LinearLR(opt, start_factor, total_iters)`
  - រាល់ epoch
* - `LambdaLR(opt, lr_lambda)`
  - រាល់ epoch
* - `SequentialLR(opt, schedulers, milestones)`
  - រាល់ epoch
* - `PolynomialLR(opt, total_iters, power)`
  - រាល់ epoch
```

{doc}`មើលការណែនាំ <loss-and-optimizers>`

## torch.utils.data

```{list-table}
:header-rows: 1
:widths: 36 64

* - API
  - ការពន្យល់
* - `Dataset`
  - map-style base class (`__len__` + `__getitem__`)
* - `IterableDataset`
  - iterable-style base class (`__iter__`)
* - `TensorDataset(*tensors)`
  - dataset ពី tensors ក្នុង memory
* - `ConcatDataset(datasets)`
  - ភ្ជាប់ datasets
* - `Subset(dataset, indices)`
  - ផ្នែកមួយនៃ dataset
* - `ChainDataset(datasets)`
  - ភ្ជាប់ IterableDatasets
* - `random_split(dataset, lengths, generator=)`
  - បំបែក train/val/test (⚠️ បញ្ជាក់ `generator=`)
* - `DataLoader(dataset, ...)`
  - batching + shuffling + workers
* - `DistributedSampler(dataset)`
  - សម្រាប់ DDP — ⚠️ ហៅ `set_epoch(epoch)`
* - `RandomSampler` / `SequentialSampler` / `WeightedRandomSampler`
  - samplers ផ្ទាល់ខ្លួន
* - `default_collate(batch)`
  - collate លំនាំដើម
```

### DataLoader parameters

`batch_size`, `shuffle`, `sampler`, `batch_sampler`, `num_workers`, `collate_fn`,
`pin_memory`, `drop_last`, `timeout`, `worker_init_fn`, `generator`,
`prefetch_factor`, `persistent_workers`

{doc}`មើលការណែនាំ <data-and-dataloader>`

## ការរក្សាទុក និងផ្ទុក

```{list-table}
:header-rows: 1
:widths: 38 62

* - API
  - ការពន្យល់
* - `torch.save(obj, path)`
  - serialize ដោយ pickle
* - `torch.load(path, map_location=, weights_only=True)`
  - deserialize (⚠️ `weights_only=True` ជាលំនាំដើមចាប់ពី 2.6)
* - `torch.jit.script(model)` / `torch.jit.trace(model, example)`
  - TorchScript
* - `torch.jit.load(path)` / `scripted.save(path)`
  - រក្សាទុក / ផ្ទុក TorchScript
* - `torch.export.export(model, args)`
  - ExportedProgram (PyTorch 2.x)
* - `torch.export.save(ep, path)` / `torch.export.load(path)`
  - រក្សាទុក / ផ្ទុក
* - `torch.onnx.export(model, args, path, ...)`
  - ONNX
* - `torch.compile(model, mode=, dynamic=)`
  - JIT compile ដោយ TorchInductor
```

{doc}`មើលការណែនាំ <saving-and-loading>` · {doc}`checkpoints`

## Mixed Precision និង Performance

```{list-table}
:header-rows: 1
:widths: 42 58

* - API
  - ការពន្យល់
* - `torch.amp.autocast(device_type, dtype)`
  - ប្ដូរប្រតិបត្តិការទៅ precision ទាបដោយស្វ័យប្រវត្តិ
* - `torch.amp.GradScaler(device, enabled)`
  - សម្រាប់ float16 — ការពារ gradient underflow
* - `scaler.scale(loss).backward()`
  - គុណ loss ឡើង បន្ទាប់មក backward
* - `scaler.step(optimizer)`
  - unscale + `optimizer.step()` (រំលងបើមាន inf)
* - `scaler.update()`
  - កែតម្រូវ scale factor
* - `scaler.unscale_(optimizer)`
  - ⚠️ ចាំបាច់មុន gradient clipping
* - `torch.set_float32_matmul_precision("highest"\|"high"\|"medium")`
  - TF32 control
* - `torch.backends.cudnn.benchmark`
  - ស្វែងរក cuDNN algorithm លឿនបំផុត
* - `torch.backends.cudnn.deterministic`
  - បង្ខំ algorithm deterministic
* - `torch.use_deterministic_algorithms(True)`
  - បង្ខំ determinism ទាំងស្រុង
* - `torch.get_num_threads()` / `set_num_threads(n)`
  - CPU threads
* - `torch.profiler.profile(...)`
  - profiling
* - `torch.nn.utils.clip_grad_norm_(params, max_norm)`
  - gradient clipping (តាម norm)
* - `torch.nn.utils.clip_grad_value_(params, clip_value)`
  - gradient clipping (តាមតម្លៃ)
* - `torch.utils.checkpoint.checkpoint(fn, *args)`
  - gradient/activation checkpointing សម្រាប់សន្សំ memory
```

{doc}`មើលការណែនាំ <mixed-precision>` · {doc}`performance`

## torch.func (functional API)

```{list-table}
:header-rows: 1
:widths: 34 66

* - API
  - ការពន្យល់
* - `torch.func.grad(fn)`
  - បង្កើតអនុគមន៍ដែលត្រឡប់ gradient
* - `torch.func.vmap(fn)`
  - vectorize អនុគមន៍លើ batch dimension
* - `torch.func.functional_call(module, params, args)`
  - forward pass ដោយប្រើ parameters ដែលបានផ្ដល់
* - `torch.func.jacrev` / `jacfwd` / `hessian`
  - Jacobian និង Hessian
```

## Distributed

```{list-table}
:header-rows: 1
:widths: 38 62

* - API
  - ការពន្យល់
* - `dist.init_process_group(backend)`
  - `"nccl"` (GPU) ឬ `"gloo"` (CPU)
* - `dist.destroy_process_group()`
  - សម្អាត
* - `dist.get_rank()` / `get_world_size()`
  - អត្តសញ្ញាណ process
* - `dist.all_reduce(tensor, op)`
  - reduce លើគ្រប់ rank
* - `dist.broadcast(tensor, src)`
  - ផ្ញើពី rank មួយ
* - `dist.all_gather(list, tensor)`
  - ប្រមូលពីគ្រប់ rank
* - `dist.barrier()`
  - ធ្វើសមកាលកម្ម
* - `nn.parallel.DistributedDataParallel`
  - DDP wrapper
* - `torch.distributed.fsdp.FullyShardedDataParallel`
  - FSDP សម្រាប់ model ធំ
```

{doc}`មើលការណែនាំ <distributed-training>`

## សន្លឹកបន្ទះសម្រាប់វដ្ដបណ្ដុះបណ្ដាល

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

model = MyModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)     # ១
        logits = model(xb)                        # ២
        loss = loss_fn(logits, yb)                # ៣
        loss.backward()                           # ៤
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                          # ៕

        running_loss += loss.item() * xb.size(0)  # ⚠️ .item() មិនមែន tensor

    scheduler.step()

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_loss += loss_fn(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
```
