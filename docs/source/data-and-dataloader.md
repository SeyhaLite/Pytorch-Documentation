# Dataset និង DataLoader

`torch.utils.data` គ្រប់គ្រងការផ្ទុកទិន្នន័យ។ វាបែងចែកជាពីរផ្នែក៖
**`Dataset`** (របៀបទទួលបានសំណាកមួយ) និង **`DataLoader`** (របៀបបង្កើត batch)។

```{admonition} តម្រូវការ
:class: note
🖥️ CPU គ្រប់គ្រាន់។ ឧទាហរណ៍ទាំង ៣ បានសាកល្បងលើ PyTorch `2.14.1+cpu`។
ឯកសារនេះមិនពឹងលើ `torchvision` ទេ — ដូច្នេះអ្នកអាចរត់វាដោយគ្រាន់តែមាន `torch`។
```

## ១. Dataset

`Dataset` គឺជា class ដែលមានតែពីរអនុគមន៍ចាំបាច់៖

```python
class MyDataset(Dataset):
    def __len__(self) -> int:
        ...                       # ចំនួនសំណាក

    def __getitem__(self, index):
        ...                       # ត្រឡប់សំណាកទី index
```

```{literalinclude} examples/data_and_dataloader_1.py
:language: python
:caption: data_and_dataloader_1.py — ✅ បានសាកល្បងលើ CPU
```

### ប្រភេទ Dataset

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - ប្រភេទ
  - អ្វីដែលត្រូវអនុវត្ត
  - ប្រើពេល
* - `Dataset` (map-style)
  - `__len__` + `__getitem__`
  - ទិន្នន័យដែលអ្នកអាចចូលដោយ index។ លំនាំដើមសម្រាប់ភាគច្រើន
* - `IterableDataset` (iterable-style)
  - `__iter__`
  - ទិន្នន័យហូរ (streaming) ដែលមិនដឹងទំហំជាមុន ឬធំពេកសម្រាប់ RAM
* - `TensorDataset`
  - គ្មាន (មានស្រាប់)
  - tensor ទាំងអស់នៅក្នុង memory រួចហើយ
* - `ConcatDataset`
  - គ្មាន
  - ភ្ជាប់ datasets ច្រើនចូលគ្នា
* - `Subset`
  - គ្មាន
  - យកផ្នែកមួយនៃ dataset (ឧ. សម្រាប់ debug)
* - `random_split()`
  - គ្មាន
  - បំបែក train/val/test
```

### ការរចនា __getitem__

```{admonition} ទម្លាប់ល្អសម្រាប់ __getitem__
:class: important
1. **អានឯកសារនៅក្នុង `__getitem__` មិនមែន `__init__`** — បើអ្នកអានទាំងអស់ក្នុង
   `__init__` អ្នកនឹងបាត់បង់អត្ថប្រយោជន៍នៃ `num_workers > 0`។
2. **ត្រឡប់ tensor** (ឬអ្វីដែល `default_collate` អាចដាក់ជា batch បាន)។
3. **អនុវត្ត transform នៅទីនេះ** មិនមែននៅក្នុង training loop។
4. **កុំធ្វើការចៃដន្យដោយគ្មាន seed** ក្នុង worker — មើលផ្នែក reproducibility។
5. **គាំទ្រ negative index** ដោយស្វ័យប្រវត្តិបើអ្នកប្រើ list/tensor indexing។
```

គំរូសម្រាប់គម្រោងពិត (អានរូបភាពពី disk)៖

```python
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageFolderLite(Dataset):
    """ឧទាហរណ៍ត្រូវការ torchvision — មិនបានសាកល្បងក្នុង CI របស់ឯកសារនេះទេ។"""

    def __init__(self, root: str, transform=None) -> None:
        self.paths = sorted(Path(root).rglob("*.jpg"))
        self.labels = {p.parent.name: i for i, p in enumerate(
            sorted({q.parent.name for q in self.paths}))}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = read_image(str(path)).float() / 255.0     # (C, H, W)
        label = self.labels[path.parent.name]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
```

## ២. DataLoader

`DataLoader` ខ្ចប់ `Dataset` ហើយផ្ដល់ batches៖

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # សម្រាប់ train។ False សម្រាប់ val/test
    num_workers=4,          # processes សម្រាប់ផ្ទុកទិន្នន័យ
    pin_memory=True,        # សម្រាប់ GPU
    drop_last=False,        # True សម្រាប់ BatchNorm ជាមួយ batch មិនទៀងទាត់
    collate_fn=None,        # របៀបបញ្ចូលសំណាកចូលគ្នា
    generator=None,         # torch.Generator សម្រាប់ shuffle reproducibility
    persistent_workers=True,  # កុំបង្កើត workers ឡើងវិញរាល់ epoch
)
```

```{literalinclude} examples/data_and_dataloader_2.py
:language: python
:caption: data_and_dataloader_2.py — ✅ បានសាកល្បងលើ CPU (batching, shuffling, collate_fn, generator)
```

### Parameters សំខាន់ៗ

```{list-table}
:header-rows: 1
:widths: 22 16 62

* - ប៉ារ៉ាម៉ែត្រ
  - លំនាំដើម
  - ការណែនាំ
* - `batch_size`
  - `1`
  - ធំជាង = gradient ស្ថិរជាង ប៉ុន្តែ memory ច្រើនជាង។ ចាប់ផ្ដើមពី 32 ឬ 64
* - `shuffle`
  - `False`
  - **`True` សម្រាប់ train ជានិច្ច**។ `False` សម្រាប់ val/test
* - `num_workers`
  - `0`
  - ចាប់ផ្ដើមពី 0។ បង្កើនដល់ 4–8 បើ `__getitem__` ធ្ងន់ (អានឯកសារ, decode រូបភាព)
* - `drop_last`
  - `False`
  - `True` បើ batch ចុងក្រោយតូចពេកសម្រាប់ BatchNorm
* - `pin_memory`
  - `False`
  - `True` ពេលបណ្ដុះបណ្ដាលលើ GPU
* - `persistent_workers`
  - `False`
  - `True` ពេល `num_workers > 0` និង dataset ធ្ងន់ក្នុងការ init
* - `prefetch_factor`
  - `2`
  - ចំនួន batches ដែល worker មួយផ្ទុកជាមុន (តែពេល `num_workers > 0`)
* - `collate_fn`
  - `default_collate`
  - ផ្ដល់ផ្ទាល់ខ្លួនសម្រាប់សំណាកដែលមានប្រវែងមិនស្មើគ្នា
* - `generator`
  - `None`
  - ផ្ដល់ `torch.Generator().manual_seed(...)` សម្រាប់ reproducibility
```

### ចំនួន batches

```text
drop_last=False :  len(loader) = ceil(len(dataset) / batch_size)
drop_last=True  :  len(loader) = floor(len(dataset) / batch_size)
```

ឧទាហរណ៍៖ សំណាក 100, `batch_size=16` →
- `drop_last=False`: 7 batches (6 × 16 + 1 × 4)
- `drop_last=True`: 6 batches (ទម្លាក់ 4 សំណាកចុងក្រោយ)

## ៣. collate_fn

`default_collate` ដាក់ tensors ជាមួយគ្នាដោយស្វ័យប្រវត្តិ។ អ្នកត្រូវការ `collate_fn`
ផ្ទាល់ខ្លួនពេលសំណាក **មានប្រវែងមិនស្មើគ្នា** (ឧ. ប្រយោគ)៖

```python
def pad_collate(batch):
    """Pad sequences ទៅប្រវែងអតិបរមានៃ batch នេះ។"""
    lengths = [seq.shape[0] for seq, _ in batch]
    max_len = max(lengths)
    dim = batch[0][0].shape[1]

    padded = torch.zeros(len(batch), max_len, dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.tensor([lab for _, lab in batch])

    for i, (seq, _) in enumerate(batch):
        padded[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = True
    return {"seq": padded, "mask": mask, "labels": labels, "lengths": lengths}
```

**`mask`** គឺសំខាន់ — ម៉ូដែលត្រូវដឹងថាធាតុណាជា padding ដើម្បីមិនគណនាលើវា។

````{admonition} Batch-level transform លឿនជាង per-sample
:class: tip
បើ transform របស់អ្នកមិនមានភាពចៃដន្យក្នុងមួយសំណាក (ឧ. normalization)
ធ្វើវានៅក្នុង `collate_fn` លើ batch ទាំងមូល៖

```python
def batch_collate(batch):
    xs = torch.stack([b[0] for b in batch]).float()
    ys = torch.stack([b[1] for b in batch])
    xs = (xs - mean) / std          # មួយប្រតិបត្តិការជំនួស 64
    return xs, ys
```

ប៉ុន្តែ **augmentation ដែលមានភាពចៃដន្យត្រូវធ្វើក្នុង `__getitem__`** ព្រោះសំណាក
នីមួយៗត្រូវការភាពចៃដន្យដាច់ដោយឡែក។
````

## ៤. Transforms និង Augmentation

```{literalinclude} examples/data_and_dataloader_3.py
:language: python
:caption: data_and_dataloader_3.py — ✅ បានសាកល្បងលើ CPU
```

### គោលការណ៍សំខាន់បំផុត

````{admonition} គណនា statistics ពី training set តែប៉ុណ្ណោះ
:class: danger
```python
# ❌ ខុស — data leakage
mu, sigma = all_data.mean(0), all_data.std(0)

# ✅ ត្រូវ
train_ds, val_ds = random_split(all_data, [0.8, 0.2], generator=g)
mu = all_data[train_ds.indices].mean(0)      # ឬគណនាពី train split
sigma = all_data[train_ds.indices].std(0)

train_tf = Standardize(mu, sigma)            # ប្រើ statistics ដូចគ្នាសម្រាប់ទាំងពីរ
val_tf = Standardize(mu, sigma)
```

បើអ្នករួមបញ្ចូលទិន្នន័យ validation ក្នុងការគណនា mean/std អ្នកបានលេចធ្លាយ
ព័ត៌មានអំពីវាទៅក្នុងការបណ្ដុះបណ្ដាល។ Metric របស់អ្នកនឹងល្អជាងការពិត។
````

```{admonition} Augmentation តែសម្រាប់ training
:class: important
`RandomCrop`, `RandomFlip`, `ColorJitter` ។ល។ គួរប្រើតែក្នុង training transform។
Validation/test transform គួរមានតែ `Resize` + `Normalize` ប៉ុណ្ណោះ — បើមិនដូច្នោះទេ
metric របស់អ្នកនឹងមាន noise ដែលមិនចាំបាច់។
```

### ប្រើ torchvision transforms (គម្រោងពិត)

```python
# ⚠️ ត្រូវការ torchvision — មិនបានសាកល្បងក្នុង CI របស់ឯកសារនេះទេ
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),                       # PIL -> float tensor [0,1], CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),   # ImageNet statistics
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

## ៥. បំបែក Train / Validation / Test

```python
from torch.utils.data import random_split

n = len(dataset)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
n_test = n - n_train - n_val

g = torch.Generator().manual_seed(42)          # ← ចាំបាច់សម្រាប់ reproducibility
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)
```

```{admonition} បញ្ជាក់ generator ជានិច្ចក្នុង random_split
:class: warning
បើអ្នកមិនបញ្ជាក់ទេ `random_split` ប្រើ global RNG។ នេះមានន័យថា **ការបំបែកអាស្រ័យ
លើអ្វីដែលបានរត់មុនវា**។ ការរត់ពីរដងដែលមានកូដដូចគ្នាអាចផ្ដល់ split ខុសគ្នា
ប្រសិនបើមានការហៅ random ផ្សេងនៅចន្លោះ។

ចង់បាន split ដូចគ្នារាល់ការរត់ → បញ្ជាក់ `generator=`។
មើល {doc}`reproducibility`។
```

## ៦. num_workers — ពេលណាប្រើ?

```{list-table}
:header-rows: 1
:widths: 40 60

* - ស្ថានភាព
  - num_workers ណែនាំ
* - Tensor ទាំងអស់នៅក្នុង RAM រួចហើយ (`TensorDataset`)
  - `0` — worker processes មិនជួយអ្វីទេ គ្រាន់តែបន្ថែម overhead
* - អានរូបភាពពី disk + decode
  - `4` – `8` (ឬចំនួន CPU cores)
* - Transform ធ្ងន់ (augmentation ច្រើន)
  - `4` – `8`
* - Debugging
  - `0` — ដើម្បីឱ្យ traceback ច្បាស់ និង deterministic
* - Windows
  - ចាប់ផ្ដើមពី `0` ឬ `2` — multiprocessing មាន overhead ខ្ពស់ជាង
```

````{admonition} num_workers > 0 ត្រូវការ worker seeding
:class: important
Worker process នីមួយៗទទួល seed ផ្ទាល់ខ្លួនដោយស្វ័យប្រវត្តិសម្រាប់ `torch`
ប៉ុន្តែ **មិនមែន** សម្រាប់ NumPy ឬ Python `random`។ បើ transform របស់អ្នកប្រើ
`np.random` សំណាកនឹងដូចគ្នាក្នុងគ្រប់ worker៖

```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

loader = DataLoader(ds, batch_size=32, num_workers=4,
                    worker_init_fn=seed_worker, generator=g)
```
````

## កំហុសទូទៅ

```{list-table}
:header-rows: 1
:widths: 34 30 36

* - សារកំហុស
  - មូលហេតុ
  - ដំណោះស្រាយ
* - `DataLoader worker (pid X) is killed by signal: Killed`
  - RAM អស់ (workers ច្រើនពេក)
  - បន្ថយ `num_workers` ឬ `batch_size`
* - `an error occurred while trying to pickle ...` ជាមួយ `num_workers > 0`
  - Dataset មាន object ដែលមិនអាច pickle បាន
  - ផ្លាស់ទី object នោះទៅ `__getitem__` ឬកំណត់ `num_workers=0`
* - `RuntimeError: DataLoader worker ... exited unexpectedly`
  - កំហុសក្នុង `__getitem__` ដែលត្រូវបានលាក់
  - រត់ជាមួយ `num_workers=0` ដើម្បីឃើញ traceback ពិត
* - `stack expects each tensor to be equal size`
  - សំណាកមាន shape ខុសគ្នា
  - ផ្ដល់ `collate_fn` ដែល pad
* - `each element in list of batch should be of equal size`
  - ដូចគ្នា
  - ដូចគ្នា
* - `Too many workers` / យឺតខ្លាំង
  - `num_workers` ច្រើនពេកសម្រាប់ dataset តូច
  - ប្រើ `0` សម្រាប់ dataset ក្នុង RAM
* - metric ប្រែប្រួលរាល់ការរត់
  - គ្មាន `generator=` លើ loader/split
  - បញ្ជាក់ `generator` ជានិច្ច
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **`Dataset` = សំណាកមួយ · `DataLoader` = batch**។
2. **អានឯកសារក្នុង `__getitem__`** មិនមែន `__init__` — ដើម្បីឱ្យ `num_workers` មានប្រយោជន៍។
3. **`shuffle=True` សម្រាប់ train, `False` សម្រាប់ val/test**។
4. **`collate_fn` សម្រាប់សំណាកប្រវែងមិនស្មើ** — ត្រឡប់ tensor + mask។
5. **គណនា normalization statistics ពី train split តែប៉ុណ្ណោះ**។
6. **Augmentation តែក្នុង training transform**។
7. **បញ្ជាក់ `generator=`** សម្រាប់ `shuffle` និង `random_split` ដើម្បី reproducibility។
8. **`num_workers=0` ពេល debug** — traceback ច្បាស់ និង deterministic។
```

## មេរៀនបន្ទាប់

{doc}`training-loop` — ដាក់អ្វីៗទាំងអស់ចូលគ្នាជាវដ្ដបណ្ដុះបណ្ដាលពេញលេញ។
