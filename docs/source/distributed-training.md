# Distributed Training

ការបណ្ដុះបណ្ដាលលើ GPU ច្រើន ឬម៉ាស៊ីនច្រើន។ ទំព័រនេះពន្យល់គោលគំនិត និង API។

```{admonition} ទំព័រនេះមិនមានឧទាហរណ៍ដែលបានសាកល្បងទេ
:class: warning
⚠️ **មិនទាន់បានសាកល្បងដោយអ្នកនិពន្ធ។** ម៉ាស៊ីនដែលបង្កើតឯកសារនេះមានតែ CPU
ហើយគ្មាន multi-GPU ឬ multi-node cluster ទេ។

កូដទាំងអស់ខាងក្រោមផ្អែកលើ API ដែលបានផ្ទៀងផ្ទាត់ថា **មាន** ក្នុង PyTorch `2.14`
(ដោយ `hasattr()` checks) ប៉ុន្តែ **មិនបានដំណើរការ** ក្នុងបរិបទ distributed ពិតទេ។
សូមសាកល្បងលើ cluster របស់អ្នកមុនពឹងលើវា។
```

## តើអ្នកត្រូវការ Distributed Training ឬទេ?

```{list-table}
:header-rows: 1
:widths: 44 56

* - ស្ថានភាព
  - អ្វីដែលត្រូវប្រើ
* - Model សមក្នុង GPU តែមួយ ហើយការបណ្ដុះបណ្ដាលបញ្ចប់ក្នុងពេលដែលអ្នកទទួលយកបាន
  - **GPU តែមួយ** — សាមញ្ញបំផុត។ កុំបន្ថែមភាពស្មុគស្មាញដោយគ្មានហេតុផល
* - Model សមក្នុង GPU តែមួយ ប៉ុន្តែទិន្នន័យច្រើនពេក
  - **DDP (DistributedDataParallel)** — បង្កើនល្បឿនតាមចំនួន GPU
* - Model មិនសមក្នុង GPU តែមួយ
  - **FSDP** ឬ pipeline/tensor parallelism
* - Batch ធំពេកសម្រាប់ GPU memory ប៉ុន្តែអ្នកមាន GPU តែមួយ
  - **Gradient accumulation** — មើល {doc}`autograd`។ សាមញ្ញជាង DDP ច្រើន
```

## គោលគំនិត DDP

**DistributedDataParallel (DDP)** ដំណើរការដោយ៖

1. ចម្លង **model ដូចគ្នា** ទៅ GPU នីមួយៗ
2. Process នីមួយៗដំណើរការ **ផ្នែកផ្សេងគ្នានៃ batch** (តាម `DistributedSampler`)
3. បន្ទាប់ពី `backward()` DDP **ធ្វើ all-reduce** លើ gradients — គ្រប់ process ទទួលបានមធ្យមភាពដូចគ្នា
4. `optimizer.step()` ធ្វើការដូចគ្នានៅគ្រប់ process (ព្រោះ gradient និង weight ដូចគ្នា)

```text
GPU 0: batch 0-31   ──► backward ──► ┐
GPU 1: batch 32-63  ──► backward ──► ├──► all-reduce (មធ្យមភាព) ──► step
GPU 2: batch 64-95  ──► backward ──► │
GPU 3: batch 96-127 ──► backward ──► ┘
```

```{admonition} មិនដូច nn.DataParallel
:class: important
`nn.DataParallel` (DP) ប្រើ **process តែមួយ** ជាមួយ threads ច្រើន ហើយត្រូវបាន
រារាំងដោយ Python GIL។ វាក៏ត្រូវការការប្រមូលផ្ដុំ output នៅលើ GPU 0 ដែលបង្កើត
memory imbalance។ **PyTorch documentation ណែនាំឱ្យប្រើ DDP ជំនួសវិញ។**

DDP ប្រើ **មួយ process ក្នុងមួយ GPU** ដូច្នេះគ្មាន GIL contention ទេ។
```

## កូដ DDP មូលដ្ឋាន

```python
# ⚠️ មិនទាន់បានសាកល្បង — ត្រូវការ multi-GPU
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup():
    """ចាប់យកព័ត៌មានពី environment variables ដែល torchrun កំណត់។"""
    dist.init_process_group(backend="nccl")       # "gloo" សម្រាប់ CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    dataset = MyDataset()
    sampler = DistributedSampler(dataset)          # ← ចែកទិន្នន័យរវាង processes
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)                   # ← ចាំបាច់សម្រាប់ shuffle ត្រឹមត្រូវ
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()                        # DDP all-reduce gradients នៅទីនេះ
            optimizer.step()

    cleanup()


if __name__ == "__main__":
    main()
```

រត់ដោយ៖

```bash
# 4 processes លើម៉ាស៊ីនតែមួយ (GPU មួយក្នុងមួយ process)
torchrun --nproc_per_node=4 train.py

# Multi-node (2 ម៉ាស៊ីន × 8 GPU)
torchrun --nnodes=2 --nproc_per_node=8 \
         --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 \
         train.py
```

## ចំណុចសំខាន់ដែលងាយភ្លេច

### ១. sampler.set_epoch(epoch)

```python
for epoch in range(epochs):
    sampler.set_epoch(epoch)       # ⚠️ បើភ្លេច shuffle នឹងដូចគ្នារាល់ epoch!
```

`DistributedSampler` ប្រើ `seed + epoch` ដើម្បីច្របល់។ បើអ្នកមិនហៅ
`set_epoch()` ទេ **រាល់ epoch នឹងមានលំដាប់ទិន្នន័យដូចគ្នា** ដែលកាត់បន្ថយ
ប្រសិទ្ធភាពនៃការបណ្ដុះបណ្ដាលយ៉ាងខ្លាំង។ វាមិនបង្ហាញ error ណាមួយឡើយ។

### ២. កុំ shuffle ក្នុង DataLoader

```python
# ❌ ខុស — shuffle និង sampler មិនអាចប្រើជាមួយគ្នាបានទេ
DataLoader(dataset, shuffle=True, sampler=sampler)   # ValueError

# ✅ ត្រូវ
DataLoader(dataset, sampler=sampler)                 # sampler គ្រប់គ្រងលំដាប់
```

### ៣. Effective batch size

```text
effective_batch = batch_size × world_size × gradient_accumulation_steps
```

បើអ្នកផ្លាស់ពី GPU 1 ទៅ GPU 4 ដោយរក្សា `batch_size=32` នោះ effective batch
កើនពី 32 ទៅ 128។ អ្នក **ត្រូវកែ learning rate** ជាធម្មតា (linear scaling rule:
lr × 4 ឬ sqrt scaling: lr × 2)។ បើមិនដូច្នោះទេលទ្ធផលនឹងខុសពីការពិសោធន៍ GPU តែមួយ។

### ៤. Logging និងរក្សាទុកតែនៅ rank 0

```python
if dist.get_rank() == 0:
    print(f"epoch {epoch}: loss = {loss.item():.4f}")
    torch.save(model.module.state_dict(), "best.pt")
dist.barrier()                    # រង់ចាំឱ្យ rank 0 បញ្ចប់
```

បើ process ទាំងអស់សរសេរទៅឯកសារដូចគ្នា អ្នកនឹងទទួលបានឯកសារខូច។
បើពួកវា print ទាំងអស់ log នឹងមាន 4 ដងសម្រាប់រាល់បន្ទាត់។

### ៥. model.module សម្រាប់ state_dict

```python
# ❌ key នឹងមាន prefix "module."
torch.save(model.state_dict(), path)

# ✅
torch.save(model.module.state_dict(), path)
```

មើល {doc}`checkpoints` សម្រាប់ការពន្យល់លម្អិត។

### ៦. Metrics ត្រូវ all-reduce

```python
correct = (preds == labels).sum()
total = torch.tensor(len(labels), device=device)

# ប្រមូលពីគ្រប់ process មុនគណនា accuracy
dist.all_reduce(correct, op=dist.ReduceOp.SUM)
dist.all_reduce(total, op=dist.ReduceOp.SUM)

accuracy = correct.item() / total.item()
```

បើអ្នកមិន all-reduce ទេ អ្នកនឹងរាយការណ៍ accuracy នៃ **1/N នៃទិន្នន័យប៉ុណ្ណោះ**។

## Backends

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Backend
  - ប្រើសម្រាប់
  - ចំណាំ
* - `nccl`
  - NVIDIA GPU
  - ល្អបំផុតសម្រាប់ multi-GPU។ លំនាំដើមសម្រាប់ CUDA
* - `gloo`
  - CPU និង debugging
  - ដំណើរការគ្រប់ទីកន្លែង។ យឺតជាង NCCL លើ GPU
* - `mpi`
  - HPC clusters
  - ត្រូវ build PyTorch ជាមួយ MPI support
```

```python
backend = "nccl" if torch.cuda.is_available() else "gloo"
dist.init_process_group(backend=backend)
```

## FSDP (Fully Sharded Data Parallel)

សម្រាប់ model ដែល **មិនសមក្នុង GPU តែមួយ** FSDP បំបែក parameters, gradients
និង optimizer state រវាង GPUs៖

```python
# ⚠️ មិនទាន់បានសាកល្បង
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = MyLargeModel().to(device)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3 ស្មើ
    device_id=local_rank,
)
```

```{list-table}
:header-rows: 1
:widths: 30 70

* - ShardingStrategy
  - អ្វីដែលត្រូវបានបំបែក
* - `FULL_SHARD`
  - parameters + gradients + optimizer state (memory តិចបំផុត ល្បឿនយឺតបំផុត)
* - `SHARD_GRAD_OP`
  - gradients + optimizer state
* - `NO_SHARD`
  - គ្មាន (ស្មើនឹង DDP)
* - `HYBRID_SHARD`
  - shard ក្នុង node, replicate រវាង nodes
```

PyTorch 2.4+ ក៏មាន **FSDP2** (`torch.distributed.fsdp.fully_shard`) ដែលជា
per-parameter sharding API ថ្មី។ សូមមើលឯកសារផ្លូវការ PyTorch សម្រាប់ការណែនាំបច្ចុប្បន្ន។

## Device Mesh (PyTorch 2.x)

```python
# ⚠️ មិនទាន់បានសាកល្បង
from torch.distributed.device_mesh import init_device_mesh

mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
# dp = data parallel ខ្នាត 2, tp = tensor parallel ខ្នាត 4
```

Device mesh អនុញ្ញាតឱ្យអ្នកកំណត់ parallelism ច្រើនប្រភេទក្នុងពេលតែមួយ
(2D/3D parallelism) សម្រាប់ model ធំខ្លាំង។

## Communication Primitives

```{list-table}
:header-rows: 1
:widths: 26 74

* - អនុគមន៍
  - អ្វីដែលវាធ្វើ
* - `dist.all_reduce(tensor, op)`
  - គ្រប់ rank ទទួលបានលទ្ធផលនៃការបូក/អតិបរមា ។ល។ លើ tensor របស់គ្រប់ rank
* - `dist.broadcast(tensor, src)`
  - ផ្ញើពី rank `src` ទៅគ្រប់ rank
* - `dist.all_gather(tensor_list, tensor)`
  - ប្រមូល tensor ពីគ្រប់ rank ទៅក្នុង list
* - `dist.reduce_scatter(output, input_list)`
  - reduce ហើយចែកផ្នែកទៅ rank នីមួយៗ
* - `dist.barrier()`
  - រង់ចាំឱ្យគ្រប់ rank មកដល់ចំណុចនេះ
* - `dist.get_rank()` / `dist.get_world_size()`
  - អត្តសញ្ញាណ process និងចំនួន process សរុប
```

## ការដោះស្រាយបញ្ហា

```{list-table}
:header-rows: 1
:widths: 34 66

* - រោគសញ្ញា
  - ដំណោះស្រាយ
* - Hang នៅ `init_process_group`
  - Firewall រារាំង port។ ពិនិត្យ `MASTER_ADDR`, `MASTER_PORT`, និង `NCCL_SOCKET_IFNAME`
* - Hang នៅ `backward()`
  - Rank មួយមិនបានចូលរួម។ ពិនិត្យថាគ្រប់ rank រត់ code ដូចគ្នា (គ្មាន `if rank == 0` ជុំវិញ forward)
* - `NCCL error: unhandled system error`
  - បញ្ហា driver/NCCL version។ រត់ជាមួយ `NCCL_DEBUG=INFO` ដើម្បីមើល log
* - ល្បឿនមិនប្រសើរឡើងជាមួយ GPU ច្រើន
  - Data loading ជា bottleneck (បង្កើន `num_workers`), ឬ PCIe bandwidth មានកំណត់
* - `RuntimeError: Expected to have finished reduction in the prior iteration`
  - Parameter ខ្លះមិនបានចូលរួមក្នុង forward។ ប្រើ `find_unused_parameters=True` (យឺតជាង)
* - Loss ខុសពីការរត់ GPU តែមួយ
  - lr មិនបាន scale តាម effective batch size ឬភ្លេច `set_epoch()`
* - OOM លើ rank 0 តែប៉ុណ្ណោះ
  - ការប្រមូលផ្ដុំ metric/logging នៅ rank 0។ ប្រើ all-reduce ជំនួស gather
```

## សង្ខេប

```{admonition} ចំណុចសំខាន់
:class: tip
1. **ប្រើ DDP មិនមែន DataParallel**។
2. **`DistributedSampler` + `sampler.set_epoch(epoch)`** — ភ្លេច `set_epoch` គឺជា silent bug។
3. **កុំប្រើ `shuffle=True` ជាមួយ sampler**។
4. **Scale learning rate** តាម effective batch size។
5. **Logging និង checkpoint តែនៅ rank 0** បន្ទាប់មក `dist.barrier()`។
6. **All-reduce metrics** មុនរាយការណ៍។
7. **`model.module.state_dict()`** សម្រាប់រក្សាទុក។
8. **`nccl` សម្រាប់ GPU · `gloo` សម្រាប់ CPU**។
9. **FSDP** បើ model មិនសមក្នុង GPU តែមួយ។
```

```{admonition} សូមផ្ទៀងផ្ទាត់ជាមួយឯកសារផ្លូវការ
:class: seealso
Distributed training API វិវត្តលឿន។ សម្រាប់ព័ត៌មានបច្ចុប្បន្ន សូមមើល
[PyTorch Distributed Overview](https://docs.pytorch.org/docs/stable/distributed.html)
ដែលជាប្រភពផ្លូវការ។
```
