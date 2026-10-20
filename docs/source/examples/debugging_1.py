"""កំហុស PyTorch ដែលជួបញឹកញាប់បំផុត — សារកំហុសពិត និងដំណោះស្រាយ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

ឯកសារនេះ **បង្កើតកំហុសដោយចេតនា** ក្នុង try/except ដើម្បីបង្ហាញសារកំហុសពិត។
រាល់កំហុសទាំងអស់ត្រូវបានចាប់យក — ស្គ្រីបនេះត្រូវតែបញ្ចប់ដោយជោគជ័យ។
"""

import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

CASES = []


def case(title: str):
    """Decorator ដែលចាប់យកកំហុស និងកត់ត្រាវា។"""
    def deco(fn):
        CASES.append((title, fn))
        return fn
    return deco


def show_error(exc: BaseException, max_lines: int = 3) -> None:
    lines = str(exc).strip().splitlines()
    for line in lines[:max_lines]:
        print("    │ " + line[:96])
    if len(lines) > max_lines:
        print(f"    │ … (+{len(lines) - max_lines} បន្ទាត់ទៀត)")


# =========================================================================== #
# កំហុសទាក់ទងនឹង shape
# =========================================================================== #
@case("RuntimeError: shape mismatch — matmul")
def _():
    torch.randn(4, 3) @ torch.randn(4, 5)


@case("RuntimeError: mat1 and mat2 shapes cannot be multiplied")
def _():
    nn.Linear(10, 5)(torch.randn(8, 12))      # Linear រំពឹង 10 features


@case("RuntimeError: The size of tensor a (3) must match tensor b (4)")
def _():
    torch.randn(2, 3) + torch.randn(2, 4)


@case("RuntimeError: view size is not compatible")
def _():
    x = torch.randn(4, 6).T                   # មិន contiguous
    x.view(24)


@case("ValueError: expected input to have at least 3 dimensions")
def _():
    nn.Conv1d(3, 8, 3)(torch.randn(4, 3))     # ត្រូវការ (N, C, L)


# =========================================================================== #
# កំហុសទាក់ទងនឹង dtype
# =========================================================================== #
@case("RuntimeError: expected scalar type Long but found Float (CrossEntropyLoss)")
def _():
    F.cross_entropy(torch.randn(4, 3), torch.tensor([0.0, 1.0, 2.0, 0.0]))


@case("RuntimeError: Found dtype Float but expected Long — Embedding")
def _():
    nn.Embedding(10, 4)(torch.tensor([1.0, 2.0, 3.0]))


@case("RuntimeError: can't convert cuda tensor / dtype promotion")
def _():
    torch.ones(3, dtype=torch.int64) + torch.ones(3, dtype=torch.float32) * 0  # ដំណើរការ
    # កំហុសពិត: ដាក់ float tensor ទៅក្នុង index
    x = torch.zeros(5)
    x[torch.tensor([1.5])]


@case("ValueError: too many dimensions 'str' — tensor ពី strings")
def _():
    torch.tensor(["cat", "dog"])


@case("RuntimeError: result type Float can't be cast to the desired output type Long")
def _():
    out = torch.zeros(3, dtype=torch.long)
    out.add_(torch.randn(3))


# =========================================================================== #
# កំហុសទាក់ទងនឹង device
# =========================================================================== #
@case("RuntimeError: Expected all tensors to be on the same device")
def _():
    if torch.cuda.is_available():
        torch.ones(3, device="cuda") + torch.ones(3, device="cpu")
    else:
        # គ្មាន GPU — បង្កើតកំហុស device ដោយប្រើ 'meta' device ជំនួស
        a = torch.ones(3, device="cpu")
        b = torch.empty(3, device="meta")
        _ = a + b


@case("TypeError: can't convert meta Tensor to numpy / .item()")
def _():
    t = torch.empty(3, device="meta")
    t.numpy()


# =========================================================================== #
# កំហុសទាក់ទងនឹង autograd
# =========================================================================== #
@case("RuntimeError: grad can be implicitly created only for scalar outputs")
def _():
    v = torch.randn(3, requires_grad=True)
    (v * 2).backward()


@case("RuntimeError: Trying to backward through the graph a second time")
def _():
    a = torch.tensor(2.0, requires_grad=True)
    y = a ** 2
    y.backward()
    y.backward()


@case("RuntimeError: element 0 of tensors does not require grad")
def _():
    x = torch.randn(3, requires_grad=True)
    y = x.detach() * 2
    y.sum().backward()


@case("RuntimeError: a leaf Variable that requires grad is being used in an in-place operation")
def _():
    leaf = torch.zeros(3, requires_grad=True)
    leaf += 1


@case("RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation")
def _():
    w = torch.tensor([2.0], requires_grad=True)
    z = w * 3                # z ត្រូវការ w សម្រាប់ backward
    w.mul_(2)                # in-place កែ w
    z.backward()


@case("UserWarning: Using a target size that is different to the input size")
def _():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # ប្រែក្លា warning ទៅជា error ដើម្បីបង្ហាញវា
        F.mse_loss(torch.randn(4, 1), torch.randn(4))


# =========================================================================== #
# កំហុសទាក់ទងនឹង module ការរក្សាទុក និងផ្ទុក
# =========================================================================== #
@case("RuntimeError: Error(s) in loading state_dict — size mismatch")
def _():
    sd = nn.Linear(4, 3).state_dict()
    nn.Linear(8, 3).load_state_dict(sd)


@case("AttributeError: 'Net' object has no attribute — ភ្លេច super().__init__()")
def _():
    class Bad(nn.Module):
        def __init__(self):
            pass                                  # ភ្លេច super().__init__()
        def forward(self, x):
            return x
    Bad().fc = nn.Linear(2, 2)


@case("RuntimeError: cannot assign module before Module.__init__() call")
def _():
    class Bad2(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(2, 2)             # មុន super().__init__()
            super().__init__()
    Bad2()


@case("ValueError: optimizer got an empty parameter list")
def _():
    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x
    torch.optim.SGD(Empty().parameters(), lr=0.1)


@case("TypeError: cannot pickle '_io.BytesIO' / បញ្ហា DataLoader num_workers")
def _():
    # បង្ហាញបញ្ហាដែលជួបញឹកញាប់: ព្យាយាម deepcopy object ដែលមិនអាច pickle បាន
    import copy
    import io
    copy.deepcopy(io.BytesIO())


# =========================================================================== #
# ដំណើរការករណីទាំងអស់
# =========================================================================== #
print("=" * 100)
print(f"កំហុស PyTorch ដែលជួបញឹកញាប់ — {len(CASES)} ករណី")
print("=" * 100)

n_caught = 0
for i, (title, fn) in enumerate(CASES, 1):
    print(f"\n{i:>2}. {title}")
    try:
        fn()
        print("    ⚠ គ្មានកំហុសកើតឡើងទេ (កំហុសនេះអាចអាស្រ័យលើ device)")
    except BaseException as exc:
        n_caught += 1
        print(f"    ✗ {type(exc).__name__}")
        show_error(exc)

print("\n" + "=" * 100)
print(f"បានចាប់យកកំហុស {n_caught}/{len(CASES)}")
assert n_caught >= len(CASES) - 2, "កំហុសភាគច្រើនគួរតែអាចបង្កើតឡើងវិញបាន"
print("OK — កំហុសទាំងអស់ត្រូវបានបង្ហាញដោយជោគជ័យ")
