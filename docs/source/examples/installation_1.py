"""ពិនិត្យ Environment និងកំណែ PyTorch របស់អ្នក។

ដំណើរការលើ៖ CPU ឬ GPU (មិនត្រូវការ GPU ទេ)
អ្នកអាចរត់ឯកសារនេះដោយផ្ទាល់៖  python installation_1.py
"""

import platform
import sys

import torch

print("=" * 58)
print("PyTorch Environment Check")
print("=" * 58)

# ១. កំណែ Python និង PyTorch
print(f"Python        : {sys.version.split()[0]}")
print(f"PyTorch       : {torch.__version__}")
print(f"Platform      : {platform.system()} {platform.release()} ({platform.machine()})")

# ២. កំណែ CUDA ដែល PyTorch ត្រូវបាន build ជាមួយ
#    ចំណាំ៖ torch.version.cuda អាចមានតម្លៃ ទោះបីជាម៉ាស៊ីនគ្មាន GPU ក៏ដោយ
#    ព្រោះវាប្រាប់ពី CUDA runtime ក្នុង wheel មិនមែន GPU របស់អ្នកទេ។
print(f"CUDA (build)  : {torch.version.cuda}")

# ៣. តើ GPU អាចប្រើបានឬទេ?
cuda_ok = torch.cuda.is_available()
print(f"CUDA available: {cuda_ok}")

if cuda_ok:
    print(f"  device count: {torch.cuda.device_count()}")
    print(f"  device name : {torch.cuda.get_device_name(0)}")
    print(f"  capability  : {torch.cuda.get_device_capability(0)}")
else:
    print("  (គ្មាន GPU — ឧទាហរណ៍ទាំងអស់ក្នុងឯកសារនេះដំណើរការលើ CPU បាន)")

# ៤. Apple Silicon (MPS backend)
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"MPS available : {mps_ok}")

# ៥. ចំនួន thread សម្រាប់ CPU
print(f"CPU threads   : {torch.get_num_threads()}")

# ៦. សាកល្បងថា PyTorch ដំណើរការពិត — បង្កើត tensor និងគណនា
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
y = x @ x.T  # matrix multiply: (2,3) @ (3,2) -> (2,2)
print("-" * 58)
print("Smoke test:  x @ x.T")
print(y)
assert y.shape == (2, 2), f"រំពឹង shape (2,2) តែទទួល {tuple(y.shape)}"
print("=" * 58)
print("OK — PyTorch ដំណើរការត្រឹមត្រូវ")
