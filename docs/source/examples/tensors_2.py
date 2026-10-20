"""លក្ខណៈសម្បត្តិ tensor៖ shape, dtype, device, stride, memory។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
គន្លឹះ៖ យល់ពី stride ដើម្បីដឹងថាហេតុអ្វី .view() បរាជ័យ។
"""

import torch

x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

print("x =")
print(x)
print("-" * 50)
print(f"shape          : {tuple(x.shape)}        # (2, 3, 4)")
print(f"size(1)        : {x.size(1)}             # វិមាត្រទី 1")
print(f"numel()        : {x.numel()}            # ចំនួនធាតុសរុប = 2*3*4")
print(f"ndim           : {x.ndim}")
print(f"dtype          : {x.dtype}")
print(f"device         : {x.device}")
print(f"element_size() : {x.element_size()} បៃ (float32 = 4 bytes)")
print(f"stride()       : {x.stride()}       # ចម្ងាយរវាងធាតុនីមួយៗតាមវិមាត្រ")
print(f"is_contiguous(): {x.is_contiguous()}")

assert x.shape == (2, 3, 4)
assert x.numel() == 24
assert x.stride() == (12, 4, 1)   # 12 = 3*4, 4 = 4, 1

# ---------------------------------------------------------------------------
# stride ពន្យល់ពី "ការមើលឃើញ" លើ memory តែមួយ
# ---------------------------------------------------------------------------
# memory ពិតគឺបន្ទាត់តែមួយ (24 ធាតុ)។ stride (12,4,1) ប្រាប់ PyTorch ពីរបៀប
# គណនា index: x[i,j,k] = memory[i*12 + j*4 + k*1]
flat = x.flatten()
assert x[1, 2, 3].item() == flat[1 * 12 + 2 * 4 + 3].item()
print("-" * 50)
print("x[1,2,3] =", x[1, 2, 3].item(),
      "| flat[1*12+2*4+3] =", flat[23].item(), "-> ដូចគ្នា")

# transpose ធ្វើឱ្យ tensor មិន contiguous (stride ប្ដូរលំដាប់)
xt = x.transpose(0, 1)
print(f"\ntranspose(0,1) shape  : {tuple(xt.shape)}")
print(f"transpose(0,1) stride : {xt.stride()}")
print(f"transpose(0,1) contig : {xt.is_contiguous()}")
assert xt.shape == (3, 2, 4)
assert not xt.is_contiguous()

# ---------------------------------------------------------------------------
# ការចម្លង vs ការចែក memory រួមគ្នា
# ---------------------------------------------------------------------------
print("-" * 50)
view = x.view(6, 4)          # ចែក memory រួមគ្នា (មិនចម្លងទេ)
copy = x.clone()             # ចម្លងទិន្នន័យពិតប្រាកដ

view[0, 0] = 999.0
print(f"កែ view[0,0]=999 -> x[0,0,0] = {x[0, 0, 0].item()}  (ប្ដូរតាម!)")
print(f"កែ view[0,0]=999 -> copy[0,0,0] = {copy[0, 0, 0].item()}  (មិនប្ដូរ)")
assert x[0, 0, 0].item() == 999.0
assert copy[0, 0, 0].item() == 0.0

# ---------------------------------------------------------------------------
# ការប្ដូរ dtype និង device
# ---------------------------------------------------------------------------
print("-" * 50)
half = x.to(torch.float16)
double = x.to(torch.float64)
print(f".to(float16) : {half.dtype}")
print(f".to(float64) : {double.dtype}")

# ប្ដូរទៅ GPU បើមាន — បើមិនមាន នៅលើ CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
moved = x.to(device)
print(f".to('{device}')   : device = {moved.device}")
assert moved.device.type == device

print("-" * 50)
print("OK — លក្ខណៈសម្បត្តិ tensor ទាំងអស់ត្រឹមត្រូវ")
