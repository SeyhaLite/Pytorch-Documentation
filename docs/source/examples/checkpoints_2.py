"""កំហុសធម្មតាជាមួយ checkpoint និងវិធីដោះស្រាយ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

torch.manual_seed(0)
WORKDIR = Path(os.environ.get("KH_EXAMPLE_TMPDIR", tempfile.gettempdir()))


class Net(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(4, 8)

# ធ្វើឱ្យ optimizer មាន state ពិតប្រាកដ (Adam moments)
nn.CrossEntropyLoss()(model(x), torch.randint(0, 3, (4,))).backward()
opt.step()

CKPT = WORKDIR / "debug_ckpt.pt"
torch.save({"model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": 5}, CKPT)

print("=" * 70)
print("កំហុសទី ១: រក្សាទុក checkpoint ពី model ដែលខ្ចប់ដោយ DataParallel")
print("=" * 70)
print("  ខុស : torch.save(model.state_dict())      # key មាន 'module.' នាំមុខ")
print("  ត្រូវ: torch.save(model.module.state_dict())")

fake_dp = {"module." + k: v for k, v in model.state_dict().items()}
try:
    Net().load_state_dict(fake_dp)
except RuntimeError as e:
    print("\n  បើផ្ទុកដោយផ្ទាល់ -> RuntimeError:")
    for line in str(e).splitlines()[:3]:
        print("   ", line[:74])

# ដំណោះស្រាយ: ដក prefix ចេញ
from collections import OrderedDict

cleaned = OrderedDict((k.replace("module.", "", 1), v) for k, v in fake_dp.items())
Net().load_state_dict(cleaned)
print("\n  ដំណោះស្រាយ: ដក 'module.' prefix ចេញ -> ផ្ទុកបាន ✓")
print("  (ការពារល្អបំផុត: រក្សាទុក model.module.state_dict() តាំងពីដើម)")

print()
print("=" * 70)
print("កំហុសទី ២: រក្សាទុកលើ GPU ផ្ទុកលើ CPU")
print("=" * 70)
print("  ខុស : torch.load(path)                    # នឹងព្យាយាមដាក់លើ cuda:0")
print("  ត្រូវ: torch.load(path, map_location='cpu')")
print("  ឬល្អជាង: torch.load(path, map_location=device)")
print("  ចំណាំ: គ្មាន GPU នៅទីនេះ ដូច្នេះមិនអាចបង្ហាញកំហុសពិតបានទេ")

print()
print("=" * 70)
print("កំហុសទី ៣: ភ្លេច model.eval() បន្ទាប់ពីផ្ទុក checkpoint សម្រាប់ inference")
print("=" * 70)
m = Net()
m.load_state_dict(torch.load(CKPT, weights_only=False)["model_state_dict"])
print(f"  training flag បន្ទាប់ពី load_state_dict: {m.training}")
print("  ⚠ វានៅតែ True! ដូច្នេះ Dropout/BatchNorm នៅក្នុង training mode")
assert m.training is True
m.eval()
print(f"  បន្ទាប់ model.eval(): {m.training} ✓")

print()
print("=" * 70)
print("កំហុសទី ៤: architecture ផ្លាស់ប្ដូរ ប៉ុន្តែ checkpoint ចាស់នៅដដែល")
print("=" * 70)
m_bigger = Net(hidden=32)          # ផ្លាស់ប្ដូរ hidden ពី 16 ទៅ 32
try:
    m_bigger.load_state_dict(torch.load(CKPT, weights_only=False)["model_state_dict"])
except RuntimeError as e:
    print("  RuntimeError:")
    for line in str(e).splitlines()[:6]:
        print("   ", line[:74])
print("\n  ⚠ សំខាន់ណាស់: strict=False មិនជួយក្នុងករណីនេះទេ!")
print("     strict=False អត់ឱនឱ្យតែ *key ដែលបាត់ ឬលើស* — មិនមែន shape ខុសគ្នាទេ។")
try:
    m_bigger.load_state_dict(
        torch.load(CKPT, weights_only=False)["model_state_dict"], strict=False
    )
except RuntimeError as e:
    print(f"     strict=False នៅតែ -> RuntimeError: {str(e).splitlines()[0][:56]}")

print("\n  ដំណោះស្រាយ ១: រក្សាទុក hyperparameters ក្នុង checkpoint ហើយផ្ទៀងផ្ទាត់មុនផ្ទុក")
print("     torch.save({'hparams': {'hidden': 16}, 'model_state_dict': ...}, path)")
print("     ck = torch.load(path); assert ck['hparams']['hidden'] == 16")

print("\n  ដំណោះស្រាយ ២: តម្រង weight ដែល shape ត្រូវគ្នាដោយដៃ (transfer learning)")
sd = torch.load(CKPT, weights_only=False)["model_state_dict"]
target_sd = m_bigger.state_dict()
compatible = {
    k: v for k, v in sd.items()
    if k in target_sd and v.shape == target_sd[k].shape
}
skipped = [k for k in sd if k not in compatible]
result = m_bigger.load_state_dict(compatible, strict=False)
print(f"     ផ្ទុកបាន {len(compatible)} tensors, រំលង {len(skipped)} (shape ខុស): {skipped}")
print(f"     missing_keys = {result.missing_keys}")
print(f"     unexpected_keys = {result.unexpected_keys}")

print("\n  ដំណោះស្រាយ ៣ (ករណីដែល strict=False ពិតជាដំណើរការ): head ថ្មីបន្ថែម")


class NetWithExtraHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)      # ដូច checkpoint
        self.fc2 = nn.Linear(16, 3)      # ដូច checkpoint
        self.fc3 = nn.Linear(16, 5)      # head ថ្មី គ្មានក្នុង checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)


m_extra = NetWithExtraHead()
res = m_extra.load_state_dict(sd, strict=False)
print(f"     missing_keys   = {res.missing_keys}   (fc3 នៅ random — ត្រូវបណ្ដុះបណ្ដាល)")
print(f"     unexpected_keys = {res.unexpected_keys}")
assert "fc3.weight" in res.missing_keys
# ផ្ទៀងផ្ទាត់ថា fc1/fc2 ពិតជាត្រូវបានផ្ទុក
assert torch.equal(m_extra.fc1.weight, model.fc1.weight)
print("     fc1.weight ដូច model ដើម:", torch.equal(m_extra.fc1.weight, model.fc1.weight), "✓")

print()
print("=" * 70)
print("កំហុសទី ៥: ឯកសារ checkpoint ខូច ឬមិនពេញលេញ (ការសរសេរត្រូវបានកាត់ផ្ដាច់)")
print("=" * 70)
BROKEN = WORKDIR / "broken.pt"
with open(BROKEN, "wb") as fh:
    fh.write(b"\x80\x02]q\x00truncated")     # បៃធន្តិការមិនពេញលេញ
try:
    torch.load(BROKEN, weights_only=False)
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:70]}")
print("\n  ដំណោះស្រាយ: សរសេរទៅឯកសារបណ្ដោះអាសន្ន ហើយប្ដូរឈ្មោះ (atomic):")
print("     torch.save(payload, path.with_suffix('.tmp'))")
print("     os.replace(path.with_suffix('.tmp'), path)")

# បង្ហាញការសរសេរ atomic
SAFE_TMP = WORKDIR / "safe.pt.tmp"
SAFE = WORKDIR / "safe.pt"
torch.save({"epoch": 1}, SAFE_TMP)
os.replace(SAFE_TMP, SAFE)
print(f"  atomic write -> {SAFE.name} ផ្ទុកបាន:", torch.load(SAFE, weights_only=False))

print()
print("=" * 70)
print("កំហុសទី ៦: checkpoint ធំពេក — វិធីកាត់បន្ថយ")
print("=" * 70)
size_full = CKPT.stat().st_size / 1024
torch.save(model.state_dict(), WORKDIR / "weights_only.pt")
size_weights = (WORKDIR / "weights_only.pt").stat().st_size / 1024
print(f"  checkpoint ពេញលេញ : {size_full:.1f} KB")
print(f"  weights តែប៉ុណ្ណោះ : {size_weights:.1f} KB")
print("  យុទ្ធសាស្ត្របន្ថែម:")
print("   - រក្សាទុក optimizer state តែក្នុង 'last.pt' មិនមែន 'best.pt'")
print("   - ប្រើ dtype តូចជាង (fp16/bf16) សម្រាប់ weight ដែលរក្សាទុក")
print("   - លុប checkpoint ចាស់ (រក្សាទុក N ចុងក្រោយ)")

half = {k: v.half() for k, v in model.state_dict().items()}
torch.save(half, WORKDIR / "weights_fp16.pt")
size_half = (WORKDIR / "weights_fp16.pt").stat().st_size / 1024
print(f"   - weights ជា fp16 : {size_half:.1f} KB  (ប្រហែលពាក់កណ្ដាល)")
assert size_half < size_weights

for f in WORKDIR.glob("*.pt"):
    f.unlink()
if BROKEN.exists():
    BROKEN.unlink()
print("\nបានលុបឯកសារបណ្ដោះអាសន្ន")
print("OK — កំហុស checkpoint ទាំងអស់ត្រូវបានបង្ហាញ និងពន្យល់")
