"""រក្សាទុក និងផ្ទុក model ដោយប្រើ state_dict (វិធីណែនាំ)។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
ឯកសារត្រូវបានសរសេរទៅ directory បណ្ដោះអាសន្ន ហើយលុបចេញវិញ។
"""

import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

torch.manual_seed(0)
WORKDIR = Path(os.environ.get("KH_EXAMPLE_TMPDIR", tempfile.gettempdir()))
MODEL_PATH = WORKDIR / "model_weights.pth"


class Net(nn.Module):
    def __init__(self, in_f: int = 8, hidden: int = 16, out_f: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.fc2 = nn.Linear(hidden, out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# ១. state_dict គឺជាអ្វី?
# ---------------------------------------------------------------------------
model = Net()
print("model.state_dict() keys:")
for k, v in model.state_dict().items():
    print(f"  {k:<12} shape={str(tuple(v.shape)):<12} dtype={v.dtype}")

print("\nចំណាំ: state_dict មានតែ weight/bias — គ្មាន architecture ឬ code ទេ។")
print("ដូច្នេះអ្នកត្រូវមាន class និយមន័យដូចគ្នាដើម្បីផ្ទុកវាវិញ។")

# ---------------------------------------------------------------------------
# ២. រក្សាទុកតែ state_dict
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), MODEL_PATH)
size_kb = MODEL_PATH.stat().st_size / 1024
print(f"\nបានរក្សាទុក: {MODEL_PATH.name}  ({size_kb:.2f} KB)")

# ប្រៀបធៀបទំហំជាមួយការរក្សាទុក model ទាំងមូល
WHOLE_PATH = WORKDIR / "model_whole.pth"
torch.save(model, WHOLE_PATH)
print(f"រក្សាទុក model ទាំងមូល: {WHOLE_PATH.stat().st_size / 1024:.2f} KB")
print("(រក្សាទុក model ទាំងមូលតម្រូវឱ្យអ្នកអានមាន module path ដូចគ្នា — មិនណែនាំ)")

# ---------------------------------------------------------------------------
# ៣. ផ្ទុកវិញ (ត្រូវតែបង្កើត model ជាមុន)
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
loaded = Net()                              # architecture ដូចគ្នា
loaded.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
loaded.eval()

x = torch.randn(4, 8)
with torch.no_grad():
    out_orig = model.eval()(x)
    out_loaded = loaded(x)

print("output ដើម   :", [round(v, 5) for v in out_orig[0].tolist()])
print("output ផ្ទុក  :", [round(v, 5) for v in out_loaded[0].tolist()])
print("ដូចគ្នា      :", torch.equal(out_orig, out_loaded))
assert torch.equal(out_orig, out_loaded)

# ---------------------------------------------------------------------------
# ៤. weights_only=True — សុវត្ថិភាព
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
print("weights_only=True (default ចាប់ពី PyTorch 2.6):")
print("  អនុញ្ញាតតែ tensors និងប្រភេទសុវត្ថិភាពមូលដ្ឋាន។")
print("  ការពារ arbitrary code execution ពីឯកសារ .pth ដែលមិនទុកចិត្ត។")

try:
    sd = torch.load(MODEL_PATH, weights_only=True)
    print("  ផ្ទុក state_dict ដោយជោគជ័យ:", len(sd), "tensors")
except Exception as e:
    print("  បរាជ័យ:", type(e).__name__)

# weights_only=False អាចផ្ទុក object ណាក៏បាន — ប្រើតែជាមួយឯកសារដែលអ្នកទុកចិត្ត
print("  weights_only=False: ប្រើតែពេលចាំបាច់ និងឯកសារជឿទុកចិត្តបាន")

# ---------------------------------------------------------------------------
# ៥. ផ្ទុកទៅ device ផ្សេង (GPU -> CPU)
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device បច្ចុប្បន្ន: {device}")

# map_location ជួយពេល checkpoint ត្រូវបានរក្សាទុកលើ GPU ប៉ុន្តែផ្ទុកលើ CPU
sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
m_cpu = Net()
m_cpu.load_state_dict(sd)
print("map_location='cpu' -> tensor device:", next(m_cpu.parameters()).device)
assert next(m_cpu.parameters()).device.type == "cpu"

# ផ្ទុកដោយផ្ទាល់ទៅ device គោលដៅ
m_dev = Net().to(device)
m_dev.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
print(f"map_location='{device}' -> tensor device:", next(m_dev.parameters()).device)

# ---------------------------------------------------------------------------
# ៦. strict=False — ផ្ទុក partial (transfer learning)
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
print("strict=False សម្រាប់ transfer learning:")


class NetExtraHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 3)
        self.fc3 = nn.Linear(16, 5)          # head ថ្មី


bigger = NetExtraHead()
result = bigger.load_state_dict(
    torch.load(MODEL_PATH, weights_only=True), strict=False
)
print("  missing_keys  :", result.missing_keys, "(head ថ្មី — នៅ random)")
print("  unexpected_keys:", result.unexpected_keys)
assert "fc3.weight" in result.missing_keys and "fc3.bias" in result.missing_keys
assert result.unexpected_keys == []

# strict=True (default) នឹងបរាជ័យ
try:
    bigger.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
except RuntimeError as e:
    print("\n  strict=True -> RuntimeError:")
    for line in str(e).splitlines()[:4]:
        print("   ", line[:76])

# ---------------------------------------------------------------------------
# ៧. រក្សាទុក tensors ធម្មតា (មិនមែន model)
# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
TENSOR_PATH = WORKDIR / "tensors.pt"
# កំហុសធម្មតា៖ ព្យាយាមដាក់ string ក្នុង tensor
try:
    torch.tensor(["cat", "dog", "bird"])
except ValueError as e:
    print("⚠ torch.tensor([...strings...]) -> ValueError:", str(e)[:40])
    print("  tensor ផ្ទុកបានតែលេខទេ។ សម្រាប់អក្សរ ប្រើ Python list ឬ map ទៅ index។")

payload = {
    "embeddings": torch.randn(10, 4),
    "label_names": ["cat", "dog", "bird"],   # Python list មិនមែន tensor
    "label_to_id": {"cat": 0, "dog": 1, "bird": 2},
    "config": {"dim": 4, "n": 10},
    "epoch": 7,
    "best_val_loss": 0.4231,
}
torch.save(payload, TENSOR_PATH)

# ឯកសារមាន dict/str ដូច្នេះ weights_only=True នឹងបរាជ័យ —
# ប្រើ weights_only=False តែជាមួយឯកសារដែលអ្នកទុកចិត្តបាន។
restored = torch.load(TENSOR_PATH, weights_only=False)
print("\nបានរក្សាទុក dict ជាមួយ keys:", list(restored.keys()))
print("  embeddings shape:", tuple(restored["embeddings"].shape))
print("  label_names     :", restored["label_names"])
print("  label_to_id     :", restored["label_to_id"])
print("  config          :", restored["config"])
print("  epoch           :", restored["epoch"])
assert torch.equal(restored["embeddings"], payload["embeddings"])
assert restored["epoch"] == 7
assert restored["label_names"] == ["cat", "dog", "bird"]

# បង្ហាញថា weights_only=True ពិតជាបដិសេធសំណុំបែបបទនេះ
try:
    torch.load(TENSOR_PATH, weights_only=True)
    print("  weights_only=True: ផ្ទុកបាន (PyTorch អនុញ្ញាតប្រភេទទាំងនេះ)")
except Exception as e:
    print("  weights_only=True ->", type(e).__name__, "(ត្រូវការ weights_only=False)")

# សម្អាត
for p in (MODEL_PATH, WHOLE_PATH, TENSOR_PATH):
    if p.exists():
        p.unlink()
print("\nបានលុបឯកសារបណ្ដោះអាសន្នទាំងអស់")
print("OK — saving/loading state_dict ដំណើរការត្រឹមត្រូវ")
