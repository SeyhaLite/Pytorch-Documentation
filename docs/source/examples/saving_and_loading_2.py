"""ទម្រង់រក្សាទុកផ្សេងទៀត៖ TorchScript, torch.export, safetensors, ONNX។

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
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


model = Net().eval()
example = torch.randn(2, 8)
with torch.no_grad():
    reference = model(example)

print("=" * 66)
print(f"{'ទម្រង់':<22}{'អាចប្រើបាន':<14}{'ទំហំ':>10}")
print("=" * 66)

# ---------------------------------------------------------------------------
# ១. TorchScript (torch.jit) — script ឬ trace
# ---------------------------------------------------------------------------
ts_path = WORKDIR / "model_scripted.pt"
try:
    scripted = torch.jit.script(model)          # វិភាគ Python code
    scripted.save(str(ts_path))
    reloaded = torch.jit.load(str(ts_path))
    with torch.no_grad():
        out = reloaded(example)
    ok = torch.allclose(out, reference, atol=1e-6)
    print(f"{'TorchScript (script)':<22}{'បាទ/ចាស':<14}{ts_path.stat().st_size / 1024:>8.1f} KB")
    assert ok

    traced = torch.jit.trace(model, example)    # កត់ត្រាប្រតិបត្តិការជាក់ស្ដែង
    trace_path = WORKDIR / "model_traced.pt"
    traced.save(str(trace_path))
    print(f"{'TorchScript (trace)':<22}{'បាទ/ចាស':<14}{trace_path.stat().st_size / 1024:>8.1f} KB")
    with torch.no_grad():
        assert torch.allclose(traced(example), reference, atol=1e-6)
    trace_path.unlink()
except Exception as e:
    print(f"{'TorchScript':<22}{'ទេ':<14}  ({type(e).__name__})")
finally:
    if ts_path.exists():
        ts_path.unlink()

print("\nចំណាំ TorchScript:")
print("  torch.jit.script : វិភាគ code ពេញលេញ — គាំទ្រ if/for ដែលអាស្រ័យលើ tensor")
print("  torch.jit.trace  : កត់ត្រាតែផ្លូវដែលបានរត់ — លឿនជាង ប៉ុន្តែបាត់ control flow")

# ---------------------------------------------------------------------------
# ២. torch.export (PyTorch 2.x) — ទម្រង់ថ្មីសម្រាប់ deployment
# ---------------------------------------------------------------------------
print("-" * 66)
export_path = WORKDIR / "model_exported.pt2"
try:
    ep = torch.export.export(model, (example,))
    torch.export.save(ep, str(export_path))
    ep2 = torch.export.load(str(export_path))
    with torch.no_grad():
        out = ep2.module()(example)
    print(f"{'torch.export':<22}{'បាទ/ចាស':<14}{export_path.stat().st_size / 1024:>8.1f} KB")
    print("  graph signature:")
    print("   ", ep.graph_signature.input_specs[:2])
    assert torch.allclose(out, reference, atol=1e-6)
except Exception as e:
    print(f"{'torch.export':<22}{'ទេ':<14}  {type(e).__name__}: {str(e)[:44]}")
finally:
    if export_path.exists():
        export_path.unlink()

# ---------------------------------------------------------------------------
# ៣. safetensors — ទម្រង់សុវត្ថិភាព សាមញ្ញ គ្មាន code execution
# ---------------------------------------------------------------------------
print("-" * 66)
st_path = WORKDIR / "model.safetensors"
try:
    from safetensors.torch import load_file, save_file

    save_file(model.state_dict(), str(st_path))
    sd = load_file(str(st_path))
    m2 = Net()
    m2.load_state_dict(sd)
    m2.eval()
    with torch.no_grad():
        assert torch.allclose(m2(example), reference, atol=1e-6)
    print(f"{'safetensors':<22}{'បាទ/ចាស':<14}{st_path.stat().st_size / 1024:>8.1f} KB")
    print("  (ត្រូវដំឡើងដោយឡែក: pip install safetensors)")
except ImportError:
    print(f"{'safetensors':<22}{'មិនបានដំឡើង':<14}")
    print("  safetensors គឺជាជម្រើសសុវត្ថិភាពជាង .pth ព្រោះវាមិនអាចរត់ code បានទេ")
    print("  ដំឡើង: pip install safetensors")
except Exception as e:
    print(f"{'safetensors':<22}{'កំហុស':<14} {type(e).__name__}: {str(e)[:40]}")
finally:
    if st_path.exists():
        st_path.unlink()

# ---------------------------------------------------------------------------
# ៤. ONNX — សម្រាប់ប្រព័ន្ធ inference ខាងក្រៅ
# ---------------------------------------------------------------------------
print("-" * 66)
onnx_path = WORKDIR / "model.onnx"
try:
    torch.onnx.export(
        model,
        (example,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"{'ONNX':<22}{'បាទ/ចាស':<14}{onnx_path.stat().st_size / 1024:>8.1f} KB")

    # ពិនិត្យថាឯកសារ ONNX ត្រឹមត្រូវ (បើ onnx បានដំឡើង)
    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(onnx_path)))
        print("  onnx.checker: ត្រឹមត្រូវ")
    except ImportError:
        print("  (onnx package មិនបានដំឡើង — រំលងការផ្ទៀងផ្ទាត់)")
except Exception as e:
    print(f"{'ONNX':<22}{'ទេ':<14} {type(e).__name__}: {str(e)[:44]}")
finally:
    if onnx_path.exists():
        onnx_path.unlink()

# ---------------------------------------------------------------------------
# ៥. តារាងប្រៀបធៀប
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print(f"{'ទម្រង់':<18}{'ប្រើពេល':<26}{'ត្រូវការ PyTorch ពេលផ្ទុក?'}")
print("=" * 66)
print(f"{'.pth state_dict':<18}{'បន្តការងារ/fine-tune':<24}{'បាទ/ចាស'}")
print(f"{'TorchScript':<18}{'C++ inference':<26}{'ទេ'}")
print(f"{'torch.export':<18}{'compiler / edge':<26}{'ទេ (runtime)'}")
print(f"{'safetensors':<18}{'ចែករំលែក weight':<24}{'បាទ/ចាស (load)'}")
print(f"{'ONNX':<18}{'b្រព័ន្ធផ្សេង':<25}{'ទេ'}")

print("\nOK — ទម្រង់រក្សាទុកទាំងអស់ត្រូវបានសាកល្បង")
