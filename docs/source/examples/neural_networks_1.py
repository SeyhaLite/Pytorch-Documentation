"""ម៉ូដែល nn.Module ដំបូង៖ MLP សម្រាប់ classification។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# ១. វិធីសរសេរទី ១: nn.Sequential (សាមញ្ញ គ្មាន branching)
# ---------------------------------------------------------------------------
mlp_seq = nn.Sequential(
    nn.Linear(784, 128),   # ធាតុចូល 784 (រូបភាព 28x28 ត្រូវបាន flatten)
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),     # ចេញ 10 ថ្នាក់
)

print("nn.Sequential:")
print(mlp_seq)

# ---------------------------------------------------------------------------
# ២. វិធីសរសេរទី ២: class ផ្ទាល់ខ្លួន (អាចមាន branching, skip connections)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Multi-Layer Perceptron សាមញ្ញសម្រាប់ classification។"""

    def __init__(self, in_features: int, hidden: int, num_classes: int) -> None:
        super().__init__()                       # ចាំបាច់ជានិច្ច!
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) -> logits: (batch, num_classes)

        ចំណាំ៖ ហៅ model(x) មិនមែន model.forward(x) ទេ ព្រោះ nn.Module
        រត់ hooks និង logic ផ្សេងៗនៅក្នុង __call__។
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.head(x)


mlp_class = MLP(in_features=784, hidden=128, num_classes=10)
print("\nclass ផ្ទាល់ខ្លួន:")
print(mlp_class)

# ---------------------------------------------------------------------------
# ៣. ពិនិត្យ parameters និង forward pass
# ---------------------------------------------------------------------------
print("-" * 62)
print(f"{'name':<12} {'shape':<16} {'numel':>8}")
total = 0
for name, p in mlp_class.named_parameters():
    print(f"{name:<12} {str(tuple(p.shape)):<16} {p.numel():>8}")
    total += p.numel()
print(f"{'សរុប':<12} {'':<16} {total:>8}")

# ពិនិត្យដោយដៃ: (784*128+128) + (128*64+64) + (64*10+10) = 100480+8256+650
expected = (784 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
print("គណនាដោយដៃ:", expected)
assert total == expected, f"{total} != {expected}"

# forward pass
batch = torch.randn(32, 784)
logits = mlp_class(batch)
print("\ninput  :", tuple(batch.shape), "(batch=32, features=784)")
print("logits :", tuple(logits.shape), "(batch=32, classes=10)")
assert logits.shape == (32, 10)

# logits គឺជាលេខឆៅ — មិនទាន់ softmax
preds = logits.argmax(dim=1)
probs = torch.softmax(logits, dim=1)
print("preds  :", tuple(preds.shape), preds[:8].tolist())
print("probs ផលបូកតាមជួរ (គួរ = 1.0):", probs.sum(dim=1)[:4].tolist())
assert preds.shape == (32,)
assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6)

# ---------------------------------------------------------------------------
# ៤. named_modules() — ដើរលើជាន់នីមួយៗ
# ---------------------------------------------------------------------------
print("-" * 62)
print("modules ទាំងអស់:")
for name, mod in mlp_class.named_modules():
    print(f"  {name or '(root)':<12} {type(mod).__name__}")

# ---------------------------------------------------------------------------
# ៥. train() និង eval() — សំខាន់ណាស់សម្រាប់ Dropout/BatchNorm
# ---------------------------------------------------------------------------
print("-" * 62)
x_test = torch.randn(4, 784)

mlp_class.train()
out_train_1 = mlp_class(x_test)
out_train_2 = mlp_class(x_test)
print("train() mode: លទ្ធផលពីរដងដូចគ្នា?",
      torch.equal(out_train_1, out_train_2), "(ទេ — Dropout ច្របល់រាល់ដង)")

mlp_class.eval()
out_eval_1 = mlp_class(x_test)
out_eval_2 = mlp_class(x_test)
print("eval()  mode: លទ្ធផលពីរដងដូចគ្នា?",
      torch.equal(out_eval_1, out_eval_2), "(បាទ/ចាស — Dropout បិទ)")

assert not torch.equal(out_train_1, out_train_2)
assert torch.equal(out_eval_1, out_eval_2)
print("model.training flag ឥឡូវ:", mlp_class.training)

print("-" * 62)
print("OK — nn.Module ដំណើរការត្រឹមត្រូវ")
