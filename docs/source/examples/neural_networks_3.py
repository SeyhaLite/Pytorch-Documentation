"""ជាន់ (layers) សំខាន់ៗរបស់ torch.nn និងអ្វីដែលពួកវាធ្វើ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)
"""

import torch
import torch.nn as nn

torch.manual_seed(0)
print("=" * 66)
print(f"{'Layer':<22} {'input shape':<18} {'output shape':<18}")
print("=" * 66)

cases = []

# Linear / fully connected
x = torch.randn(8, 16)
cases.append(("nn.Linear(16, 4)", x, nn.Linear(16, 4)))

# Convolution
x_img = torch.randn(8, 3, 32, 32)
cases.append(("nn.Conv2d(3,8,3,pad=1)", x_img, nn.Conv2d(3, 8, 3, padding=1)))

# Pooling
cases.append(("nn.MaxPool2d(2)", x_img, nn.MaxPool2d(2)))
cases.append(("nn.AdaptiveAvgPool2d(1)", x_img, nn.AdaptiveAvgPool2d(1)))

# Normalization
cases.append(("nn.BatchNorm1d(16)", x, nn.BatchNorm1d(16)))
cases.append(("nn.LayerNorm(16)", x, nn.LayerNorm(16)))

# Activation
cases.append(("nn.ReLU()", x, nn.ReLU()))
cases.append(("nn.GELU()", x, nn.GELU()))
cases.append(("nn.Sigmoid()", x, nn.Sigmoid()))
cases.append(("nn.Softmax(dim=-1)", x, nn.Softmax(dim=-1)))
cases.append(("nn.LogSoftmax(dim=-1)", x, nn.LogSoftmax(dim=-1)))

# Regularization
cases.append(("nn.Dropout(0.5)", x, nn.Dropout(0.5)))

# Reshaping
cases.append(("nn.Flatten()", x_img, nn.Flatten()))
cases.append(("nn.Unflatten(1,(4,4))", x, nn.Unflatten(1, (4, 4))))

# Embedding
ids = torch.randint(0, 100, (8, 10))     # (batch, seq_len) នៃ token id
cases.append(("nn.Embedding(100, 8)", ids, nn.Embedding(100, 8)))

# Recurrent / Attention
cases.append(("nn.LSTM(16,8,batch_first)", x.unsqueeze(1), nn.LSTM(16, 8, batch_first=True)))
cases.append(("nn.MultiheadAttention", x.unsqueeze(0), None))

for name, inp, layer in cases:
    if layer is None:
        mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        with torch.no_grad():
            out, _ = mha(inp, inp, inp)
        print(f"{name:<22} {str(tuple(inp.shape)):<18} {str(tuple(out.shape)):<18}")
        assert out.shape == inp.shape
        continue

    with torch.no_grad():
        out = layer(inp)
    if isinstance(out, tuple):           # LSTM ត្រឡប់ (output, (h, c))
        out_shape = tuple(out[0].shape)
        extra = f" + h{tuple(out[1][0].shape)}"
    else:
        out_shape = tuple(out.shape)
        extra = ""
    print(f"{name:<22} {str(tuple(inp.shape)):<18} {str(out_shape) + extra:<18}")

# ---------------------------------------------------------------------------
# BatchNorm: train vs eval ខុសគ្នាច្បាស់
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("BatchNorm1d ក្នុង train() vs eval()")
print("=" * 66)
bn = nn.BatchNorm1d(4)
sample = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])

bn.train()
out_train = bn(sample)
print("train() output:\n", out_train)
print("running_mean បន្ទាប់ :", bn.running_mean.tolist())
print("running_var  បន្ទាប់ :", [round(v, 3) for v in bn.running_var.tolist()])

bn.eval()
out_eval = bn(sample)
print("\neval() output (ប្រើ running statistics):\n", out_eval)
assert not torch.equal(out_train, out_eval)

# ---------------------------------------------------------------------------
# Dropout: train() សម្រាប់បិទធាតុ eval() ធ្វើឱ្យដូចគ្នា
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
drop = nn.Dropout(p=0.5)
v = torch.ones(1, 8)

drop.train()
t1, t2 = drop(v), drop(v)
print("train() លទ្ធផល ១:", t1.flatten().tolist())
print("train() លទ្ធផល ២:", t2.flatten().tolist())
print("(ធាតុខ្លះ = 0 ហើយអ្វីដែលនៅសល់ត្រូវគុណ 1/(1-p) = 2.0)")

drop.eval()
e1 = drop(v)
print("eval()  លទ្ធផល  :", e1.flatten().tolist(), "<- មិនប្ដូរ")
assert torch.equal(e1, v)

print("=" * 66)
print("OK — layers ទាំងអស់ដំណើរការត្រឹមត្រូវ")
