# EXPECTED FAILURE: RuntimeError
# EXPECTED OUTPUT: mat1 and mat2 shapes cannot be multiplied
"""ឧទាហរណ៍នៃកំហុសពិតដែលមិនត្រូវបានចាប់យក — សម្រាប់បង្ហាញ traceback ពេញលេញ។

ដំណើរការលើ៖ CPU (មិនត្រូវការ GPU)

ឯកសារនេះត្រូវបានសម្គាល់ដោយ `# EXPECTED FAILURE:` ដូច្នេះ test runner
ផ្ទៀងផ្ទាត់ថាវាបរាជ័យដោយសារមូលហេតុត្រឹមត្រូវ។ នេះគឺជាអ្វីដែលអ្នកនឹងឃើញ
នៅក្នុង terminal របស់អ្នកនៅពេល Linear layer ទទួល feature ចំនួនខុស។
"""

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),     # <- រំពឹងថាធាតុចូលមាន 10 features
    nn.ReLU(),
    nn.Linear(32, 3),
)

batch = torch.randn(8, 12)      # ប៉ុន្តែយើងផ្ដល់ 12 features
print("input shape :", tuple(batch.shape))
print("Linear រំពឹង: in_features = 10")
print("កំពុងរត់ forward pass...\n")

out = model(batch)              # <- RuntimeError នៅទីនេះ
print("output shape:", tuple(out.shape))   # បន្ទាត់នេះមិនដែលដំណើរការទេ
