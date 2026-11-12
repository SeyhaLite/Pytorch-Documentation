# ឯកសារ PyTorch ជាភាសាខ្មែរ

សៀវភៅណែនាំ **PyTorch ជាភាសាខ្មែរ** សម្រាប់អ្នកដែលចង់រៀនពីមូលដ្ឋាន Tensor រហូតដល់ការបណ្ដុះបណ្ដាល និងប្រើប្រាស់ Deep Learning model។

<https://pytorch-documentation.readthedocs.io/km/latest/>

> **ចំណាំ៖** នេះជាឯកសារសិក្សាដាច់ដោយឡែកដែលរៀបចំជាភាសាខ្មែរ មិនមែនជាការបកប្រែឯកសារ PyTorch ផ្លូវការទាំងស្រុងទេ។ ខ្លឹមសារយោងតាម [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)។ សូមមើល [LICENSE](LICENSE) សម្រាប់អាជ្ញាបណ្ណ និងការទទួលស្គាល់ប្រភព។

## តំណសំខាន់ៗ

- **[អានមេរៀនតាមគេហទំព័រ](https://pytorch-documentation.readthedocs.io/km/latest/)**
- [របាយការណ៍ Build និង Test](docs/BUILD_REPORT.md)
- [របៀបចូលរួមកែលម្អឯកសារ](CONTRIBUTING.md)
- [Prompt ពិនិត្យគុណភាព និងផ្ទៀងផ្ទាត់ជាមួយ PyTorch ផ្លូវការ](prompts/PYTORCH_DOCS_QUALITY_AND_PARITY_REVIEW_KM.md)
- [អាជ្ញាបណ្ណ](LICENSE)

## ស្ថានភាព Version របស់ PyTorch

គួរបែងចែករវាង **កំណែ PyTorch ផ្លូវការថ្មីបំផុត** និង **កំណែដែលគម្រោងនេះបាន Test ពិត**៖

- PyTorch ផ្លូវការបានចេញ **2.15.0** នៅថ្ងៃទី 28 តុលា 2026 ([Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0))។
- Build Report របស់ Repository បច្ចុប្បន្នកត់ត្រា code examples ដែលបាន Test លើ **2.14.1+cpu** ([មើលលទ្ធផល](docs/BUILD_REPORT.md))។
- ដូច្នេះ កុំសន្មតថាគ្រប់មេរៀន និង code example បាន Test លើ 2.15.0 រួចហើយ។ ត្រូវរង់ចាំការផ្ទៀងផ្ទាត់ version/API និង test ជាក់ស្ដែង។

សម្រាប់ការត្រួតពិនិត្យភាពត្រឹមត្រូវ និង Version compatibility សូមប្រើ [Prompt PyTorch Official Parity Audit](prompts/PYTORCH_DOCS_QUALITY_AND_PARITY_REVIEW_KM.md)។

## គម្រោងនេះមានអ្វីខ្លះ?

- មេរៀនជាភាសាខ្មែរ ចាប់ពីការដំឡើង រហូតដល់ការបណ្ដុះបណ្ដាល និងវាយតម្លៃ Model។
- ឧទាហរណ៍ Python នៅក្នុងឯកសារ `.py` ដាច់ដោយឡែក ហើយបញ្ចូលទៅក្នុងមេរៀនដោយ `literalinclude`។ ដូច្នេះ កូដដែលអ្នកអាន គឺជាកូដដែល Test Runner យកទៅសាកល្បង។
- ការណែនាំអំពី CPU, GPU, Autograd, Neural Networks, DataLoader, Optimizer និងការរក្សាទុក Model។
- គេហទំព័រគាំទ្រ Light Mode និង Dark Mode, ការស្វែងរក, ប៊ូតុងចម្លងកូដ និងទម្រង់សម្រាប់បោះពុម្ព។
- មានស្គ្រីបសម្រាប់ពិនិត្យកូដឧទាហរណ៍, Markdown fence, ពណ៌ Contrast, Khmer UI, Sitemap និងការសាងសង់ Sphinx។

## តើគួរចាប់ផ្ដើមរៀនពីណា?

| អ្នករៀន | លំដាប់ណែនាំ |
| --- | --- |
| អ្នកចាប់ផ្ដើម | Installation → Getting Started → Tensors → Shapes and Broadcasting → Devices |
| អ្នកចេះ Python និង NumPy រួច | Tensors → Autograd → Neural Networks |
| អ្នកកំពុងបង្កើត Model | Data and DataLoader → Training Loop → Evaluation and Inference |
| អ្នកជួបបញ្ហាក្នុងកូដ | Debugging → API Quick Reference |

## បញ្ជីមេរៀន

តំណខាងក្រោមនាំទៅកាន់ទំព័រមេរៀនដែលបានបង្ហោះលើ Read the Docs។

| មេរៀន | ខ្លឹមសារសំខាន់ |
| --- | --- |
| [Installation](https://pytorch-documentation.readthedocs.io/km/latest/installation.html) | ដំឡើង PyTorch និងរៀបចំ Python Environment |
| [Getting Started](https://pytorch-documentation.readthedocs.io/km/latest/getting-started.html) | ចាប់ផ្ដើមប្រើ Tensor និង PyTorch |
| [Tensors](https://pytorch-documentation.readthedocs.io/km/latest/tensors.html) | បង្កើត Tensor, Indexing និង Operations |
| [Shapes and Broadcasting](https://pytorch-documentation.readthedocs.io/km/latest/shapes-and-broadcasting.html) | Shape, Dimension, Stride និង Broadcasting |
| [Devices](https://pytorch-documentation.readthedocs.io/km/latest/devices.html) | ប្រើ CPU, CUDA GPU និង MPS |
| [Autograd](https://pytorch-documentation.readthedocs.io/km/latest/autograd.html) | Gradient និង Automatic Differentiation |
| [Neural Networks](https://pytorch-documentation.readthedocs.io/km/latest/neural-networks.html) | `nn.Module`, Layers, MLP និង CNN |
| [Loss and Optimizers](https://pytorch-documentation.readthedocs.io/km/latest/loss-and-optimizers.html) | Loss Functions, SGD, Adam និង Optimizers |
| [Data and DataLoader](https://pytorch-documentation.readthedocs.io/km/latest/data-and-dataloader.html) | Dataset, DataLoader និងការរៀបចំទិន្នន័យ |
| [Training Loop](https://pytorch-documentation.readthedocs.io/km/latest/training-loop.html) | វដ្ដបណ្ដុះបណ្ដាល Model |
| [Evaluation and Inference](https://pytorch-documentation.readthedocs.io/km/latest/evaluation-and-inference.html) | វាយតម្លៃ និងទាញលទ្ធផលពី Model |
| [Saving and Loading](https://pytorch-documentation.readthedocs.io/km/latest/saving-and-loading.html) | រក្សាទុក និងផ្ទុក Model |
| [Checkpoints](https://pytorch-documentation.readthedocs.io/km/latest/checkpoints.html) | រក្សាទុកចំណុចបន្តការបណ្ដុះបណ្ដាល |
| [Reproducibility](https://pytorch-documentation.readthedocs.io/km/latest/reproducibility.html) | Random Seeds និងលទ្ធផលដែលអាចធ្វើឡើងវិញ |
| [Performance](https://pytorch-documentation.readthedocs.io/km/latest/performance.html) | វិធីកែលម្អល្បឿន និងការប្រើប្រាស់ធនធាន |
| [Mixed Precision](https://pytorch-documentation.readthedocs.io/km/latest/mixed-precision.html) | ការគណនាដោយប្រើ Precision ច្រើនប្រភេទ |
| [Distributed Training](https://pytorch-documentation.readthedocs.io/km/latest/distributed-training.html) | Distributed Data Parallel និងការបណ្ដុះបណ្ដាលលើច្រើន GPU |
| [Debugging](https://pytorch-documentation.readthedocs.io/km/latest/debugging.html) | កំហុសទូទៅ និងវិធីស្វែងរកមូលហេតុ |
| [API Quick Reference](https://pytorch-documentation.readthedocs.io/km/latest/api-quick-reference.html) | តារាងយោង API រហ័ស |
| [Style Guide](https://pytorch-documentation.readthedocs.io/km/latest/style-guide.html) | ពណ៌ ពុម្ពអក្សរ និងធាតុ UI របស់ឯកសារ |
| [License](https://pytorch-documentation.readthedocs.io/km/latest/license.html) | អាជ្ញាបណ្ណ និងការទទួលស្គាល់ប្រភព |

## ស្ថានភាពគុណភាព និងការសាកល្បង

តាមលទ្ធផលដែលបានកត់ត្រានៅក្នុង [docs/BUILD_REPORT.md](docs/BUILD_REPORT.md)៖

| ចំណុចផ្ទៀងផ្ទាត់ | លទ្ធផល |
| --- | --- |
| ឯកសារ Markdown | ២២ ទំព័រ |
| ឧទាហរណ៍ Python | ៤០ ឯកសារ |
| ឧទាហរណ៍ដែល Test បានជោគជ័យ | ៣៨ |
| ឧទាហរណ៍ដែលត្រូវរំលង | ២ ព្រោះ Environment សាកល្បងមិនមាន CUDA GPU |
| ឧទាហរណ៍ដែលបរាជ័យ | ០ |
| Sphinx Build | ០ Warnings |
| Contrast | ១២០ គូពណ៌ ឆ្លងកម្រិត WCAG AA (យ៉ាងតិច ៤.៥:១) |

**ចំណាំអំពី Test៖** លទ្ធផលខាងលើមិនមានន័យថាកូដ GPU ទាំងអស់បានដំណើរការលើ GPU ពិតទេ។ ឧទាហរណ៍ពីរដែលត្រូវការ CUDA ត្រូវបានរំលងក្នុង Environment ដែលបានរាយការណ៍។ សម្រាប់ព័ត៌មានលម្អិត សូមអាន Build Report។

## បង្កើតគេហទំព័រនៅលើកុំព្យូទ័ររបស់អ្នក

### តម្រូវការ

- Python។ កំណែដែលបានសាកល្បងមាន Python 3.11 សម្រាប់ការសាងសង់ក្នុងម៉ាស៊ីន និង Python 3.12 សម្រាប់ Read the Docs។
- Dependencies របស់ Sphinx ដែលមាននៅក្នុង `docs/requirements.txt`។
- PyTorch និង NumPy ត្រូវការតែពេលចង់ដំណើរការកូដឧទាហរណ៍ប៉ុណ្ណោះ មិនចាំបាច់សម្រាប់ Build ឯកសារ HTML ទេ។

### ១. រៀបចំ Environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r docs/requirements.txt
```

### ២. Build ឯកសារ HTML

```bash
python -m sphinx -b html docs/source docs/_build/html -q -W --keep-going
```

ជម្រើស `-W` ធ្វើឱ្យ Sphinx ចាត់ទុក Warning ជាកំហុស ដើម្បីកុំឱ្យបញ្ហាក្នុងឯកសាររំលងដោយមិនដឹងខ្លួន។

### ៣. បើកមើលក្នុង Browser

```bash
python -m http.server 8000 --directory docs/_build/html
```

បន្ទាប់មកបើក <http://localhost:8000>។

## សាកល្បងកូដឧទាហរណ៍

ការសាកល្បងឧទាហរណ៍ត្រូវការដំឡើង PyTorch និង NumPy ដាច់ដោយឡែកពី Sphinx៖

```bash
# ដំឡើង PyTorch សម្រាប់ CPU
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy

# ដំណើរការឧទាហរណ៍ទាំងអស់
python tests/run_examples.py

# ពិនិត្យថាឧទាហរណ៍ទាំងអស់មាន reference ក្នុងមេរៀន
python tests/check_examples_referenced.py
```

បើចង់សាកល្បងឧទាហរណ៍ដែលត្រូវការ GPU សូមប្រើម៉ាស៊ីនដែលមាន CUDA និងដំឡើង PyTorch តាមជម្រើសដែលត្រូវនឹង GPU របស់អ្នក ពី [ទំព័រដំឡើង PyTorch ផ្លូវការ](https://pytorch.org/get-started/locally/)។

## ផ្ទៀងផ្ទាត់គម្រោងទាំងមូល

មុនពេលបញ្ជូនការកែប្រែ សូមដំណើរការ៖

```bash
bash tools/verify.sh
```

ស្គ្រីបនេះដំណើរការតាមលំដាប់៖

1. ពិនិត្យកូដឧទាហរណ៍ និងការភ្ជាប់ទៅមេរៀន។
2. ពិនិត្យ Markdown fences, Contrast, JavaScript និង Read the Docs YAML។
3. Build Sphinx ដោយបើកការពិនិត្យ Warning។
4. ពិនិត្យ Khmer locale, UI strings, Google Search Console និង Sitemap បន្ទាប់ពី Build។

ដើម្បីសាកល្បងដំណើរការ Build លើ Repository ដែល Clone ថ្មីពី GitHub សូមប្រើ៖

```bash
bash tools/verify_clean_clone.sh
```

ស្គ្រីបទាំងនេះត្រូវបានបង្កើតសម្រាប់ Linux/macOS ឬ Environment ដែលអាចដំណើរការ Bash បាន។ សូមមើល [Build Report](docs/BUILD_REPORT.md) សម្រាប់លទ្ធផល និងដែនកំណត់ដែលបានកត់ត្រា។

## រចនាសម្ព័ន្ធ Repository

```text
Pytorch-Documentation/
├── .readthedocs.yaml          # ការកំណត់ Build របស់ Read the Docs
├── CONTRIBUTING.md            # គោលការណ៍សម្រាប់អ្នកចូលរួម
├── LICENSE                    # អាជ្ញាបណ្ណ និង Attribution
├── README.md                  # ឯកសារណែនាំ Repository
├── docs/
│   ├── BUILD_REPORT.md        # របាយការណ៍ Build និង Test
│   ├── requirements.txt       # Dependencies សម្រាប់ Sphinx
│   └── source/
│       ├── index.md           # ទំព័រដើម និងតារាងមាតិកា
│       ├── *.md               # មេរៀន និងឯកសារយោង
│       ├── examples/          # កូដ Python ដែលអាច Test បាន
│       ├── _ext/              # Sphinx extensions របស់គម្រោង
│       ├── _static/            # CSS និង JavaScript
│       ├── _templates/         # HTML templates
│       ├── locale/             # Khmer translation catalog
│       └── extra/              # ឯកសារបន្ថែមសម្រាប់គេហទំព័រ
├── tests/                      # ស្គ្រីបផ្ទៀងផ្ទាត់ និង Test
└── tools/                      # ឧបករណ៍ Build និង Verify
```

## Read the Docs

គម្រោងនេះប្រើ [Read the Docs](https://readthedocs.org/) ដើម្បី Build និងបង្ហោះឯកសារ។ ការកំណត់ស្ថិតនៅក្នុង `.readthedocs.yaml`៖

- `pre_build` ពិនិត្យឯកសារយោងរបស់ឧទាហរណ៍ និង Markdown fences។
- Sphinx Build ប្រើ `fail_on_warning: true`។
- `post_build` ពិនិត្យ Khmer locale, UI strings, Google Search Console និង Sitemap។
- PyTorch មិនត្រូវបានដំឡើងក្នុង RTD Build ទេ ព្រោះវាចាំបាច់សម្រាប់ដំណើរការឧទាហរណ៍ មិនមែនសម្រាប់ Render ឯកសារទេ។

## ពិនិត្យភាពត្រឹមត្រូវជាមួយ PyTorch ផ្លូវការ

គម្រោងនេះមាន [Prompt សម្រាប់ AI Agent](prompts/PYTORCH_DOCS_QUALITY_AND_PARITY_REVIEW_KM.md) ដើម្បី Audit គ្រប់មេរៀន និងឧទាហរណ៍ ប្រៀបធៀប API/behavior ជាមួយ Official Documentation និង Release Notes រក version mismatch, deprecated API, code errors និងចំណុចខ្វះក្នុងការបង្រៀន។ Prompt នេះក៏កំណត់វិធីសាកល្បង និងទម្រង់របាយការណ៍ ដើម្បីមិនអះអាងលទ្ធផលដែលមិនបានផ្ទៀងផ្ទាត់។

## ចូលរួមកែលម្អ

សូមអាន [CONTRIBUTING.md](CONTRIBUTING.md) មុនពេលកែប្រែ។ ចំណុចសំខាន់ៗ៖

- រក្សាឈ្មោះ PyTorch API ដូចដើម ដូចជា `torch.Tensor`, `nn.Module`, `DataLoader` និង `backward()`។
- ដាក់កូដឧទាហរណ៍ក្នុង `docs/source/examples/` ហើយប្រើ `literalinclude` ដើម្បីបញ្ចូលវាទៅក្នុងមេរៀន។
- កុំអះអាងលទ្ធផល Benchmark ឬ GPU ដែលមិនបានសាកល្បងពិត។
- ដំណើរការ `bash tools/verify.sh` មុនបញ្ជូន Pull Request។

## អាជ្ញាបណ្ណ

- ខ្លឹមសារឯកសារ៖ **CC BY 4.0**។
- កូដឧទាហរណ៍៖ **BSD 3-Clause**។
- PyTorch និងរូបសញ្ញា PyTorch ជាស្លាកសញ្ញារបស់ PyTorch Foundation។

សូមមើល [LICENSE](LICENSE) និង [docs/source/license.md](docs/source/license.md) សម្រាប់លក្ខខណ្ឌ និង Attribution ពេញលេញ។
