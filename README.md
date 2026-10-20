# ឯកសារ PyTorch ជាភាសាខ្មែរ (PyTorch Documentation in Khmer)

សៀវភៅណែនាំ PyTorch ជាភាសាខ្មែរ សម្រាប់អ្នកចាប់ផ្ដើមដំបូង រហូតដល់អ្នកសិក្សា Deep Learning កម្រិតមធ្យម។
ឯកសារនេះសាងសង់ដោយ [Sphinx](https://www.sphinx-doc.org/) + [sphinx_rtd_theme](https://sphinx-rtd-theme.readthedocs.io/)
ហើយត្រូវបានបង្ហោះលើ [Read the Docs](https://app.readthedocs.org/)។

> **ចំណាំសំខាន់** — នេះ **មិនមែន**ជាការបកប្រែឯកសារ PyTorch ផ្លូវការទាំងស្រុងទេ។
> វាគឺជាឯកសារសិក្សាដាច់ដោយឡែក (independent Khmer study guide) ដែលយោងតាម
> PyTorch ផ្លូវការ និង API ពិតប្រាកដ។ ខ្លឹមសារបកប្រែយោងតាម
> [PyTorch Documentation](https://docs.pytorch.org/docs/stable/) ដែលមានអាជ្ញាបណ្ណ
> [CC BY 4.0](https://github.com/pytorch/pytorch/blob/main/docs/source_licenses/LICENSE)។
> មើល [LICENSE](LICENSE) និង [docs/source/license.md](docs/source/license.md) សម្រាប់ Attribution ពេញលេញ។

## ស្ថានភាពមាតិកា (Content status)

| ទំព័រ | ប្រធានបទ | ស្ថានភាព | កូដឧទាហរណ៍ |
| --- | --- | --- | --- |
| `index` | ទំព័រដើម / មាតិកា | ✅ រួចរាល់ | — |
| `installation` | ដំឡើង PyTorch និង Environment | ✅ | បានសាកល្បង |
| `getting-started` | ចាប់ផ្ដើមដំបូង | ✅ | បានសាកល្បង |
| `tensors` | Tensor Creation និង Operations | ✅ | បានសាកល្បង |
| `shapes-and-broadcasting` | Shapes, Dimensions និង Broadcasting | ✅ | បានសាកល្បង |
| `devices` | CPU និង GPU | ✅ | បានសាកល្បងលើ CPU (ផ្នែក GPU គ្រាន់តែបង្ហាញកូដ) |
| `autograd` | Autograd / Automatic Differentiation | ✅ | បានសាកល្បង |
| `neural-networks` | Neural Networks និង `torch.nn` | ✅ | បានសាកល្បង |
| `loss-and-optimizers` | Loss Functions និង Optimizers | ✅ | បានសាកល្បង |
| `data-and-dataloader` | Dataset និង DataLoader | ✅ | បានសាកល្បង |
| `training-loop` | Training Loop | ✅ | បានសាកល្បង |
| `evaluation-and-inference` | Model Evaluation និង Inference | ✅ | បានសាកល្បង |
| `saving-and-loading` | រក្សាទុក និងផ្ទុក Model | ✅ | បានសាកល្បង |
| `checkpoints` | Checkpoints | ✅ | បានសាកល្បង |
| `reproducibility` | Random Seeds និង Reproducibility | ✅ | បានសាកល្បង |
| `performance` | Performance Optimization | ✅ | បានសាកល្បង |
| `mixed-precision` | Mixed Precision | ✅ | សាកល្បងបានតែផ្នែក CPU (គ្មាន GPU) |
| `distributed-training` | Distributed Training | ⚠️ ទ្រឹស្ដី (មិនបានសាកល្បង multi-node) | មិនបាន Run |
| `debugging` | Debugging និង Common Errors | ✅ | បានសាកល្បង |
| `style-guide` | មគ្គុទ្ទេសក៍រចនា (palette, typography, ធាតុទាំងអស់) | ✅ | បានសាកល្បង |
| `api-quick-reference` | តារាងយោង API រហ័ស | ✅ | — |

សរុប៖ ២២ ទំព័រ · ៤០ ឧទាហរណ៍ (៣៨ បានរត់ pass, ០ fail, ២ skip ព្រោះត្រូវការ GPU)។
ការត្រួតពិនិត្យស្វ័យប្រវត្តិ ១០ ឆ្លងកាត់ទាំងអស់ — មើល `bash tools/verify.sh`។
មើល [docs/BUILD_REPORT.md](docs/BUILD_REPORT.md) សម្រាប់របាយការណ៍ Build/Test ពេញលេញ។

## រចនាសម្ព័ន្ធ (Project structure)

```text
.
├── .readthedocs.yaml              # Read the Docs build configuration
├── .gitignore
├── LICENSE                        # អាជ្ញាបណ្ណគម្រោង + Attribution
├── CONTRIBUTING.md                # របៀបចូលរួម (រួមទាំងច្បាប់ MyST fence)
├── docs/
│   ├── requirements.txt           # Sphinx build dependencies
│   ├── BUILD_REPORT.md            # របាយការណ៍ Build/Test ពេញលេញ
│   └── source/
│       ├── conf.py                # Sphinx configuration
│       ├── index.md               # ទំព័រដើម + មាតិកា
│       ├── *.md                   # មេរៀន ២២ (ឈ្មោះឯកសារអង់គ្លេស ចំណងជើងខ្មែរ)
│       ├── examples/              # កូដឧទាហរណ៍ ៤០ — ដំណើរការពិតដោយ test runner
│       ├── _ext/khmer_assets.py   # ចុះឈ្មោះ copy button + print CSS
│       ├── _static/css/custom.css # Palette dark + light (CSS custom properties)
│       ├── _static/css/print.css  # Print stylesheet
│       ├── _static/js/copybutton.js
│       └── _templates/searchbox.html  # Dark/Light control + inline script
├── tests/
│   ├── run_examples.py            # សាកល្បងឧទាហរណ៍ទាំង ៤០ (subprocess ដាច់ដោយឡែក)
│   └── check_examples_referenced.py   # ការពារ orphan/stale references
└── tools/
    ├── check_markdown.py          # Markdown fence linter (report-only)
    ├── build_locale.py            # បង្កើត Khmer UI catalog (.mo + .js)
    ├── check_contrast.py          # WCAG AA verifier (១០៤ គូពណ៌)
    ├── make_preview.py            # បង្កើត offline preview HTML តែមួយ
    ├── build_locale.py            # បង្កើត Khmer UI catalog (.mo + .js) ដោយ deterministic
    ├── verify.sh                  # ដំណើរការគ្រប់ការត្រួតពិនិត្យក្នុងពាក្យបញ្ជាតែមួយ
    └── verify_clean_clone.sh      # simulate RTD build លើ clone ស្អាតពី GitHub
```

### ហេតុអ្វីឈ្មោះឯកសារជាអង់គ្លេស?

ឈ្មោះឯកសារភាសាខ្មែរ (ឧ. `ចាប់ផ្ដើម.md`) បង្កើត URL ដែលត្រូវ percent-encode
(`%E1%9E%85%E1%9E%B6%E1%9E%94...`) ដែលធ្វើឱ្យ Search, Cross-references និងការចែករំលែកតំណពិបាក។
ដូច្នេះគម្រោងនេះប្រើ **ឈ្មោះឯកសារអង់គ្លេស + ចំណងជើងភាសាខ្មែរ**។

## ការសាងសង់ឯកសារនៅក្នុងម៉ាស៊ីនរបស់អ្នក (Build locally)

តម្រូវការ៖ Python 3.10+ និង PyTorch ≥ 2.0 (PyTorch ចាំបាច់សម្រាប់ **សាកល្បងកូដឧទាហរណ៍** ប៉ុណ្ណោះ — ការ build ឯកសារមិនត្រូវការវាទេ)។

```bash
# 1) បង្កើត virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) ដំឡើង dependencies របស់ Sphinx
pip install -r docs/requirements.txt

# 3) Build ជា HTML
sphinx-build -b html docs/source docs/_build/html

# 4) មើលក្នុង browser
python -m http.server -d docs/_build/html 8000
```

Build ដែលប្រើសម្រាប់ CI/ផ្ទៀងផ្ទាត់ (zero warnings)៖

```bash
sphinx-build -b html docs/source docs/_build/html -q -W --keep-going
```

## ការសាកល្បងកូដឧទាហរណ៍ (Test the code examples)

កូដឧទាហរណ៍ទាំងអស់ស្ថិតនៅក្នុង `docs/source/examples/` ហើយត្រូវបានបញ្ចូលទៅក្នុងឯកសារដោយ
`literalinclude` — ដូច្នេះអ្វីដែលអ្នកអាន គឺជាអ្វីដែលត្រូវបានសាកល្បងពិត។

```bash
# ដំឡើង PyTorch (CPU) — សូមមើល https://pytorch.org/get-started/locally/ សម្រាប់ជម្រើស CUDA
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy

# សាកល្បងឧទាហរណ៍ទាំងអស់
python tests/run_examples.py
```

ឧទាហរណ៍ដែលបង្ហាញ **កំហុសដោយចេតនា** មានសរសេរ `# EXPECTED FAILURE:` ពីលើ
ហើយ test runner ផ្ទៀងផ្ទាត់ថាពួកវាពិតជាបរាជ័យដោយសារមូលហេតុត្រឹមត្រូវ។

## ពាក្យបញ្ជាផ្ទៀងផ្ទាត់ (Verification)

រត់ទាំងអស់មុន commit — ទាំងអស់ត្រូវតែ exit 0៖

```bash
# ១. កូដឧទាហរណ៍
python tests/run_examples.py
python tests/check_examples_referenced.py

# ២. MyST fence lint (ការរំលងនេះបំផ្លាញទំព័រទាំងមូល)
python tools/check_markdown.py

# ៣. Contrast (WCAG AA ≥ 4.5:1 សម្រាប់គ្រប់គូពណ៌)
python tools/check_contrast.py

# ៣b. UI ជាភាសាខ្មែរ (ត្រូវ build ជាមុន)
python tests/check_locale.py
python tests/check_ui_strings.py

# ៤. Sphinx build (ត្រូវតែ 0 warnings)
sphinx-build -b html docs/source docs/_build/html -q -W --keep-going

# ៥. Offline preview សម្រាប់ត្រួតពិនិត្ថដោយភ្នែក
python tools/make_preview.py --page style-guide.html
open preview.html        # ឬ xdg-open / start
```

ឬដំណើរការទាំងអស់ក្នុងពាក្យបញ្ជាតែមួយ (ណែនាំមុន push)៖

```bash
bash tools/verify.sh      # ១១ ការត្រួតពិនិត្យ ជា ៣ ដំណាក់កាល; exit 0 លុះត្រាតែឆ្លងទាំងអស់
```

មុន push ទៅ Read the Docs សូមរត់ការពិនិត្យលើ **clone ស្អាត** ផងដែរ៖

```bash
bash tools/verify_clean_clone.sh
```

វា clone ពី GitHub ហើយដំណើរការលំដាប់ RTD ពិតប្រាកដ (pre_build → sphinx ជាមួយ
flags របស់ RTD → post_build ជាមួយ `$READTHEDOCS_OUTPUT`)។ `verify.sh` មិនអាចចាប់យក
ការត្រួតពិនិត្យដែលពឹងលើស្ថានភាពម៉ាស៊ីនបានទេ ព្រោះវារត់លើ working tree ដែលមាន
build directory និង mtime ជាស្រេច — នេះជាមូលហេតុដែល RTD build ពីរលើកជាប់គ្នា
បានបរាជ័យដោយសារ check ដែលឆ្លងកាត់ក្នុងម៉ាស៊ីន។

មើល [docs/BUILD_REPORT.md](docs/BUILD_REPORT.md) សម្រាប់លទ្ធផលដែលបានវាស់
និងដែនកំណត់ដែលគេស្គាល់។

## Google Search Console verification (កុំលុប)

គេហទំព័រនេះផ្ទៀងផ្ទាត់ម្ចាស់កម្មសិទ្ធិតាម **ពីរវិធី** ក្នុងពេលតែមួយ៖

```{list-table}
:header-rows: 1
:widths: 20 44 36

* - វិធី
  - ឯកសារ
  - URL ដែលបម្រើជាក់ស្ដែង
* - **Meta tag** (វិធីដែលដំណើរការ)
  - `_templates/layout.html` បញ្ចេញ `<meta name="google-site-verification">` លើ **គ្រប់ទំព័រ** ដោយអានតម្លៃពី `google_site_verification` ក្នុង `conf.py`
  - `https://pytorch-documentation.readthedocs.io/` → redirect → `/km/latest/` ដែលមាន tag
* - **HTML file**
  - `docs/source/extra/google59471f16170f14f0.html` ចម្លងដោយ `html_extra_path`
  - ⚠️ `/km/latest/google59471f16170f14f0.html` តែប៉ុណ្ណោះ — **មិនមែន** domain root
```

````{admonition} ហេតុអ្វី HTML-file method បរាជ័យលើ Read the Docs
:class: important
Read the Docs mount build នៅ `/<language>/<version>/`។ គម្រោងនេះកំណត់
`language = "km"` ដូច្នេះ site ស្ថិតនៅ `/km/latest/`។ **គ្មានអ្វីដែល Sphinx បង្កើត
អាចបម្រើនៅ domain root បានទេ** មិនថា `html_extra_path` មានអ្វីក៏ដោយ៖

```text
/google59471f16170f14f0.html            -> 404
/km/latest/google59471f16170f14f0.html  -> 200 (53 bytes)
/en/latest/                             -> 404 (ភាសាគឺ km មិនមែន en)
```

ដូច្នេះ Google ត្រូវផ្ទៀងផ្ទាត់ដោយ **meta tag** ឬអ្នកត្រូវចុះឈ្មោះ property
ជា URL-prefix `https://pytorch-documentation.readthedocs.io/km/latest/`។
````

ឯកសារ HTML នៅតែត្រូវបានរក្សាទុក ព្រោះវាដំណើរការសម្រាប់ URL-prefix property។

`tests/check_search_console.py` ការពារវិធីទាំងពីរ — ព្រោះ Google ត្រួតពិនិត្យជាប្រចាំ
ហើយបើឯកសារ ឬ tag បាត់ property នឹងត្រូវដកការផ្ទៀងផ្ទាត់ដោយស្ងាត់ ដោយគ្មាន
អ្វីក្នុង build បង្ហាញថាមានបញ្ហា (Sphinx នៅតែ exit 0)។

```{admonition} google-site-verification token មិនមែនជា secret ទេ
:class: note
តម្លៃនេះត្រូវតែលេចឡើងក្នុង HTML ដែលបម្រើ ព្រោះ Google អានវាពីទីនោះ។
ការ commit វាគឺត្រឹមត្រូវ ហើយមិនមែនជាការលេចធ្លាយ credential ទេ — វាមិនផ្ដល់
សិទ្ធិចូលដំណើរការអ្វីទាំងអស់។ កុំ "rotate" វាដូចជា secret។
```

## Light Mode និង Dark Mode## Light Mode និង Dark Mode

គេហទំព័រនេះគាំទ្រពណ៌ទាំងពីរ៖

- ដោយស្វ័យប្រវត្តិតាមប្រព័ន្ធ (`prefers-color-scheme`)
- ឬជ្រើសរើសដោយផ្ទាល់តាមការគ្រប់គ្រង **ពណ៌បង្ហាញ → ងងឹត / ភ្លឺ** នៅ Sidebar
  (ជម្រើសសកម្មត្រូវបានបំពេញពណ៌ ហើយការជ្រើសរើសត្រូវបានរក្សាទុកក្នុង `localStorage`)

Palette ទាំងមូលស្ថិតនៅក្នុង `docs/source/_static/css/custom.css` ជា CSS custom properties
(`--kh-page-bg`, `--kh-text` ។ល។) — ពណ៌ទាំងអស់ឆ្លងកាត់កម្រិត Contrast **≥ 4.5:1** (WCAG AA)។

## ចូលរួម (Contributing)

សូមអាន [CONTRIBUTING.md](CONTRIBUTING.md) ជាមុនសិន — វាមានច្បាប់ស្ដីពី
ការបកប្រែ ការរក្សាឈ្មោះ API និងរបៀបសាកល្បងកូដ។

## អាជ្ញាបណ្ណ (License)

ខ្លឹមសារឯកសារ៖ **CC BY 4.0**។ កូដឧទាហរណ៍៖ **BSD 3-Clause** (ដូច PyTorch)។
PyTorch និងរូបសញ្ញា PyTorch គឺជាស្លាកសញ្ញារបស់ PyTorch Foundation។
មើល [LICENSE](LICENSE)។
