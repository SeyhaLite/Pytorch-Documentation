# របាយការណ៍ Build និង Test

ឯកសារនេះកត់ត្រាលទ្ធផល **ដែលបានវាស់ពិត** ពី environment បង្កើតឯកសារ។
លេខទាំងអស់អាចបង្កើតឡើងវិញបានដោយពាក្យបញ្ជាដែលបានបញ្ជាក់។

```{admonition} Environment ដែលបានប្រើ
:class: note
```text
OS            : Linux (sandboxed container)
Python        : 3.11.2
PyTorch       : 2.14.1+cpu
NumPy         : 2.4.6
Sphinx        : 7.4.7
sphinx-rtd-theme : 3.1.0
myst-parser   : 3.x
CUDA available: False   ← គ្មាន GPU ក្នុង environment នេះ
CPU threads   : 8
```
```

## ១. សង្ខេប

```{list-table}
:header-rows: 1
:widths: 46 20 34

* - ការវាស់
  - លទ្ធផល
  - ពាក្យបញ្ជាផ្ទៀងផ្ទាត់
* - Sphinx build (`-W --keep-going`)
  - **exit 0 · 0 warnings**
  - `sphinx-build -b html docs/source docs/_build/html -q -W --keep-going`
* - ទំព័រ HTML ដែលបានបង្កើត
  - **24**
  - `ls docs/_build/html/*.html | wc -l`
* - រយៈពេល build
  - **~6 វិនាទី**
  - `time sphinx-build ...`
* - កូដឧទាហរណ៍សរុប
  - **40 ឯកសារ · 5,661 បន្ទាត់**
  - `ls docs/source/examples/*.py | wc -l`
* - ឧទាហរណ៍ដែលបានរត់ **pass**
  - **38**
  - `python tests/run_examples.py`
* - ឧទាហរណ៍ **fail**
  - **0**
  - ដូចគ្នា
* - ឧទាហរណ៍ **skip** (ត្រូវការ GPU)
  - **2** — `mixed_precision_2`, `performance_2`
  - ដូចគ្នា
* - ឧទាហរណ៍ orphan / stale reference
  - **0**
  - `python tests/check_examples_referenced.py`
* - បន្ទាត់ឯកសារ Markdown សរុប
  - **8,007** (22 ទំព័រ)
  - មើលតារាងផ្នែក ៣
* - Contrast pairs ផ្ទៀងផ្ទាត់
  - **104 · ទាំងអស់ ≥ 4.5:1**
  - `python tools/check_contrast.py`
* - MyST fence problems
  - **0**
  - `python tools/fix_myst_fences.py --check`
* - តំណខាងក្នុង
  - **780+ · 0 បាក់**
  - ស្គ្រីបក្នុងផ្នែក ៤
* - JS syntax (`node --check`)
  - **pass** ទាំង ២ (copybutton.js + inline theme script)
  - `node --check docs/source/_static/js/copybutton.js`
* - Search index
  - **22 docnames · 3,291 terms**
  - មើលផ្នែក ៥
```

## ២. លទ្ធផល Test ឧទាហរណ៍

ចម្លងពីការរត់ `python tests/run_examples.py` (PyTorch 2.14.1+cpu, Python 3.11.2)៖

```text
Examples found: 40
------------------------------------------------------------------------
PASS  autograd_1                            2.22s  rc=0
PASS  autograd_2                            2.16s  rc=0
PASS  autograd_3                            3.20s  rc=0
PASS  checkpoints_1                         3.39s  rc=0
PASS  checkpoints_2                         3.28s  rc=0
PASS  data_and_dataloader_1                 1.69s  rc=0
PASS  data_and_dataloader_2                 2.14s  rc=0
PASS  data_and_dataloader_3                 1.88s  rc=0
PASS  debugging_1                           3.60s  rc=0
PASS  debugging_2                           1.88s  rc=0
PASS  debugging_3                           1.74s  rc=1  (expects RuntimeError)
PASS  devices_1                             1.75s  rc=0
PASS  evaluation_and_inference_1            1.74s  rc=0
PASS  evaluation_and_inference_2            5.88s  rc=0
PASS  getting_started_1                     1.81s  rc=0
PASS  getting_started_2                     3.37s  rc=0
PASS  installation_1                        1.65s  rc=0
PASS  loss_and_optimizers_1                 1.65s  rc=0
PASS  loss_and_optimizers_2                 3.56s  rc=0
PASS  loss_and_optimizers_3                 3.53s  rc=0
PASS  mixed_precision_1                     3.53s  rc=0
SKIP  mixed_precision_2                     (no CUDA device available)
PASS  neural_networks_1                     1.66s  rc=0
PASS  neural_networks_2                     1.67s  rc=0
PASS  neural_networks_3                     1.59s  rc=0
PASS  performance_1                         5.32s  rc=0
SKIP  performance_2                         (no CUDA device available)
PASS  reproducibility_1                     3.33s  rc=0
PASS  saving_and_loading_1                  1.61s  rc=0
PASS  saving_and_loading_2                  3.80s  rc=0
PASS  shapes_1                              1.65s  rc=0
PASS  shapes_2                              1.75s  rc=0
PASS  shapes_3                              1.62s  rc=0
PASS  tensors_1                             1.65s  rc=0
PASS  tensors_2                             1.75s  rc=0
PASS  tensors_3                             1.65s  rc=0
PASS  tensors_4                             1.68s  rc=0
PASS  tensors_5                             1.71s  rc=0
PASS  training_loop_1                       3.71s  rc=0
PASS  training_loop_2                       3.54s  rc=0
------------------------------------------------------------------------
passed 38 · failed 0 · skipped 2 · total 40
```

`debugging_3.py` មាន `rc=1` **ដោយការរចនា** — វាបង្ហាញ traceback ពិត។
ស្គ្រីបសាកល្បងផ្ទៀងផ្ទាត់ថាវាបរាជ័យដោយសារ `RuntimeError` និងមានពាក្យ
`shape` ក្នុង output (តាម marker `# EXPECTED FAILURE:`)។

### របៀបដែលឧទាហរណ៍ត្រូវបានសាកល្បង

```{list-table}
:header-rows: 1
:widths: 34 66

* - លក្ខណៈ
  - ការអនុវត្ត
* - ភាពដាច់ដោយឡែក
  - ឧទាហរណ៍នីមួយៗរត់ជា **subprocess ដាច់ដោយឡែក** ក្នុង temp cwd — គ្មាន state ចែករួម
* - Reproducibility
  - `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
* - ឯកសារមិនសរសេរចូល repo
  - `KH_EXAMPLE_TMPDIR` ចង្អុលទៅ temp dir; ឧទាហរណ៍ដែលរក្សាទុកឯកសារសម្អាតវាវិញ
* - Assertions
  - ឧទាហរណ៍នីមួយៗមាន `assert` លើ shape, dtype, តម្លៃ និងលទ្ធផលរំពឹងទុក
* - Timeout
  - 180 វិនាទីក្នុងមួយឯកសារ (អាច override ដោយ `# TIMEOUT:`)
* - Expected failures
  - `# EXPECTED FAILURE:` + `# EXPECTED OUTPUT:` markers
* - Skips
  - `# REQUIRES: gpu` / `# REQUIRES: <module>`
```

## ៣. មាតិកាតាមទំព័រ

```{list-table}
:header-rows: 1
:widths: 30 10 8 10 12 30

* - ទំព័រ
  - បន្ទាត់
  - H2
  - តារាង
  - ឧទាហរណ៍
  - ស្ថានភាពកូដ
* - `api-quick-reference`
  - 729
  - 15
  - 18
  - 0
  - — (តារាងយោង)
* - `tensors`
  - 517
  - 10
  - 10
  - 5
  - ✅ ៥/៥ បានរត់
* - `neural-networks`
  - 509
  - 11
  - 9
  - 3
  - ✅ ៣/៣ បានរត់
* - `debugging`
  - 466
  - 10
  - 8
  - 3
  - ✅ ៣/៣ បានរត់
* - `performance`
  - 400
  - 12
  - 7
  - 2
  - ✅ ១ · 🔒 ១ (GPU)
* - `data-and-dataloader`
  - 383
  - 9
  - 4
  - 3
  - ✅ ៣/៣ បានរត់
* - `loss-and-optimizers`
  - 379
  - 7
  - 6
  - 3
  - ✅ ៣/៣ បានរត់
* - `saving-and-loading`
  - 362
  - 11
  - 4
  - 2
  - ✅ ២/២ បានរត់
* - `style-guide`
  - 536
  - 15
  - 8
  - 1
  - ✅ ១/១ បានរត់
* - `distributed-training`
  - 341
  - 10
  - 5
  - 0
  - ⚠️ គ្មានឧទាហរណ៍បានរត់
* - `autograd`
  - 332
  - 11
  - 3
  - 3
  - ✅ ៣/៣ បានរត់
* - `reproducibility`
  - 329
  - 9
  - 4
  - 1
  - ✅ ១/១ បានរត់
* - `shapes-and-broadcasting`
  - 324
  - 7
  - 4
  - 3
  - ✅ ៣/៣ បានរត់
* - `training-loop`
  - 316
  - 7
  - 2
  - 2
  - ✅ ២/២ បានរត់
* - `evaluation-and-inference`
  - 314
  - 8
  - 6
  - 2
  - ✅ ២/២ បានរត
* - `mixed-precision`
  - 296
  - 8
  - 5
  - 2
  - ✅ ១ · 🔒 ១ (GPU)
* - `checkpoints`
  - 283
  - 7
  - 5
  - 2
  - ✅ ២/២ បានរត់
* - `devices`
  - 278
  - 11
  - 4
  - 1
  - ✅ ១/១ បានរត់
* - `getting-started`
  - 277
  - 5
  - 6
  - 2
  - ✅ ២/២ បានរត់
* - `installation`
  - 274
  - 7
  - 2
  - 1
  - ✅ ១/១ បានរត់
* - `index`
  - 188
  - 5
  - 2
  - 0
  - —
* - `license`
  - 174
  - 7
  - 1
  - 0
  - —
* - **សរុប**
  - **8,007**
  - **202**
  - **123**
  - **42 refs**
  - 40 ឯកសារដាច់ដោយឡែក
```

## ៤. ការផ្ទៀងផ្ទាត់លទ្ធផល HTML

### CSS load order

ត្រូវបានផ្ទៀងផ្ទាត់ក្នុង `<head>` នៃ `index.html` ដែលបាន build៖

```text
1. _static/pygments.css          (Pygments លំនាំដើម)
2. _static/css/theme.css         (sphinx_rtd_theme)
3. _static/css/print.css         (media="print")
4. _static/css/custom.css        ← ផ្ទុកចុងក្រោយ ដូច្នេះឈ្នះ cascade
```

### Selectors ដែលបានផ្ទៀងផ្ទាត់ថាមានវត្តមាន

**៤៤ selectors** ត្រូវបានពិនិត្យក្នុង `_static/css/custom.css` ដែលបាន build —
**មានវត្តមានទាំងអស់ ៤៤**។ គ្របដណ្ដប់៖

```{list-table}
:header-rows: 1
:widths: 26 74

* - ក្រុម
  - Selectors ដែលបានផ្ទៀងផ្ទាត់
* - តារាង
  - `table.docutils:not(.field-list) tr:nth-child(2n-1) td`, `…(2n) td`, `table.docutils th`
* - Inline chips
  - `kbd`, `.guilabel`, `.file`, `.samp`, `.rst-content code`
* - Code blocks
  - `.highlight pre`, Pygments `.c1 .k .s .mi .nf` (dark) និង `.k` (light)
* - Sidebar
  - `li.current li a`, `li.toctree-l2.current li.toctree-l3 > a`, `p.caption`, `::-webkit-scrollbar-thumb`, `input[type="text"]`
* - Content
  - `a:visited`, `.wy-breadcrumbs`, `.admonition-title`, `.warning`, `.seealso`, `blockquote`, `.figure p.caption`, `.footnote`, `footer`
* - Prev/next និង search
  - `.btn-neutral:visited`, `ul.search li`, `ul.search li div.context`, `mark`, `span.highlighted`
* - Controls
  - `button.kh-theme-toggle`, `.kh-copy-btn`
* - Accessibility
  - `a:focus-visible`, `@media (prefers-reduced-motion: reduce)`, `@media (hover: none)`, `@media screen and (max-width: 768px)`
* - Fonts
  - `Kantumruy Pro`, `Noto Sans Khmer`, `--kh-page-bg: #14161a`, `--kh-page-bg: #ffffff`
```

### តំណខាងក្នុង

```text
internal .html links : 780+
broken               : 0
```

### Dark / Light mode

```{list-table}
:header-rows: 1
:widths: 40 60

* - លក្ខណៈ
  - ស្ថានភាព
* - Palette tokens
  - `data-theme="dark"` 64 កន្លែង · `data-theme="light"` 120 កន្លែង ក្នុង custom.css
* - ការជ្រើសរើសដំបូង
  - `localStorage` → `prefers-color-scheme` → dark
* - ការគ្រប់គ្រង
  - ធាតុអត្ថបទធម្មតាតែមួយ (`<button id="kh-theme-toggle">`) គ្មាន icon គ្មាន box គ្មាន border
* - ទីតាំង `<script>`
  - **inline ភ្លាមៗបន្ទាប់ពីធាតុ** — ផ្ទៀងផ្ទាត់ថាគ្មានអ្វីនៅចន្លោះ `</button>` និង `<script>`
* - ភាពឯករាជ្យពី `html_js_files`
  - `conf.py` មិនមាន `html_js_files` ទេ។ ស្គ្រីប copy ប្រើ `loading_method="defer"`
* - `color-scheme`
  - កំណត់ទាំង dark និង light ដើម្បីឱ្យ form controls និង scrollbars ត្រូវគ្នា
```

### Contrast (WCAG AA)

`tools/check_contrast.py` ផ្ទៀងផ្ទាត់ **១០៤ គូ** ក្នុង theme ទាំងពីរ៖

```{list-table}
:header-rows: 1
:widths: 40 20 40

* - ក្រុម
  - ចំនួនគូ
  - លទ្ធផល
* - Dark — page palette
  - 15
  - ✅ ទាំងអស់ ≥ 4.5:1 (តិចបំផុត 5.31:1)
* - Dark — chrome (sidebar, table header, toggle)
  - 14
  - ✅ ទាំងអស់ ≥ 4.5:1 (តិចបំផុត 5.92:1)
* - Dark — Pygments tokens
  - 12
  - ✅ ទាំងអស់ ≥ 4.5:1 (តិចបំផុត 5.31:1 comment)
* - Dark — admonition titles
  - 5
  - ✅ ទាំងអស់ ≥ 4.5:1
* - Dark — admonition bodies
  - 9
  - ✅ ទាំងអស់ ≥ 4.5:1
* - Light — page palette
  - 15
  - ✅ ទាំងអស់ ≥ 4.5:1 (តិចបំផុត 6.00:1)
* - Light — chrome
  - 8
  - ✅ ទាំងអស់ ≥ 4.5:1
* - Light — Pygments tokens
  - 12
  - ✅ ទាំងអស់ ≥ 4.5:1
* - Light — admonition titles
  - 5
  - ✅ ទាំងអស់ ≥ 4.5:1
* - Light — admonition bodies
  - 9
  - ✅ ទាំងអស់ ≥ 4.5:1
```

ឧទាហរណ៍លេខដែលបានវាស់៖

```text
DARK  body text  #d9dde4 on #14161a  = 13.29:1  PASS
DARK  muted      #8b949e on #1c2027  =  5.31:1  PASS
DARK  chip text  #dfe3ea on #262b33  = 11.05:1  PASS
LIGHT body text  #24292f on #ffffff  = 14.65:1  PASS
LIGHT link       #0b5cad on #f6f8fa  =  6.27:1  PASS
```

## ៥. ដែនកំណត់ដែលគេស្គាល់

```{list-table}
:header-rows: 1
:widths: 30 70

* - ដែនកំណត់
  - ការពន្យល់ និងផលប៉ះពាល់
* - **ឧទាហរណ៍ GPU មិនបានរត់**
  - `mixed_precision_2.py` និង `performance_2.py` មាន `# REQUIRES: gpu` ហើយត្រូវបាន **រំលង**។ កូដផ្អែកលើ API ដែលបានផ្ទៀងផ្ទាត់ថាមាន (`hasattr` checks) ប៉ុន្តែ **មិនបានដំណើរការ**។ ទំព័រទាំងពីរសម្គាល់ច្បាស់ថា ⚠️ មិនទាន់បានសាកល្បង។
* - **Distributed training គ្មានឧទាហរណ៍បានរត់**
  - ត្រូវការ multi-GPU ឬ multi-node។ ទំព័រ `distributed-training` ជាអត្ថបទ + កូដ **បង្ហាញជាឧទាហរណ៍** ប៉ុណ្ណោះ ហើយមាន admonition ព្រមាននៅផ្នែកខាងលើ។
* - **ការស្វែងរកជាភាសាខ្មែរមានដែនកំណត់**
  - ភាសាខ្មែរសរសេរដោយគ្មានដកឃ្លារវាងពាក្យ។ Sphinx គ្មាន Khmer tokenizer ទេ ដូច្នេះ `html_search_language = "en"`។ លទ្ធផល៖ ពាក្យ API ជាអក្សរឡាតាំង (`torch`, `backward`, `checkpoint`, `gradient`) ស្វែងរកបានត្រឹមត្រូវ; ពាក្យខ្មែរត្រូវបាន index ជា chunk រវាងសញ្ញាខណ្ឌ។ ការស្វែងរកពាក្យខ្មែរជាក់លាក់អាចមិនពេញលេញ។
* - **មិនបានសាកល្បងក្នុង browser ពិត**
  - គ្មាន browser/GUI ក្នុង environment នេះ។ Layout ត្រូវបានផ្ទៀងផ្ទាត់ដោយ (ក) វត្តមាន selectors ក្នុង CSS ដែលបាន build, (ខ) JS syntax check ដោយ `node --check`, (គ) contrast គណនាដោយកូដ, និង (ឃ) `preview.html` សម្រាប់អ្នកប្រើត្រួតពិនិត្ថដោយភ្នែក។ **ការត្រួតពិនិត្ថដោយភ្នែកនៅតែចាំបាច់។**
* - **ពុម្ពអក្សរខ្មែរផ្ទុកពី CDN**
  - `@import` ពី Google Fonts។ បើ offline ឬ CDN ត្រូវបានរារាំង វាធ្លាក់ទៅ fallback stack (`Khmer OS`, `Noto Sans Khmer` ក្នុងម៉ាស៊ីន ។ល។)។ ពុម្ពអក្សរ **មិនបាន** រក្សាទុកក្នុង repo ដើម្បីរក្សាទំហំឱ្យតូច។ `preview.html` ដក `@import` ចេញ ហើយបង្ហាញ banner ពន្យល់។
* - **លេខល្បឿនមិនអាចផ្ទេរបាន**
  - រាល់ការវាស់រយៈពេលក្នុង `performance_1.py` និង `evaluation_and_inference_2.py` គឺជាក់លាក់ចំពោះម៉ាស៊ីនដែលរត់វា។ ទំព័រទាំងពីរមាន admonition បញ្ជាក់ច្បាស់។
```

## ៦. បញ្ហាដែលបានរកឃើញ និងជួសជុលកំឡុងពេលអភិវឌ្ឍ

ឧទាហរណ៍ទាំងអស់មាន `assert` ពិតប្រាកដ — ហើយ assertions ទាំងនោះបានចាប់យក
កំហុសពិតជាច្រើនកំឡុងពេលសរសេរ។ នេះជាភស្តុតាងថាការសាកល្បងមានតម្លៃ៖

```{list-table}
:header-rows: 1
:widths: 30 34 36

* - ឯកសារ
  - បញ្ហា
  - ការជួសជុល
* - `getting_started_2`
  - ការបង្កើតទិន្នន័យខុស (`randn(n,2) + offset`) ធ្វើឱ្យ class ជាន់គ្នា → accuracy < 0.9
  - បង្កើត cluster ដាច់ដោយឡែក `cat([base - 3.0, base + 3.0])`
* - `tensors_3`
  - `assert x[mask]` ភ្លេចរាប់ 15 ក្នុង `x > 10`
  - បន្ថែម 15 ទៅក្នុងលទ្ធផលរំពឹងទុក
* - `tensors_3`
  - `clamped[clamped <= 10].numel() == 0` ខុស ព្រោះតម្លៃ 0 ត្រូវបានជំនួស
  - ពិនិត្យចន្លោះ `[1, 10]` ជំនួសវិញ
* - `tensors_4`
  - `0.1 + 0.2 == 0.3` គឺ **ពិត** ក្នុង float32 (ខុសពី float64) ដូច្នេះឧទាហរណ៍ underflow មិនដំណើរការ
  - ប្ដូរទៅ `0.3 - 0.1 vs 0.2` និង `sum(0.1 × 10) vs 1.0` ដែលបង្ហាញកំហុសពិត
* - `shapes_2`
  - `torch.full((4,3), 2.0)` ខុស — broadcasting `[1,2,3]` លើជួរនីមួយៗផ្ដល់ `2,3,4`
  - កែទៅ `tensor([[2.,3.,4.]] * 4)`
* - `loss_and_optimizers_3`
  - `ReduceLROnPlateau.step()` ទាមទារ `metrics` argument → `TypeError`
  - បន្ថែម `needs_metric` flag និងពន្យល់ភាពខុសគ្នានេះជាមេរៀន
* - `autograd_1`
  - ខ្ញុំសរសេរ `assert z.grad == 60.0` ប៉ុន្តែតម្លៃពិតគឺ `30.0` (d(z*10)/dz = 10 × 3 ដង)
  - កែ assertion និងអត្ថបទពន្យល់
* - `checkpoints_2`
  - អះអាងថា `strict=False` ដោះស្រាយ shape mismatch — **ខុស** វាអត់ឱនឱ្យតែ key បាត់/លើស
  - សរសេរឡើងវិញ៖ បង្ហាញ RuntimeError បន្ទាប់មកបង្ហាញដំណោះស្រាយពិត (តម្រង shape ដោយដៃ)
* - `saving_and_loading_1`
  - `torch.tensor(["cat","dog","bird"])` → `ValueError: too many dimensions 'str'`
  - ប្រើ Python list + dict ហើយបង្រៀនកំហុសនេះដោយចេតនា
* - `performance_1`
  - ប្រៀបធៀប `big.mean(dim=0)` ជាមួយ per-row means (គួរជា `dim=1`) → diff 1.04e-01
  - កែ ហើយបន្ថែមផ្នែកពន្យល់អំពីអត្ថន័យ `dim=` ព្រោះវាជាកំហុសទូទៅ
```

### បញ្ហា Sphinx / MyST ដែលបានរកឃើញ

```{list-table}
:header-rows: 1
:widths: 34 66

* - បញ្ហា
  - ការដោះស្រាយ
* - admonition ដែលមាន nested code block បំផ្លាញទំព័រទាំងមូល
  - CommonMark បិទ fence ដោយ backtick ចំនួនដូចគ្នា។ ដំណោះស្រាយ៖ outer directive ប្រើ ៤ backticks។ សរសេរ `tools/fix_myst_fences.py` ដើម្បី lint + repair ដោយស្វ័យប្រវត្តិ និងបន្ថែមច្បាប់ទៅ `CONTRIBUTING.md`
* - backtick ក្នុង admonition title ធ្វើឱ្យ fence state machine ខូច
  - ដក backtick ចេញពីចំណងជើងទាំងអស់ (API names នៅដដែល គ្រាន់តែគ្មាន code markup)
* - `add_js_file(..., body="end")` មិនមែនជាប៉ារ៉ាម៉ែត្រពិតរបស់ Sphinx
  - វាត្រូវបានបោះចោលជា HTML attribute ហើយស្គ្រីបនៅតែ blocking ក្នុង `<head>`។ ដំណោះស្រាយ៖ `loading_method="defer"` — ផ្ទៀងផ្ទាត់ថា `defer="defer"` លេចឡើងក្នុង output
* - `display_version` និង `github_url` theme options មិនត្រូវបានគាំទ្រក្ន sphinx_rtd_theme 3.1
  - ដកចេញពី `html_theme_options`; GitHub link ត្រូវបានកំណត់តាម `html_context`
* - `{version}` role មិនមាន
  - ជំនួសដោយអក្សរធម្មតា
* - `{mark}` role មិនមាន
  - ជំនួសដោយ `<mark>` HTML inline (MyST អនុញ្ញាត)
* - nested fence repair script ដំបូងបានបំផ្លាញ nested fence closes
  - សរសេរ linter ឡើងវិញដោយប្រើ stack parser ត្រឹមត្រូវ (CommonMark: closing fence ត្រូវមានប្រវែង ≥ opening)
```

### បញ្ហារចនាដែល check_contrast.py រកឃើញ

ឧបករណ៍ផ្ទៀងផ្ទាត់បានរកឃើញ **៧ គូដែលបរាជ័យ** ដែលការត្រួតពិនិត្ថដោយភ្នែកមិនឃើញ៖

```{list-table}
:header-rows: 1
:widths: 40 60

* - បញ្ហា
  - ការជួសជុល
* - Admonition titles៖ អក្សរស `#ffffff` លើ accent hue ភ្លឺ (`#3fb950`, `#d29922`, `#f85149`, `#b392f0`) = 2.5:1–3.4:1
  - ប្ដូរទម្រង់៖ ឥឡូវ **bar ងងឹតដែលមាន saturation** + **ស្លាក accent ភ្លឺ**។ បន្ថែម tokens `--kh-<kind>-title-bg` / `--kh-<kind>-title-fg` សម្រាប់ theme ទាំងពីរ
* - Inline chip text `#dfe3ea` ត្រូវបាន hardcode ជា dark-only → ក្នុង light mode វានៅលើ `#eef1f5` = 1.14:1
  - ជំនួសដោយ `var(--kh-chip-text)` (mode-aware)
* - `kbd` text `#e6eaf0` hardcode → 1.07:1 ក្នុង light mode
  - ដូចគ្នា — ប្រើ token
* - Sidebar ប្រើ hex ងងឹត (`#c8cfd8` ។ល។) ដែលសមស្របតែលើផ្ទៃងងឹត → light mode sidebar (`#f3f4f6`) មិនអាចអានបាន
  - បន្ថែមផ្នែក "Light-mode chrome" ពេញលេញ៖ nav items, captions, current page, scrollbars, version flyout, theme toggle, search placeholder
```

បន្ទាប់ពីជួសជុល៖ **១០៤/១០៤ គូ pass**។

## ៧. ឯកសារដែលបានបង្កើត

```{list-table}
:header-rows: 1
:widths: 44 14 42

* - ឯកសារ
  - បន្ទាត់
  - តួនាទី
* - `docs/source/_static/css/custom.css`
  - 1,396
  - Palette dark + light ទាំងមូល (CSS custom properties)
* - `docs/source/_static/css/print.css`
  - 61
  - Print stylesheet (media="print")
* - `docs/source/_static/js/copybutton.js`
  - 87
  - ប៊ូតុងចម្លងអក្សរធម្មតា សម្រាប់រាល់ code block
* - `docs/source/_templates/searchbox.html`
  - 81
  - Sidebar override៖ ការគ្រប់គ្រង dark/light + inline script + search box
* - `docs/source/conf.py`
  - 166
  - Sphinx configuration
* - `docs/source/_ext/khmer_assets.py`
  - 46
  - ចុះឈ្មោះ assets ជាមួយ `loading_method="defer"`
* - `tools/fix_myst_fences.py`
  - 168
  - MyST fence linter + auto-repair (CI `--check`)
* - `tools/check_contrast.py`
  - 203
  - WCAG AA contrast verifier (១០៤ គូ)
* - `tools/make_preview.py`
  - 181
  - បង្កើត offline preview HTML តែមួយ
* - `tests/run_examples.py`
  - 329
  - ឧបករណ៍សាកល្បងឧទាហរណ៍ (isolation, markers, JSON report)
* - `tests/check_examples_referenced.py`
  - 65
  - ការពារ orphan/stale example references
```

## ៨. ពាក្យបញ្ជាផ្ទៀងផ្ទាត់ទាំងអស់

```bash
# ១. សាកល្បងកូដឧទាហរណ៍ទាំងអស់
python tests/run_examples.py
python tests/run_examples.py --json report.json      # របាយការណ៍ machine-readable

# ២. ពិនិត្យថាគ្រប់ឧទាហរណ៍ត្រូវបានយោង
python tests/check_examples_referenced.py

# ៣. Lint MyST fences
python tools/fix_myst_fences.py --check              # exit 1 បើមានបញ្ហា
python tools/fix_myst_fences.py                      # repair ដោយស្វ័យប្រវត្តិ

# ៤. ផ្ទៀងផ្ទាត់ contrast
python tools/check_contrast.py                       # exit 1 បើ < 4.5:1
python tools/check_contrast.py -q                    # តែបរាជ័យ

# ៥. Build (ត្រូវតែ exit 0 ជាមួយ 0 warnings)
sphinx-build -b html docs/source docs/_build/html -q -W --keep-going

# ៦. បង្កើត offline preview
python tools/make_preview.py --page style-guide.html

# ៧. JS syntax
node --check docs/source/_static/js/copybutton.js
```

## ៩. អ្វីដែលនៅសល់សម្រាប់អ្នកថែទាំនាពេលអនាគត

```{list-table}
:header-rows: 1
:widths: 30 70

* - កិច្ចការ
  - ការពន្យល់
* - សាកល្បងឧទាហរណ៍ GPU
  - រត់ `python tests/run_examples.py mixed performance` លើម៉ាស៊ីនដែលមាន CUDA។ ឯកសារទាំងពីរនឹងបោះពុម្ពលេខវាស់ពិត — បន្ទាប់មកអាចដកសញ្ញា ⚠️ ចេញ
* - សាកល្បង DDP
  - រត់ឧទាហរណ៍ `distributed-training` លើ multi-GPU ហើយបន្ថែមជាឯកសារ `.py` ដែលបានសាកល្បង
* - ការស្វែងរកជាភាសាខ្មែរ
  - បើ Sphinx បន្ថែម Khmer tokenizer ឬអ្នកចង់ប្រើ `sphinxcontrib-jieba`-style plugin សូមពិចារណា។ បច្ចុប្បន្ន `html_search_language = "en"`
* - ពុម្ពអក្សរ offline
  - បើអ្នកចង់ឱ្យឯកសារដំណើរការដោយគ្មានអ៊ីនធឺណិត ទាញយក woff2 ទៅ `_static/fonts/` ហើយប្ដូរ `@import` ទៅ `@font-face` ក្នុងស្រុក (ដាក់បញ្ចូល SIL OFL attribution)
* - បញ្ចូល CI
  - ពាក្យបញ្ជាទាំង ៧ ខាងលើអាចដាក់ក្នុង GitHub Actions។ បច្ចុប្បន្នមានតែ `pre_build` job ក្នុង `.readthedocs.yaml` (ត្រួតពិនិត្យ example references)
* - ការត្រួតពិនិត្ថដោយភ្នែកក្នុង browser
  - បើក `preview.html` ឬ `docs/_build/html/index.html` ក្នុង browser។ ពិនិត្យ dark/light, sidebar nesting, table striping, copy button, focus rings និង responsive layout
```
