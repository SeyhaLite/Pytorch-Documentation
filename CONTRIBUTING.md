# ការចូលរួម (Contributing)

អរគុណដែលចង់ជួយ! ឯកសារនេះគឺជាឯកសារ PyTorch ជាភាសាខ្មែរ ដែលសហគមន៍រួមគ្នាថែទាំ។

## មុនចាប់ផ្ដើម

1. Fork repository នេះ ហើយបង្កើត branch ថ្មីពី `main`។
2. ដំឡើង dependencies៖ `pip install -r docs/requirements.txt`
3. Build និងពិនិត្យមើល៖ `sphinx-build -b html docs/source docs/_build/html`

## ច្បាប់សំខាន់ៗ (Must-read rules)

### ១. កុំបកប្រែឈ្មោះ API

រក្សាឈ្មោះទាំងនេះជាភាសាអង់គ្លេសជានិច្ច៖

`torch`, `torch.nn`, `torch.optim`, `torch.autograd`, `Tensor`, `backward()`,
`requires_grad`, `DataLoader`, `nn.Module`, `state_dict`, `optimizer.step()`,
`loss.backward()`, `torch.cuda`, `AMP`, `DDP` ។ល។

```rst
✅ ត្រូវ៖ "ប្រើ `optimizer.zero_grad()` ដើម្បីសម្អាត gradient"
❌ ខុស៖ "ប្រើ `optimizer.សូន្យក្រាដ()` ..."
```

### ២. បកប្រែការពន្យល់ មិនមែនពាក្យបច្ចេកទេស

ពន្យល់ពាក្យដូចជា Tensor, Gradient, Loss, Optimizer, Backpropagation
ជាភាសាខ្មែរធម្មជាតិ ដោយដាក់ពាក្យអង់គ្លេសក្នុងវង់ក្រចកនៅលើកដំបូង៖

```text
✅ "Gradient គឺជាលេខដែលប្រាប់ថា Loss ផ្លាស់ប្ដូរប៉ុន្មាន
    បើយើងប្ដូរ Weight បន្តិច (ដេរីវេភាគយក partial derivative)។"
```

### ៣. កូដឧទាហរណ៍ត្រូវតែដំណើរការពិត

- ដាក់កូដទាំងអស់ក្នុង `docs/source/examples/<page>_<n>.py`
- បញ្ចូលវាទៅក្នុង Markdown ដោយ៖
  ````markdown
  ```{literalinclude} examples/tensors_1.py
  :language: python
  ```
  ````
- **កុំ** copy/paste កូដចូល Markdown ដោយផ្ទាល់ — វានឹងខុសពីកូដដែលសាកល្បង។
- រត់ `python tests/run_examples.py` មុន commit។ ត្រូវ pass 100%។

ប្រសិនបើឧទាហរណ៍បង្ហាញកំហុសដោយចេតនា សូមដាក់បន្ទាត់ដំបូង៖

```python
# EXPECTED FAILURE: RuntimeError
# EXPECTED OUTPUT: shape
```

### ៤. កុំអះអាងលទ្ធផលដែលមិនបានសាកល្បង

កុំសរសេរ benchmark, ល្បឿន GPU, accuracy ឬ training results ដែលអ្នកមិនបានវាស់ពិត។
សរសេរជាការណែនាំទូទៅវិញ ឬសម្គាល់ច្បាស់ថា «មិនទាន់បានសាកល្បង»។

### ៥. កុំប៉ះ Changelog / Release Notes

ឯកសារប្រវត្តិការផ្លាស់ប្ដូររបស់ PyTorch ត្រូវរក្សាទុកដូចដើម។ កុំបកប្រែ កុំកែ។

### ៦. វិធាន Markdown fence (សំខាន់ណាស់!)

ឯកសារទាំងអស់ជា Markdown (MyST សម្រាប់ `docs/source/`, CommonMark សម្រាប់ GitHub)។
ច្បាប់ទាំងបីខាងក្រោម បើរំលង នឹងធ្វើឱ្យ **ទំព័រទាំងមូលខូច** — ផ្នែកដែលនៅសល់
ទាំងអស់នឹងក្លាយជា code block តែមួយ។

**ច្បាប់ ១៖ fence ខាងក្រៅត្រូវតែវែងជាង fence ខាងក្នុងដែលវាខ្ចប់។**

CommonMark បិទ fence ដោយបន្ទាត់ដែលមាន backtick **ចំនួនដូចគ្នា ឬច្រើនជាង**។
ដូច្នេះ admonition ដែលបើកដោយ ៣ ហើយខ្ចប់ code block ៣ នឹងបិទត្រង់ code block
នោះ មិនមែនត្រង់ចុង admonition ទេ។

ដើម្បីខ្ចប់ code block ក្នុង admonition ប្រើ **៤ backticks** ខាងក្រៅ៖

`````markdown
✅ ត្រូវ៖

````{admonition} ចំណងជើង
:class: tip
```python
x = 1
```
````
`````

`````markdown
❌ ខុស (នឹងបំផ្លាញទំព័រទាំងមូល):

```{admonition} ចំណងជើង
:class: tip
```python
x = 1
```
```
`````

**ច្បាប់ ២៖ កុំដាក់ backtick ក្នុងចំណងជើង admonition។**

MyST វិភាគចំណងជើងជា inline markup ហើយ backtick នឹងធ្វើឱ្យ fence state
machine ខូច។ សរសេរឈ្មោះ API ដោយផ្ទាល់ (គ្មាន code markup)៖

```markdown
✅ ត្រូវ៖  ```{admonition} ប្រើ torch.softmax() ជានិច្ច
❌ ខុស៖   ```{admonition} ប្រើ `torch.softmax()` ជានិច្ច
```

**ច្បាប់ ៣៖ ពេល *បង្ហាញ* fence ជាឧទាហរណ៍ ប្រើ fence វែងជាងមួយកម្រិតទៀត។**

នេះជាច្បាប់ដែលងាយបំពានបំផុត ព្រោះឯកសារអំពី fence ត្រូវតែមាន fence ខ្លួនឯង។
ដូចឧទាហរណ៍ខាងលើ៖ មាតិកាមាន ៤ backticks ដូច្នេះឯកសារខាងក្រៅត្រូវការ **៥**។

| មាតិកាខាងក្នុងមាន | fence ខាងក្រៅត្រូវការ |
| --- | --- |
| ៣ backticks | ៤ |
| ៤ backticks | ៥ |

ជម្រើសដែលសុវត្ថិភាពបំផុតគឺ **indented code block** (ចូល ៤ ចន្លោះ) ព្រោះវាគ្មាន
fence delimiter សោះ ដូច្នេះអាចផ្ទុក backtick ចំនួនណាក៏បាន។

**ពិនិត្យដោយស្វ័យប្រវត្តិមុន commit — ចាំបាច់៖**

```bash
python tools/check_markdown.py            # exit 1 បើមានបញ្ហា (គ្រប់ .md ក្នុង repo)
python tools/check_markdown.py --quiet    # បង្ហាញតែឈ្មោះឯកសារដែលខូច
```

ឧបករណ៍នេះស្កេន **គ្រប់ឯកសារ `.md` ក្នុង repository** — ទាំង `docs/source/`
(ដែល Sphinx/MyST render) និង `README.md` / `CONTRIBUTING.md` (ដែល GitHub render)។
វា **រាយការណ៍តែប៉ុណ្ណោះ មិនកែដោយស្វ័យប្រវត្តិទេ** ព្រោះពេល fence ត្រូវបានកាត់
ឧបករណ៍មិនអាចដឹងថាតើ bare fence មួយណាជាចុងដែលអ្នកនិពន្ធចង់បានទេ —
ការស្មាននឹងសរសេរជាន់បន្ទាត់ខុស ហើយធ្វើឱ្យឯកសារខូចដោយស្ងាត់។

### ៧. គុណភាព Contrast

ពណ៌ថ្មីទាំងអស់ត្រូវឆ្លងកាត់កម្រិត **≥ 4.5:1** (WCAG AA)។
គណនានៅ <https://webaim.org/resources/contrastchecker/>។
Palette ទាំងមូលស្ថិតនៅ `docs/source/_static/css/custom.css` ជា CSS custom properties —
កែតម្លៃ variable មិនមែន hardcode ពណ៌ទេ។

### ៨. សុវត្ថិភាព (Security)

- **កុំ** commit Token, API Key, Password, Cookie, Private Key ឬ Secret ណាមួយ។
- ប្រើ Environment Variables ឬ Credential Manager។
- បើ Secret លេចធ្លាយ សូម Revoke/Rotate ភ្លាមៗ មុនធ្វើអ្វីផ្សេង។
- **កុំ** ប្រើ `git push --force`, `git reset --hard` ឬ `git clean -fdx`
  លើ branch ដែលមានការងាររបស់អ្នកដទៃ។

## ទម្រង់ Commit Message

ប្រើ [Conventional Commits](https://www.conventionalcommits.org/) ជាភាសាអង់គ្លេស ខ្លី និងច្បាស់៖

```text
docs(tensors): add broadcasting examples
fix(ui): table striping contrast in dark mode
feat(autograd): add gradient accumulation lesson
test(examples): make training loop deterministic
```

## Pull Request Checklist

- [ ] កូដឧទាហរណ៍ទាំងអស់ pass ដោយ `python tests/run_examples.py`
- [ ] `sphinx-build -b html docs/source docs/_build/html -W --keep-going` exit 0 (zero warnings)
- [ ] ឈ្មោះ API មិនត្រូវបានបកប្រែ
- [ ] គ្មាន Secret ឬ credential ក្នុង diff
- [ ] បានបន្ថែមទំព័រទៅក្នុង `toctree` នៃ `docs/source/index.md`
- [ ] បានពិនិត្យមើលក្នុង Light Mode **និង** Dark Mode
