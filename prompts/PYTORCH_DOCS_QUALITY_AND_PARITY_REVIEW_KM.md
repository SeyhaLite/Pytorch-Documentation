# Prompt សម្រាប់ AI Agent៖ ពិនិត្យគុណភាព និងភាពត្រឹមត្រូវជាមួយ PyTorch ផ្លូវការ

> **គោលបំណង៖** ប្រើ Prompt នេះឱ្យ AI Agent ពិនិត្យមេរៀនទាំងអស់ក្នុង Repository, ផ្ទៀងផ្ទាត់ API / កូដ / ព័ត៌មានបច្ចេកទេសជាមួយ PyTorch ផ្លូវការ, រកព័ត៌មានចាស់ឬខុស, កែលម្អគុណភាពបង្រៀនជាភាសាខ្មែរ និងបន្ថែមមាតិកាដែលខ្វះដោយមានប្រភពច្បាស់លាស់។

**Repository:** https://github.com/SeyhaLite/Pytorch-Documentation

**ភាសាសម្រាប់ការពន្យល់ និងរបាយការណ៍៖** ខ្មែរ ធម្មតា ងាយយល់។ រក្សាឈ្មោះ API, class, function, parameter, command និង technical identifier ជាភាសាអង់គ្លេសដូចដើម។

---

## ១. តួនាទី និងគោលដៅ

អ្នកគឺជា Senior PyTorch Documentation Reviewer, Python/PyTorch Engineer, Technical Editor និង Khmer Learning-Content Reviewer។

ភារកិច្ចមិនមែនគ្រាន់តែពិនិត្យអក្ខរាវិរុទ្ធទេ។ អ្នកត្រូវផ្ទៀងផ្ទាត់ខ្លឹមសារដោយយោងទៅ PyTorch ផ្លូវការ ហើយឆ្លើយសំណួរទាំងនេះដោយមានភស្តុតាង៖

1. តើការពន្យល់ក្នុងមេរៀនត្រឹមត្រូវតាមគោលការណ៍ និង behavior ពិតរបស់ PyTorch ដែរឬទេ?
2. តើ API, function, class, parameter, default value និង code example នៅតែមានសុពលភាពក្នុង PyTorch version ដែលគម្រោងប្រកាសគាំទ្រដែរឬទេ?
3. តើមាន API ណាដែលត្រូវបាន Deprecated, Removed, Renamed ឬប្ដូរឥរិយាបថ?
4. តើការណែនាំអំពី Python, pip, CPU, CUDA, ROCm, MPS, XPU និង Platform ត្រូវនឹងព័ត៌មានផ្លូវការដែរឬទេ?
5. តើកូដឧទាហរណ៍អាច Run បានពិត និងបង្ហាញលទ្ធផលត្រឹមត្រូវដែរឬទេ?
6. តើមេរៀនមានគុណភាពសម្រាប់អ្នករៀនពីដំបូងដែរឬទេ ឬគ្រាន់តែបង្ហាញពាក្យបច្ចេកទេសនិងកូដដោយគ្មានការពន្យល់?
7. តើមានប្រធានបទសំខាន់ណាខ្វះ ដែលគួរបន្ថែមដោយមិនធ្វើឱ្យមេរៀនស្ទួន ឬបង្កើត Noise?

**កុំសន្មតថាខ្លឹមសារចាស់ត្រឹមត្រូវ។ ក៏កុំសន្មតថាខ្លឹមសារចាស់ខុសទាំងអស់ដែរ។** ត្រូវវិនិច្ឆ័យតាមប្រភពផ្លូវការ, version, test និង context។

## ២. ព័ត៌មាន Version ដែលត្រូវផ្ទៀងផ្ទាត់

Snapshot ដែលបានពិនិត្យនៅថ្ងៃទី **2026-11-13**៖

- របាយការណ៍ `docs/BUILD_REPORT.md` បង្ហាញថា code examples បាន Run លើ `PyTorch 2.14.1+cpu`។
- PyTorch ផ្លូវការបានប្រកាស `PyTorch 2.15.0` General Availability នៅថ្ងៃទី **2026-10-28**។
- PyTorch 2.15 មានការផ្លាស់ប្ដូរក្នុង Release Notes និង Python/platform support។ ឧទាហរណ៍ Release announcement បញ្ជាក់ថា Python 3.10 លែងត្រូវបានគាំទ្រ ហើយ 2.15 ត្រូវការ Python 3.11 ឡើងទៅ។

ប្រភពចាប់ផ្ដើម៖

- PyTorch Stable Documentation: https://docs.pytorch.org/docs/stable/
- PyTorch 2.15 Release Notes: https://github.com/pytorch/pytorch/releases/tag/v2.15.0
- PyTorch 2.15 Release Blog: https://pytorch.org/blog/pytorch-2-15-release-blog/
- PyTorch Installation Selector: https://pytorch.org/get-started/locally/
- PyTorch Previous Versions: https://pytorch.org/get-started/previous-versions/
- PyTorch Release Cadence: https://github.com/pytorch/pytorch/blob/main/RELEASE.md

**ការប្រុងប្រយ័ត្នអំពី Version៖** ការដែល Project Test លើ 2.14.1 ខណៈមាន 2.15.0 មិនបញ្ជាក់ថាមេរៀនទាំងអស់ខុស ឬមិនអាចប្រើបានទេ។ ត្រូវបែងចែកឱ្យច្បាស់រវាង៖

- **Latest stable version**៖ កំណែ Stable ថ្មីបំផុតនៅថ្ងៃពិនិត្យ។
- **Project-tested version**៖ កំណែដែលមានលទ្ធផល Test ពិតក្នុង Build Report ឬ CI។
- **Supported/documented version range**៖ កំណែដែលគម្រោងសម្រេចថាគាំទ្រ និងបានផ្ទៀងផ្ទាត់។

កុំប្ដូរ version, dependency, installation command ឬ API គ្រប់ទីកន្លែងដោយស្វ័យប្រវត្តិ។ ផ្ទៀងផ្ទាត់ផលប៉ះពាល់ជាមុន រួចសម្រេចដោយមានហេតុផល និង Test។

## ៣. ប្រភពដែលត្រូវប្រើ និងលំដាប់ភាពជឿជាក់

ប្រើប្រភពតាមលំដាប់នេះ៖

1. PyTorch versioned documentation ដែលត្រូវនឹង version ដែលកំពុងផ្ទៀងផ្ទាត់។
2. PyTorch Stable Documentation និង API reference បច្ចុប្បន្ន។
3. PyTorch Release Notes និង Official Release Blog សម្រាប់ការផ្លាស់ប្ដូរតាម version។
4. PyTorch Installation Selector និង Official Previous Versions សម្រាប់ installation commands និង hardware builds។
5. PyTorch tutorials ដែលស្ថិតក្នុង Official PyTorch organization។
6. Source code, tests និង docstrings ក្នុង Official `pytorch/pytorch` repository នៅពេល documentation មិនទាន់ច្បាស់។

តំណ Repository ផ្លូវការ៖

- https://github.com/pytorch/pytorch
- https://github.com/pytorch/tutorials
- https://docs.pytorch.org/tutorials/

ច្បាប់ប្រើប្រភព៖

- ចូលមើលទំព័រដើមពេញលេញ ឬផ្នែកដែលពាក់ព័ន្ធ។ កុំសម្រេចដោយផ្អែកតែលើ Search Snippet។
- កត់ត្រា URL, version, section ឬ symbol និងថ្ងៃដែលបានពិនិត្យ។
- ពេលព័ត៌មានខុសគ្នារវាង version ចាស់និងថ្មី ត្រូវបង្ហាញភាពខុសគ្នានោះ មិនមែនយកព័ត៌មានថ្មីទៅអះអាងថា version ចាស់ខុសទាំងអស់ឡើយ។
- កុំយក Blog របស់អ្នកទីបី ឬ Stack Overflow មកជំនួស Official Documentation សម្រាប់ការសម្រេច API correctness។
- បើមិនអាចផ្ទៀងផ្ទាត់ព័ត៌មានណាមួយបាន ត្រូវសម្គាល់ថា **UNVERIFIED** មិនត្រូវស្មាន ឬបង្កើតប្រភព។
- កុំ Copy Documentation ផ្លូវការជាច្រើនកថាខណ្ឌ។ សរសេរពន្យល់ជាពាក្យរបស់គម្រោង និងរក្សា Attribution តាម LICENSE។

## ៤. តំបន់ដែលត្រូវពិនិត្យក្នុង Repository

ចាប់ផ្ដើមដោយអាន README, CONTRIBUTING, BUILD_REPORT, `.readthedocs.yaml`, `docs/requirements.txt`, `docs/source/conf.py`, `docs/source/index.md`, tests និង tools ជាមុន។

បន្ទាប់មកពិនិត្យ **គ្រប់ទំព័រ** ក្នុង `docs/source/` និងកូដពាក់ព័ន្ធក្នុង `docs/source/examples/`៖

- `installation.md`
- `getting-started.md`
- `tensors.md`
- `shapes-and-broadcasting.md`
- `devices.md`
- `autograd.md`
- `neural-networks.md`
- `loss-and-optimizers.md`
- `data-and-dataloader.md`
- `training-loop.md`
- `evaluation-and-inference.md`
- `saving-and-loading.md`
- `checkpoints.md`
- `reproducibility.md`
- `performance.md`
- `mixed-precision.md`
- `distributed-training.md`
- `debugging.md`
- `api-quick-reference.md`
- `style-guide.md`
- `license.md`

ក៏ត្រូវពិនិត្យ README, index/toctree, Build Report និង version/dependency/configuration files ដើម្បីរកភាពមិនស៊ីគ្នារវាងឯកសារ។

កុំរំលងឯកសារដែលមានតែពាក្យពីរបី ឬគ្មាន code example។ ត្រូវកំណត់ថាវាមានតួនាទីអ្វី និងតើត្រូវការកែលម្អឬអត់។

## ៥. Checklist ផ្ទៀងផ្ទាត់ PyTorch Correctness

ពិនិត្យតាមប្រធានបទខាងក្រោម ហើយបន្ថែមចំណុចថ្មីនៅពេល Official Docs មានការផ្លាស់ប្ដូរ។

### ៥.១ Installation និង Environment

- Python version support តាម PyTorch release។
- pip/conda command និង CPU/CUDA/ROCm/MPS/XPU wheel selector។
- OS, architecture និង hardware compatibility។
- កុំលាយ command សម្រាប់ CPU, CUDA, ROCm ឬ platform ផ្សេងគ្នា។
- កុំប្រើ CUDA version, wheel index URL ឬ Python version ដែលមិនមានការផ្ទៀងផ្ទាត់។
- បញ្ជាក់ថា PyTorch ត្រូវការ ឬមិនត្រូវការនៅក្នុង Sphinx/Read the Docs build។
- បំបែក dependencies សម្រាប់ Build ឯកសារ និង dependencies សម្រាប់ Run examples។

### ៥.២ Tensor និង Operations

- `torch.tensor`, `torch.empty`, `torch.zeros`, `torch.ones`, `torch.arange`, `torch.rand` និង factory functions។
- `dtype`, `device`, `requires_grad`, `shape`, `ndim`, `stride`, `layout` និង memory behavior។
- Indexing, slicing, views, reshape, transpose, contiguous និង in-place operations។
- Broadcasting rules និង edge cases។
- Device/dtype mismatch, integer division, type promotion និង precision។
- កុំអះអាងថា `view()` និង `reshape()` មាន behavior ដូចគ្នាគ្រប់ករណីទេ។

### ៥.៣ Autograd

- Computational graph និង gradient tracking។
- `requires_grad`, `backward()`, `.grad`, `torch.autograd.grad`។
- Gradient accumulation និងការសម្អាត gradient។
- `detach()`, `no_grad()`, `inference_mode()` និងការប្រើប្រាស់ខុសគ្នា។
- Non-leaf tensor និង graph lifetime។
- អ្វីដែល in-place operation អាច ឬមិនអាចធ្វើបាននៅពេល track gradient។

### ៥.៤ Neural Networks, Loss និង Optimizers

- `nn.Module`, `Parameter`, `forward()` និង module registration។
- Layers, activations, initialization និង model mode។
- `train()` និង `eval()` មិនត្រូវពន្យល់ថាជា gradient on/off ដោយស្វ័យប្រវត្តិទេ។
- Loss input/target shape, dtype និង expected range។
- `CrossEntropyLoss`, `BCEWithLogitsLoss`, `MSELoss` និង reduction behavior។
- SGD, Adam, AdamW, learning rate schedulers និង optimizer state។
- `zero_grad(set_to_none=...)`, backward order និង gradient accumulation។
- Train/eval loop ត្រូវបង្ហាញ data flow ត្រឹមត្រូវ។

### ៥.៥ Dataset, DataLoader និង Training

- `Dataset`, `IterableDataset`, `DataLoader`, `Sampler`, `collate_fn`។
- `batch_size`, `shuffle`, `drop_last`, `num_workers`, `pin_memory`, `persistent_workers` និង version/platform constraints។
- Randomness និង worker seeding។
- Shape/dtype របស់ batch។
- Checkpoint/resume flow និង optimizer/scheduler/RNG state។
- កុំអះអាង reproducibility 100% ដោយមិនបញ្ជាក់ hardware/backend/version limitations។

### ៥.៦ Evaluation, Saving និង Loading

- `torch.no_grad()`, `torch.inference_mode()` និង model evaluation។
- Metrics, prediction, batching និង post-processing។
- `state_dict`, `load_state_dict`, `torch.save`, `torch.load`។
- Security / `weights_only` / serialization behavior តាម version។
- TorchScript, `torch.export`, ONNX និង safetensors ត្រូវសម្គាល់ភាពខុសគ្នា និងស្ថានភាព support; កុំចាត់ទុកថាជាវិធីតែមួយ ឬអាចជំនួសគ្នាទាំងស្រុង។
- កុំបង្ហាញទម្រង់ save/load ដែលលែងណែនាំ ដោយមិនពន្យល់ limitation។

### ៥.៧ Performance, Mixed Precision និង Distributed

- `torch.compile`, profiler, benchmark methodology និង warm-up/synchronization។
- CPU/GPU timing ត្រូវគោរព CUDA asynchronous behavior។
- AMP APIs និង `torch.amp` / `GradScaler` តាម version។
- `float16`, `bfloat16`, numerical stability និង hardware support។
- DDP, DistributedSampler, `torchrun`, process rank/world size និង cleanup។
- FSDP / FSDP2, checkpointing និង API stability status។
- បែងចែក Stable, Beta/Preview, Prototype ឬ API-Unstable តាម label ផ្លូវការដែលមាននៅក្នុង release នោះ។
- កុំអះអាងថា GPU/multi-node code បាន Test ប្រសិនបើ environment មិនមាន hardware ឬ cluster នោះ។

### ៥.៨ API Reference និង Common Errors

- API ឈ្មោះ, signature, parameter, return value និង behavior។
- Deprecated/removed API និង replacement។
- Exception name/message ត្រូវស្របនឹង code និង version ដែលបាន Run។
- កុំសរសេរ error output ដោយស្មានថាជា actual traceback។
- បង្ហាញមូលហេតុ, វិធីសម្គាល់ និង fix ដែលអាចដំណើរការបាន។

## ៦. Checklist គុណភាពមេរៀន និង Khmer Learning Experience

មេរៀនមួយមានគុណភាព មិនមែនគ្រាន់តែមាន code ឬបញ្ជី API ទេ។ ពិនិត្យចំណុចទាំងនេះ៖

1. **Learning objective**៖ ប្រាប់ច្បាស់ថាចប់មេរៀននេះអ្នករៀនអាចធ្វើអ្វីបាន។
2. **Prerequisites**៖ បញ្ជាក់ចំណេះដឹងដែលត្រូវមាន មិនលាក់ចំណុចសំខាន់។
3. **Simple explanation**៖ ពន្យល់ជាភាសាខ្មែរសាមញ្ញ មុនបញ្ចូលពាក្យបច្ចេកទេសច្រើន។
4. **Concept and intuition**៖ បង្ហាញគំនិត និងហេតុផល មិនត្រឹមតែប្រាប់ syntax។
5. **Step-by-step example**៖ ដើរតាម code line-by-line និងពន្យល់ input → process → output។
6. **Real output**៖ បង្ហាញ output ដែលអាចបង្កើតឡើងវិញបាន។
7. **Common mistakes**៖ បង្ហាញកំហុសដែលសមស្របនឹងមេរៀន និងវិធីកែ។
8. **Practice**៖ មានលំហាត់តូចៗដែលវាស់ការយល់ដឹង មិនមែនសំណួរទូទៅប៉ុណ្ណោះ។
9. **Next step**៖ មានតំណទៅមេរៀនមុន/បន្ទាប់ និងរក្សា learning order។
10. **Terminology**៖ ប្រើពាក្យខ្មែរដូចគ្នាទាំងគម្រោង ហើយបង្ហាញ English term ក្នុងវង់ក្រចកលើកដំបូង។
11. **No filler / no noise**៖ ដកប្រយោគស្ទួន, emoji មិនចាំបាច់, intro វែងដោយគ្មានព័ត៌មាន និងការអះអាងផ្សព្វផ្សាយដែលគ្មានប្រយោជន៍។
12. **Accessibility**៖ ចំណងជើងមាន hierarchy ត្រឹមត្រូវ, table/code អាចអានបាន និង UI ប្រើបានទាំង light/dark។

កុំធ្វើឱ្យមេរៀនវែងដោយគ្រាន់តែបន្ថែមអត្ថបទ។ បន្ថែមតែអ្វីដែលជួយឱ្យយល់ និងអាចអនុវត្តបាន។

## ៧. របៀបវិនិច្ឆ័យ Finding

**ប្រភេទ**

- `PYTORCH_MISMATCH`៖ មិនត្រូវនឹង official behavior/API។
- `OUTDATED`៖ ត្រឹមត្រូវសម្រាប់ version ចាស់ ប៉ុន្តែមិនត្រឹមត្រូវ ឬមិនពេញលេញសម្រាប់ target version។
- `TECHNICAL_ERROR`៖ គោលការណ៍ ឬគណនាបច្ចេកទេសខុស។
- `CODE_BUG`៖ code example មិន compile/run ឬ output ខុស។
- `VERSION_GAP`៖ មិនបានប្រកាស version support ឬការសាកល្បងមិនគ្រប់។
- `MISSING_CONTENT`៖ ខ្វះប្រធានបទចាំបាច់។
- `TEACHING_QUALITY`៖ ពន្យល់មិនច្បាស់ ឬរៀបចំការរៀនមិនល្អ។
- `KHMER_LANGUAGE`៖ ពាក្យ/វេយ្យាករណ៍មិនធម្មជាតិ ឬពាក្យបច្ចេកទេសមិនស្របគ្នា។
- `DOCS_BUILD`៖ Markdown, MyST, Sphinx, link ឬ toctree issue។
- `MAINTENANCE`៖ Version, dependency, report ឬ configuration មិនស៊ីគ្នា។

**កម្រិតធ្ងន់ធ្ងរ**

- `CRITICAL`៖ អាចបង្កគ្រោះថ្នាក់ដល់ទិន្នន័យ/សុវត្ថិភាព ឬបង្រៀន API ដែលបង្ក failure ធ្ងន់។
- `HIGH`៖ API/គោលការណ៍/installation instruction សំខាន់ខុស ឬ obsolete។
- `MEDIUM`៖ ខ្លឹមសារមិនពេញលេញ, example មិន robust ឬ teaching flow ខ្សោយ។
- `LOW`៖ អក្ខរាវិរុទ្ធ, style, link text ឬ polish តូចៗ។

កុំចាត់ទុកការខ្វះ content ដែលមិនមែនជា project scope ជា bug។ បញ្ជាក់ impact និងមូលហេតុជានិច្ច។

## ៨. វិធីធ្វើការជាដំណាក់កាល

### ដំណាក់កាល A — Inspect មុនកែ

- ពិនិត្យ branch, working tree និងស្ថានភាព Git។
- អាន README, CONTRIBUTING, BUILD_REPORT, requirements, Sphinx config, RTD config, index/toctree, tests និង tools។
- រាយបញ្ជីគ្រប់ source pages និង example files។
- រក្សា changes ដែលមានស្រាប់។ កុំសរសេរជាន់ uncommitted work។
- បង្កើត branch ថ្មីសម្រាប់ audit និង fix។

### ដំណាក់កាល B — កំណត់ Version Policy

- ផ្ទៀងផ្ទាត់ latest stable version ពី Official PyTorch release source។
- ផ្ទៀងផ្ទាត់ project-tested version ពី test report/log ពិត។
- កំណត់ target support range ដោយពិនិត្យ Python, OS, hardware និង project dependencies។
- បើ version policy មិនមាន សរសេរជា finding មួយ និងស្នើ policy ដែលអាចអនុវត្តបាន។
- កុំប្ដូរ Python/torch dependencies ដោយមិន run tests និង Sphinx build។
- កុំច្រឡំ Sphinx Python environment ជាមួយ PyTorch runtime environment។

### ដំណាក់កាល C — Audit រាល់ទំព័រ

សម្រាប់គ្រប់ទំព័រ៖

- អានខ្លឹមសារទាំងមូល។
- ស្រង់ claims ដែលអាចផ្ទៀងផ្ទាត់បាន៖ version, API, parameter, behavior, output, platform, performance, installation។
- ប្រៀបធៀបជាមួយ Official Docs របស់ target version។
- ពិនិត្យ code example ដែលទំព័រនោះយោងទៅ។
- ពិនិត្យថាមាន test/assertion គាំទ្រលទ្ធផលដែរឬទេ។
- ពិនិត្យ cross-reference និង toctree។
- កត់ត្រា result របស់ទំព័រនោះ មុនចាប់ផ្ដើមទំព័របន្ទាប់។

### ដំណាក់កាល D — កែលម្អ និងបន្ថែម

- កែ factual errors និង outdated API ដោយប្រើ official reference។
- កែលម្អពន្យល់ឱ្យអ្នកចាប់ផ្ដើមអាចយល់។
- បន្ថែម example ឬលំហាត់តែពេលមាន learning gap ពិត។
- ដាក់ example source ក្នុង `docs/source/examples/` ហើយបញ្ចូលដោយ MyST `literalinclude`។ កុំ copy/paste example ទៅក្នុង Markdown ដាច់ដោយឡែក។
- បន្ថែម assertions និង test សម្រាប់ code behavior។
- បើមាន API-version difference ត្រូវបង្ហាញ note ឬ version scope ច្បាស់។
- Update README, index, BUILD_REPORT, dependency/version notes និង compatibility report នៅពេលខ្លឹមសារពាក់ព័ន្ធបានប្ដូរ។
- បើមិនអាច Test hardware ពិតបាន កត់ត្រា `NOT RUN` ឬ `SKIPPED` និងមូលហេតុ។

### ដំណាក់កាល E — Tests និង Build

ស្វែងយល់ពី test commands ដែល Repository កំណត់ ហើយប្រើជាពិសេស៖

```bash
python tests/run_examples.py
python tests/check_examples_referenced.py
python tools/check_markdown.py
python tools/check_contrast.py
python tests/check_locale.py
python tests/check_ui_strings.py
python tests/check_search_console.py
python tests/check_sitemap.py
python -m sphinx -b html docs/source docs/_build/html -q -W --keep-going
bash tools/verify.sh
bash tools/verify_clean_clone.sh
```

បន្ថែមពិនិត្យ JavaScript និង YAML តាម scripts ដែលមានក្នុង Repository។ កុំរត់ពាក្យបញ្ជាដែលបំផ្លាញ data ឬ `rm -rf` ក្រៅពី output/build directory ដែល script បានកំណត់ និងបានពិនិត្យ។

តេស្តត្រូវបែងចែកតាមស្ថានភាពពិត៖

- `PASS`៖ command បានរត់ និង exit 0។
- `FAIL`៖ command បានរត់ ហើយមាន failure។
- `SKIPPED`៖ test ត្រូវបានរំលងដោយលក្ខខណ្ឌដែលកត់ត្រាច្បាស់។
- `NOT RUN`៖ មិនបានដំណើរការ។
- `BLOCKED`៖ មិនអាចដំណើរការដោយសារ dependency, permission, network ឬ hardware។

កុំរាប់ `SKIPPED` ជា `PASS`។ កុំ claim CUDA, ROCm, MPS, XPU ឬ distributed test បានជោគជ័យ ប្រសិនបើគ្មាន hardware/runtime ពិត។

### ដំណាក់កាល F — Review ម្ដងទៀត

ក្រោយកែ៖

- អាន Diff ទាំងមូល។
- ពិនិត្យថាមិនមាន unrelated changes។
- ពិនិត្យ link និង code fences។
- ប្រៀបធៀប version info នៅ README, index, BUILD_REPORT, requirements និង RTD config ឱ្យស៊ីគ្នា។
- ពិនិត្យ Search Console ownership file នៅ `docs/source/extra/` និង test របស់វា។ **កុំលុប ឬកែ token/file ដោយមិនមានហេតុផល និងការអនុញ្ញាត**។
- ពិនិត្យ sitemap extension/test ដោយមិនបំផ្លាញ behavior ដែលមានស្រាប់។
- កុំបង្កើត badge, version number, coverage percentage ឬ test result ដែលមិនមាន evidence។

### ដំណាក់កាល G — Commit និង Pull Request

- បង្កើត commits តូចៗ និងមានន័យ ដោយប្រើ Conventional Commits។
- បង្កើត Pull Request ជាភាសាខ្មែរ ឬមានសេចក្ដីសង្ខេបខ្មែរច្បាស់។
- PR ត្រូវមាន scope, findings, official references, files changed និង test matrix។
- បង្ហាញបញ្ហាដែលនៅសល់ និង limitation ដោយមិនលាក់។
- កុំ Merge ឬលុប branch ដោយស្វ័យប្រវត្តិ លុះត្រាតែអ្នកប្រើស្នើឱ្យធ្វើ។
- កុំបញ្ចូលការកែប្រែដែល Test មិនឆ្លង ដោយអះអាងថា ready។

## ៩. ទម្រង់របាយការណ៍ដែលត្រូវបង្កើត

បង្កើត Audit Report ក្នុង Repository ឧទាហរណ៍៖

`docs/reports/PYTORCH_OFFICIAL_PARITY_AUDIT.md`

បើថត `docs/reports/` មិនទាន់មាន អាចបង្កើតបាន។ កុំសរសេរជាន់ report ចាស់ដោយបាត់ប្រវត្តិ។ របាយការណ៍ត្រូវមាន៖

### A. Executive Summary

- ថ្ងៃ Audit
- Latest stable PyTorch version ដែលផ្ទៀងផ្ទាត់
- Project-tested version(s)
- Target support policy
- ចំនួន pages/examples ដែលបានពិនិត្យ
- ចំនួន findings តាម severity
- លទ្ធផល overall និង limitation

### B. Page-by-page Audit Matrix

| File / Section | Claim ឬ API | Official source + Version | Result | Severity | Action |
| --- | --- | --- | --- | --- | --- |

Result ត្រូវប្រើ `MATCH`, `MISMATCH`, `OUTDATED`, `MISSING`, `UNVERIFIED` ឬ `NOT_APPLICABLE`។

### C. Version Compatibility Table

| Component | Latest official | Project tested | Project target | Evidence / Gap |
| --- | --- | --- | --- | --- |
| PyTorch | | | | |
| Python | | | | |
| CUDA / ROCm / MPS / XPU | | | | |
| Sphinx / theme / MyST | | | | |

កុំបំពេញលេខដោយស្មាន។ ប្រើ `Unknown` / `Not tested` នៅពេលខ្វះ evidence។

### D. Findings

រាល់ finding ត្រូវមាន៖

- ID
- Severity និង Category
- File និង line/section
- អ្វីដែលសរសេរបច្ចុប្បន្ន
- អ្វីដែល Official PyTorch កំណត់
- URL និង version/section
- ផលប៉ះពាល់
- ការកែប្រែដែលបានធ្វើ ឬមូលហេតុដែលមិនទាន់កែ
- Test ដែលបញ្ជាក់

### E. Test Results

| Command / Environment | Result | Evidence | Note |
| --- | --- | --- | --- |

### F. Remaining Risks

រាយ limitation ដែលនៅសល់ ដូចជា GPU ឬ multi-node មិនមាន, API មិនទាន់អាចផ្ទៀងផ្ទាត់, version dependency conflict និង RTD deployment ដែលមិនបានពិនិត្យ។

## ១០. លក្ខខណ្ឌទទួលយក (Acceptance Criteria)

ការងារមិនអាចសម្គាល់ថាចប់ដោយសារបានសរសេរឯកសាររួចប៉ុណ្ណោះទេ។ ត្រូវបំពេញលក្ខខណ្ឌទាំងនេះ៖

- [ ] បានអាន និង Audit គ្រប់ទំព័រដែលបានកំណត់។
- [ ] រាល់ API/technical claims សំខាន់ៗមាន Official PyTorch source និង version scope។
- [ ] បានរកឃើញ និងកត់ត្រា outdated / removed / deprecated API។
- [ ] Code examples ត្រូវបានផ្ទៀងផ្ទាត់ដោយ Test ឬសម្គាល់ច្បាស់ថាមិនទាន់ Test។
- [ ] Installation និង Python/hardware compatibility មិនមានការអះអាងដោយគ្មានប្រភព។
- [ ] Khmer explanation ងាយយល់ និងពាក្យបច្ចេកទេសប្រើស្របគ្នា។
- [ ] បានបន្ថែមមាតិកាដែលខ្វះតែពេលមានហេតុផល និងប្រភពច្បាស់។
- [ ] README/index/version report/requirements មានព័ត៌មានស៊ីគ្នា។
- [ ] Markdown, reference, Sphinx និង project verification ត្រូវបានរត់ ឬកត់ត្រា limitation។
- [ ] មាន Audit Report ដែលអាចពិនិត្យឡើងវិញបាន។
- [ ] PR មាន evidence, test status និង clear limitations។
- [ ] គ្មាន secret, credential, ឬ unrelated destructive changes ក្នុង Diff។

## ១១. របៀបចាប់ផ្ដើម (Copy & Run)

ប្រើ Prompt ខាងក្រោមជាសេចក្ដីបញ្ជាផ្ទាល់ទៅ AI Agent៖

> **ធ្វើ Full Audit and Fix សម្រាប់ Repository នេះ ដោយប្រើគោលការណ៍ទាំងអស់ក្នុង `prompts/PYTORCH_DOCS_QUALITY_AND_PARITY_REVIEW_KM.md`។**
>
> ចាប់ផ្ដើមពី Inspect repository និង target PyTorch version។ ពិនិត្យគ្រប់មេរៀន និងគ្រប់ code example ម្ដងមួយៗ។ ផ្ទៀងផ្ទាត់ technical claims ជាមួយ Official PyTorch Documentation, API reference និង Release Notes តាម version។ រក API ចាស់, API ខុស, behavior ដែលពន្យល់ខុស, installation/version mismatch និង missing learning content។
>
> បន្ទាប់មកកែបញ្ហាដែលមាន evidence បន្ថែម example/test តាមតម្រូវការ កែលម្អការពន្យល់ខ្មែរឱ្យច្បាស់ និងបង្កើត `docs/reports/PYTORCH_OFFICIAL_PARITY_AUDIT.md`។ រក្សា feature, data, Search Console verification, sitemap និង test protections ដែលមានស្រាប់។ កុំបង្កើត result ក្លែងក្លាយ កុំ upgrade dependency ដោយមិន Test និងកុំលុបអ្វីដែលមិនចាំបាច់។
>
> ចុងក្រោយ run project verification, review Diff, រៀបចំ Pull Request និងរាយការណ៍ជាភាសាខ្មែរ។ បញ្ជាក់ `PASS / FAIL / SKIPPED / NOT RUN / BLOCKED` តាមលទ្ធផលពិត។ កុំ Merge ដោយស្វ័យប្រវត្តិ។

## ១២. សេចក្ដីរំលឹកសំខាន់

**គោលដៅគឺឯកសារដែលត្រឹមត្រូវតាម PyTorch ផ្លូវការ និងអាចបង្រៀនអ្នកចាប់ផ្ដើមបានពិតប្រាកដ—not merely a longer README.**

កុំបកប្រែ API, កុំបង្កើត feature/API ដែលមិនមាន, កុំអះអាង test ដែលមិនបាន Run និងកុំចម្លង documentation ផ្លូវការដោយមិនគោរព Attribution។
