# Native MERIT: additive expert pool

This experiment changes the expert registry, not original MERIT's acquisition,
acceptance, evidence rendering, or decoding algorithm. The former capability-set
coverage experiment is a separate preserved run and is not this intervention.

## Actual new resources

Three frozen image/text models expose eight fixed-catalog interfaces. These are
relative image/text scores, not eight independent classifiers or calibrated
medical probabilities. All checkpoints are pinned and publisher-hash checked.
Existing dependencies, official preprocessing and source code are reused.

| Backbone | Interface | Current catalog |
|---|---|---|
| UniMed-CLIP | Lung histopathology | Benign, adenocarcinoma, squamous carcinoma |
| UniMed-CLIP | Colon histopathology | Benign tissue, adenocarcinoma |
| UniMed-CLIP | Breast histopathology | Normal, benign, in-situ, invasive |
| UniMed-CLIP | CT anatomy | 11 named organs/structures |
| UniMed-CLIP | Knee MRI | Normal versus meniscal abnormality prompts |
| UniMed-CLIP | Breast ultrasound | Benign/malignant prompts |
| FLAIR | Fundus photography | Normal, macular edema, diabetic retinopathy, glaucoma, macular hole |
| MONET | Skin photography | Bullae, blister, reference concepts |

No new segmentation or free-text generative specialist is added in this round.
Existing specialists remain untouched. Input compatibility is not demonstrated
diagnostic competence: this does not establish broad coverage of all organs,
diseases, acquisition protocols, or medical tasks. Volumetric CT/MRI, video/echo,
specialized OCT/endoscopy, cytology, hematology and many pathology subtypes remain
unvalidated or unsupported. A generalist fallback does not count as expert coverage.

UniMed and Quilt have overlapping training sources; multiple responses are not
independent votes. LC25000/BACH names identify prompt catalogs, not proof of
training-data independence or absence of benchmark leakage. Training overlap
requires separate auditing before broader scientific claims.

## Execution and checks

- Full original PathVQA 6719 IDs, two authorized host GPUs, unchanged split.
- Original `CapabilityRuntime.run('all_evidence')`; retain old registration order
  and append applicable new contracts. No set-cover ranking or old evidence pruning.
- Original call/decision budget and per-case input/output limits retained.
- Exact request-matched native result reuse; unchanged cases reuse full incumbent.
- Canary and affected-case assertions check original calls, requests, execution,
  adoption, raw evidence and final delivery. Canary also reproduces the original
  final answer text and token IDs with new experts disabled.
- Both GPU canaries passed. The breast case keeps CONCH, BiomedParse and Quilt,
  then adds UniMed. The lung case keeps BiomedParse and Quilt, then adds UniMed.
- CPU regression: 1026 passed in 14.39 seconds. No shared dependency upgrades.
- Local root `runs/native-merit-expanded-v2`; preparation-only v1 is retained,
  not mixed into v2. Both full lanes complete: 25 newly applicable cases executed
  with new evidence delivered and old evidence preserved; other 6694 unchanged
  cases reuse the exact complete native MERIT+Quilt incumbent.
- Fresh full-run runtime calls total 52.989 seconds across GPUs, excluding new
  model loading/prefetch, canaries, old expert cost and offline scoring. The output
  file time span is 253.505 seconds, largely cache validation/materialization.
  Reused expert cost is inherited, not zero-cost medical inference.

Offline evaluation rescales nothing: both final-text arms are scored together by
the same frozen ANCHOR scorer. Current scorer source is pinned; loaded dependencies
must also match the historical scorer. The separate changed quality-check file is
recorded, not silently used to relax a historical identity check. A streaming read
avoids retaining dense segmentation masks during text scoring; raw files remain
unaltered. The first memory-heavy scoring attempt was stopped; the streaming retry
completed with the same prediction files and scoring functions.

## Completed formal results

Full 6719 cases, ANCHOR `medheval-decoded-eval-v13-explanatory-uncertainty-review`.
Both arms were scored together, and every loaded historical scorer dependency
matched its frozen hash. Main result is a mixed VQA score, not pure accuracy.

| Method | Mixed VQA score | CLOSED accuracy (3362) | OPEN token recall (3357) |
|---|---:|---:|---:|
| Native MERIT + Quilt, cached control | 31.6562% | 55.6514% | 7.6253% |
| Same native MERIT + appended experts | 31.6711% | 55.6811% | 7.6253% |

- Full mixed-score delta: +0.014883 percentage points; improved1, harmed0,
  text-changed12. These are scorer outcomes, not clinical safety guarantees.
- 25/6719 (0.3721%) cases received new evidence; all25 preserved original evidence.
  Other6694 exact outputs were reused. The 25-case score delta is +4 percentage
  points, but this applicability-conditioned subset is not a separate benchmark.
- Paired image-cluster bootstrap (858 images, 2000 resamples): delta95% interval
  [0, +0.045767] percentage points. Weak/local evidence, not a convincing general
  improvement or proof of broad expert coverage.
- Current PathVQA run evaluates only the three new pathology catalog interfaces;
  it supplies no efficacy evidence for the new fundus/skin/MRI/ultrasound interfaces.
- Raw local report: `runs/native-merit-expanded-v2/evaluation-main.json`; predictions
  and traces in `cases/`, expert requests/native outputs in `prefetch/`. Do not
  commit restricted raw images or dense native case data to GitHub.
