# Channel expansion: resource audit and reference implementations

Status 2026-09-14: segmentation full run is active on two physical GPUs.
Other-channel and MIMIC runs are separate experiments, not results of that run.
No training, calibration, test-selected strengths, or cross-case adaptation.

## Actual scheduling and validation

MIMIC two-report canary is running on host GPU0 via `merit-runner`, session
`mimic-soft-canary`. It imports the unchanged spatial `full_case`, preserves
the historical 256-token generation limit, checks both historical shard
configurations and all 694 cache-file IDs against the full manifest, and loads
no references. It stops after the first two manifest rows, does not certify
the missing historical root completion marker, and never auto-expands.
Root: `runs/mimic-report-soft-canary/889ed16b82d7197312e0dfd72bc000a66ded75a69281b0a9b60ab61cff40319d`.
Log: `runs/soft-full-checks/mimic-report-canary.log`.

`run_channel_soft_canary.py` implements .5 convex guidance separately for
classification-conditioned, generated-observation and real retrieved-text
contexts. It preserves nonselected expert evidence, refuses to call omitted
or displaced context successful guidance, and checks current-context and
zero-control token parity. The classification path is text-mediated, NOT
native image-classifier logit alignment. CAD is a reference/future separately
labelled arm, not implemented by this convex interpolation.

Six fixed scheduling cases are selected solely from manifest order and
available capability, not scores: first classification and generation cache
per VQA dataset, and first manifest row for public-text retrieval. The seed
contains six real sourced paragraphs and is fingerprinted. Both no-channel
and ordinary text controls accompany the soft candidate. There is no promise
that a channel will be relevant or helpful on these cases.

CPU validation: nine targeted tests passed and Ruff passed. Both new CLIs
passed real-resource preflight. Channel session `channel-soft-canary` on the
host waits on the running MIMIC canary's existing lock before loading a model;
it is QUEUED, not completed. Root:
`runs/channel-soft-canary/c4d3055ff05209352975d69876c31d36646d56b2b511d2f3c74408919ffcd999`.
Log: `runs/soft-full-checks/channel-canary.log`.

The exploratory slot shares GPU0 with the full VQA/SLAKE shard (approximately
two model copies fit the 49-GiB card). Record contention explicitly; these
timings are not isolated latency measurements. The full run's own measured
runtime may likewise be affected from this point onward. Inference policies
and model versions remain unchanged. No full channel/report scores exist yet.

## Real cached resources (not effective delivery)

| Dataset | Cached cases | Segmentation | Classification | Generated observation | Retrieval |
|---|---:|---:|---:|---:|---:|
| VQA-RAD | 451 | 227 | 6 | 174 | 0 |
| SLAKE | 2094 | 852 | 60 | 740 | 0 |
| MIMIC existing report subset | 694 | 694 | 0 | 16 | 0 |

Classification records are `biomed_anatomy` catalog similarity evidence, not
calibrated disease probabilities. MIMIC masks are principally `cxr_anatomy`,
not lesion ground truth. There is no cached retrieval result to reuse. The
existing public-text seed has only six paragraphs; retrieval from it is a
mechanism test, not a comprehensive medical RAG system. Its explicit general
knowledge scope must not become a current-patient finding.

The MIMIC cache has 694 compact case files and two shard protocol files, but
no root protocol completion marker. A file count alone does not certify an
aligned complete baseline. Validate full IDs and protocols before full reuse.
The user confirmed the task remains report generation with the original
manifest, not a newly synthesized VQA task.

## Literature and official code inspected

- [CAD, NAACL 2024](https://aclanthology.org/2024.naacl-short.69/),
  [official implementation](https://github.com/xhan77/context-aware-decoding/blob/main/group_decode_fileio.py):
  signed weighted logits from context-aware and context-free branches, sharing
  the emitted prefix. Contrastive amplification is not the current convex
  interpolation. Evaluate it as a separate frozen arm, never relabel it as our
  invention. Official code also retains per-branch KV state; porting that idea
  still requires exact local zero/control parity before speed claims.
- [VCD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html),
  [official decoding code](https://github.com/DAMO-NLP-SG/VCD/blob/master/vcd_utils/vcd_sample.py):
  visual counterfactual contrast and plausibility filtering are relevant
  controls. A predicted anatomy mask is not the paper's distorted image, and
  VCD results are not evidence for medical expert-mask utility.
- [PPLM, ICLR 2020](https://arxiv.org/abs/1912.02164),
  [official code](https://github.com/uber-research/PPLM/blob/master/run_pplm.py):
  attribute-guided hidden-state perturbations require a compatible attribute
  interface. An image classifier cannot directly substitute for its language
  discriminator; no such equivalence or completed reproduction is claimed.
- [FUDGE, NAACL 2021](https://aclanthology.org/2021.naacl-main.276/),
  [official code](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation):
  the future discriminator is learned on partial text. Training it violates
  this project's training-free constraint, so it is a conceptual reference,
  not an enabled baseline or a reason to add a trained gate.

## Separate channel experiments

Keep the original image, prompt contract, cached expert outputs, generation
budget and nonintervened channels fixed. Compare no selected channel, ordinary
text delivery, convex soft guidance, and separately labelled CAD guidance.
Segmentation additionally compares native spatial guidance and deletion-only.
Classification-conditioned language distributions must be labelled as such:
they are not native image-classifier-to-token logit transport. Do not invent
a probability calibration or boost arbitrary label subwords.

Count actual delivery, candidate production, token changes, old gains retained,
old harms recovered, new harms, and complete per-branch cost. Empty/omitted
channels are unavailable, not successful zero-call guidance. Retrieval needs
identical real retrieved text on hard/soft arms, with source audit preserved.
No outcome-based choice of cases, strength, source library, or stop rules.

For MIMIC, the existing final evaluator is
`anchor/medeval/evaluate_report_generation_final.py`, version
`medical-report-three-metrics-v2-source-audited`. It reports BLEU-4, ROUGE-L-F1
and METEOR with clustered bootstrap. Reuse it for aligned comparisons, but
these are textual overlap metrics, NOT clinical factual accuracy. Additional
fact checks, if available and authorized, must be reported separately. Do not
apply the VQA scorer to reports. References remain offline-only.
