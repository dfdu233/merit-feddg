# Observation-blind selection: expanded SLAKE TRAIN confirmation

Completed 128/128 cases from the entire existing native-slake128-confirm-v2
schedule, without reference-based selection or new splits. This is TRAIN
diagnostics, not the full SLAKE test set and not a SOTA evaluation.
The user requested expansion after the negative 8-case diagnostic; the
likelihood rule, candidates, weights, prompts and crop construction were frozen.
Old 8-case results remain untouched. No new medical algorithm was implemented.

## Results

Frozen ANCHOR CLOSED parsing plus OPEN token recall, **not clinical accuracy**:

| Arm | Score | Delta vs Baseline (pp) | Improved / harmed |
|---|---:|---:|---:|
| Huatuo Baseline, reused | 58.6068% | 0 | 0 / 0 |
| Native MERIT, reused | 57.4349% | -1.1719 | 14 / 14 |
| Original-image candidate likelihood ranking | 59.5833% | +0.9766 | 2 / 0 |
| Segmentation-text ranking + unavailable fallback | 58.6068% | 0 | 0 / 0 |
| Native-crop ranking + unavailable fallback | 58.6068% | 0 | 0 / 0 |
| Repeated-full-view control + unavailable fallback | 58.6068% | 0 | 0 / 0 |

Image-cluster paired bootstrap delta for image-only ranking: [0.0000, 2.7344]
percentage points, 128 image clusters. MERIT: [-8.5938, 5.6641] pp. The interval
touching zero, inspected TRAIN cohort and lexical scorer do not establish
generalization, safety, or statistical superiority. Degenerate zero-delta
intervals for fallback arms are not affirmative non-inferiority evidence.

Image-only ranking selected MERIT 5/128 times and Baseline 123/128. All five
MERIT selections were Chinese OPEN questions. Two increased token recall:
one longer answer included the reference anatomical term; another replaced a
specific treatment answer with a longer uncertainty statement yet increased
reference-token overlap (0.25 to 0.50). Neither is established clinical
correction. Three changes had no metric benefit. Six sum-vs-mean decisions
differed; sum remains a diagnostic, not a newly selected strategy.

## Coverage and negative finding

Only 11/128 (8.59%) cases had an applicable native BiomedParse crop under the
unchanged English exact-native-name rule. 64 Chinese questions were outside
that adapter's language scope; 53 English questions lacked a matching usable
native mask. No masks or translated rules were invented.

All 128 cases received image-only likelihood evaluation. The three observation
arms explicitly retained Baseline on 117 unavailable cases, without model
calls; those are not successful evidence decisions. Within the 11 genuinely
observed cases, Baseline and all four rankers scored 54.5455%; MERIT scored
36.3636%, with one gain and three harms. All rankers retained Baseline on all
11. Real crop vs repeated-full-image changed no selection. Equal visual-token
budget comparison applies only to these 11 cases, not the fallback rows.

Thus the larger experiment does not support segmentation-mediated improvement.
Whole-answer likelihood still favors the original self-generated answer and
can depend on wording, length and EOS. The tiny overall gain comes from the
image-only control, not spatial evidence. Do not proceed to a SOTA claim or
silently tune the rule against these outcomes.

## Engineering and cost

- 36 relevant CPU tests passed; git diff check passed. Not a full-library pass.
- 128/128 native single-image token/tensor parity checks passed.
- 128/128 normal/deep-copied inherited operation views matched.
- 322 teacher-forcing calls passed native answer alignment and CE parity.
- 11 actual crops differed from the original after model preprocessing.
- Reused 256 candidate outputs; zero new answer-generation calls.
- New scoring: 32.834 s total (image-only 22.104 s, segmentation text 2.467 s,
  crop 4.134 s, repeated full view 4.128 s). Model load: 18.289 s.
- Peak allocated GPU memory: 19,383,659,008 bytes (18.05 GiB).
- Historical candidate seconds: Baseline 33.697 s; MERIT 133.240 s, reused,
  not zero-cost. Unrecorded historical/preprocessing overhead remains unknown;
  measured scoring plus loading is not end-to-end wall time.
- One idle authorized GPU was used: host GPU1 / container GPU0,
  UUID GPU-3846413a-4238-d307-b1f3-10c2dfbe002c. This small cached experiment
  finished without needing another model replica. tmux protected execution.
- Existing Huatuo Python environment and /home/dbw/models/HuatuoGPT-Vision-7B
  were reused offline. No weights, dependencies or shared model code changed.

## Reproduction and identities

Base code: aba66780cbc32dc1ab5f28c2ee1f7b1343605ba8 plus this commit's
queue-expansion and offline reporting changes. Scientific functions unchanged.
Protocol identity:
`99ce7d059863f5a004c88f2e3eacb607d237197bee42bd88a9329764ccff5507`.
The private protocol records the exact inference-script SHA, source identity,
ordered rows, image and candidate file hashes. Scorer hashes are in the public
aggregate JSON and were checked before scoring.

From /home/dbw/merit-feddg-observation-blind, with existing huatuo Python:

```bash
export OBSERVATION_FULL_QUEUE=1 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/dbw/merit-feddg-observation-blind:/home/dbw/ANCHOR
python scripts/observation_blind_probe.py prepare
python scripts/observation_blind_probe.py run
python scripts/evaluate_observation_blind_probe.py
```

Outputs are exclusive-write; do not rerun prepare over an existing result.
Private artifacts: runs/observation-blind-slake128-v1/{protocol.json,cases/,
run.log,load-run.json,complete.json,evaluation.json,evaluation.log}.
Public aggregate: reports/observation-blind-slake128-v1.json.
Only code, synthetic tests and aggregate/de-identified descriptions are committed.
Patient questions, images, answers and masks remain on the server.

Next scientific issue: disentangle factual support from self-likelihood and
answer-form preference before interpreting a selector as trustworthy. This
run does not implement or validate a claim-level selector, classification
expert utility, general multilingual routing, or a full benchmark comparison.
