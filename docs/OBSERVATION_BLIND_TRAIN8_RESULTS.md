# Observation-mediated Huatuo selection: first real TRAIN diagnostic

## Outcome

Completed 8/8 fixed SLAKE TRAIN-image cases. All four ranking conditions retain
the Baseline answer. This is an engineering success but **not** successful
evidence selection; one available MERIT correction was missed. Stop expansion.
Do not claim segmentation is useless: crops change likelihoods, but this
full-answer ranking does not turn that change into better decisions.

| Arm | Frozen mixed score | Gains / harms vs Baseline |
|---|---:|---:|
| Reused Huatuo Baseline | 37.5% | 0 / 0 |
| Reused native MERIT | 25.0% | 1 / 2 |
| Original-image likelihood ranking | 37.5% | 0 / 0 |
| Selected segmentation text ranking | 37.5% | 0 / 0 |
| Original image + native crop ranking | 37.5% | 0 / 0 |
| Original image + repeated-full-view ranking | 37.5% | 0 / 0 |

These are 8 previously examined TRAIN images, selected without references by
existing schedule order, English question, explicit native structure mention,
and available nonempty/non-full crop. Not a held-out sample, full benchmark,
population estimate or clinical accuracy. Original candidate pool, canonical
questions/prompts, frozen weights and offline ANCHOR scorer were reused.
Zero paired score differences give degenerate bootstrap intervals; they are
not evidence of prospective non-inferiority or safety.

## Exactly what was implemented

One isolated experiment entry, no new framework or learned model. BiomedParse
original-coordinate masks for explicitly named structures become an operation
view; no declaration that the queried object exists. Binary mask uses the
original 0.5 postprocessing convention, not a new medical admission threshold.
Union bbox with fixed 10% context padding gives a crop from original pixels.
No mask overlay substitutes for those pixels. No physical units are invented.

The original Huatuo chatbot supports multiple images even though the existing
MERIT wrapper permits only one. This entry uses native chatbot preprocessing
with two real placeholders and tensors: `[2,3,336,336]`. It does not modify
shared model code or concatenate images into a montage. Full-view control has
the same image-token count. Candidate scores are normalized native logits,
teacher-forced on the exact saved answer IDs including EOS. Mean is the frozen
ranking rule; sum is an unselected diagnostic and produces the same choices.
No repetition-penalty-processed scores are relabeled as model likelihoods.

Important scope limits:

- This is a **full-answer likelihood plumbing diagnostic**, not the proposed
  claim-level semantic selector. No new answer generation occurred.
- The two original candidates exist for every case; only 6/8 have different
  text, and two of those differences are punctuation-only. Do not call all
  six semantic correction opportunities.
- Exact native-name matching is an explicit limited selection/admission
  boundary, not a solved general request parser. Other experts are not changed.
- Anatomical crops may assist looking for abnormalities but do not certify
  disease or lesion extent. Relational questions may lack the second required
  structure; providing one crop does not complete a measurement protocol.
- Deep-copied inherited packets and normal packets produce identical operation
  views; this tests this entry's pure view function, not every production runner.
- Classification/text-expert efficacy and newly generated candidates are not
  tested here. No full benchmark is launched.

## Mechanism findings

There is a genuine correction opportunity where the native MERIT answer
contradicts the short Baseline answer and matches the reference direction.
The crop moves the MERIT-vs-Baseline mean-log-probability margin from about
-2.1022 to -1.8201, but the ranking still selects Baseline. This establishes
an observed response to the crop, not verified medical benefit.

Two punctuation-only candidate pairs show approximately 2.15 and 2.40 nats/token
differences under original-image scoring. Thus whole-answer scores remain
strongly affected by phrasing/EOS and are not clean critical-fact measurements.
Changing to sum normalization does not resolve choices in this pilot.
Neither crop nor segmentation text rescues the existing correction.

The next scientific decision is whether to address natural claim representation
before further selection work. Do not tune alpha, confidence cutoffs, crop
padding or question-specific rules using these outcomes. In particular, do not
scale this exact all-Baseline selector and call it a trusted Gate.

## Acceptance and cost

- 61 relevant CPU tests passed, including 3 new tests; syntax and diff checks
  passed. An initial old-test path did not exist; corrected to the actual
  `test_evidence_agent.py`, not reported as a test pass.
- 8/8 single-image token/tensor serializations exactly match the frozen ANCHOR
  Huatuo adapter. 8/8 crops differ from original after actual preprocessing.
- 64/64 candidate scoring calls verify answer-label alignment and agreement
  with native cross-entropy. 8/8 normal/inherited operation views match.
- Total candidate scoring time 10.199 seconds; crop arm 3.015 seconds,
  repeated-full-view 2.996 seconds. Two model loads total 40.875 seconds.
- Peak PyTorch allocated memory 19,383,659,008 bytes, approximately18.05 GiB.
- Times exclude some preprocessing, data preparation, evaluation and historical
  candidate/expert acquisition; not an end-to-end speedup claim.
- Only host GPU1 / container GPU0 used, UUID
  `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`. CUDA check passed with timeout.
  Host GPU0 was checked but not needed for 8 cases. Other jobs untouched.
- Both inference stages ran in tmux and completed. GPU returned to18 MiB usage.

## Version, artifacts, commands

Base commit `783350dbfae0dd2b2fe72be71b69eb9d542fbe6f`; isolated branch
`experiments/observation-blind-train-v1`, worktree
`/home/dbw/merit-feddg-observation-blind`. Frozen experiment identity:

`d1f2415fac6ca0d9db6023ee585d6a5c47e69e363605ee7c83e49828759d6bfe`.

Reused controls:
`/home/dbw/merit-feddg-huatuo-critic/runs/native-slake128-confirm-v2`.
Raw local outputs: `runs/observation-blind-slake8-v1/` including `protocol.json`,
`cases/`, `canary.log`, `run.log`, `complete.json`, `evaluation.json`.
Only [sanitized aggregate](../reports/observation-blind-slake8-v1.json) is tracked;
images, patient text, mask arrays, weights and credentials stay untracked.

Commands, executed with the existing environment:

```bash
export PYTHONPATH=/home/dbw/merit-feddg-observation-blind:/home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -m pytest tests/test_evidence_agent.py tests/test_evidence_admission.py tests/test_observation_blind_probe.py -o addopts='' -q
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/observation_blind_probe.py prepare
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -u scripts/observation_blind_probe.py canary
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python -u scripts/observation_blind_probe.py run
/home/dbw/.runtime/miniconda3/envs/huatuo/bin/python scripts/evaluate_observation_blind_probe.py
```

Outputs are exclusive-write; these commands are a record, not instructions to
overwrite/restart this completed run. A new experiment needs a new output root.
Full design/review remains at
`/home/dbw/research-notes/trainingfree-evidence-utility-20260919/RESEARCH.md`.
This crop/likelihood composition is not claimed as novel: see MedRAX
([paper](https://proceedings.mlr.press/v267/fallahpour25a.html)), ViperGPT
([code](https://github.com/cvlab-columbia/viper/blob/main/image_patch.py)),
and existing project CRES/paired-support diagnostics.
