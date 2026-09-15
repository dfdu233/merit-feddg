# Classification and generated-observation guidance, GPU1 only

2026-09-15. The previous segmentation full run is complete. This is a NEW
comparison targeting two different expert capabilities, not a relabeling of
the old segmentation `text_soft` result.

## Device and frozen scope

Only container GPU0, physical host GPU1,
`GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`. The runner checks this UUID and
`CUDA_VISIBLE_DEVICES=0` before loading a model. No host GPU0 job is launched.
Existing environment, LLaVA-Med source/checkpoint, CLIP and expert caches are
reused. No downloads, training, dependency upgrades, score calibration or
test-based strength selection. Older code and result roots are unchanged.

VQA-RAD 451 and SLAKE 2094 original full manifests are used independently for
classification and generated-text channels. All rows remain in each channel's
denominator. Cached classification coverage is only 6/451 and 60/2094; generated
observations cover 174/451 and 740/2094. Actual delivery must be checked again.
Unsupported rows reuse the completed matching control and are unavailable,
not successful zero-call interventions. The ordinary generalist and experts
are not rerun; a matched text control is generated only for applicable cases.

This phase does not restart segmentation, retrieval or MIMIC report experiments.
MIMIC text-generation transfer still needs a separate report-scoring run; it
must not be mixed into this VQA scorer or reported as completed here.

## Literature-to-code decisions

| Primary source / official code | What is relevant | What this implementation does NOT claim |
|---|---|---|
| [CAD, NAACL 2024](https://aclanthology.org/2024.naacl-short.69/), [code](https://github.com/xhan77/context-aware-decoding/blob/main/group_decode_fileio.py) | Same emitted prefix, signed weighted conditional/unconditional logits. Implement a distinct fixed CAD arm. | This is a medical adaptation with greedy decoding, not a reproduction of all paper datasets/settings, and not a new invention. |
| [DExperts, ACL 2021](https://aclanthology.org/2021.acl-long.522/), [code](https://github.com/alisawuffles/DExperts) | Experts and anti-experts can steer generation through distributions over a common vocabulary. | Image classifier similarities are NOT language-model logits. We do not have the paper's trained expert/anti-expert language models. |
| [FUDGE, NAACL 2021](https://aclanthology.org/2021.naacl-main.276/), [code](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation) | Distinguishes partial-text attribute likelihood from next-token LM probability. | Its learned future discriminator cannot be replaced by an image classifier without an additional validated interface. No discriminator is trained here. |
| [PPLM, ICLR 2020](https://arxiv.org/abs/1912.02164), [code](https://github.com/uber-research/PPLM/blob/master/run_pplm.py) | Attribute objectives can steer hidden states at inference. | No validated differentiable image-classifier/text-prefix objective exists here; no PPLM performance or native classifier guidance is claimed. |
| [CoCoA, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.348/) | Warns that fixed contrastive policies can fail in low-conflict settings; uses entropy/divergence measures. | Model distribution concentration is not calibrated medical correctness. This paper is reviewed, not reproduced; no adaptive gate is silently added. |
| [TRACE, ICML 2025](https://proceedings.mlr.press/v267/weng25b.html) | Future attribute probabilities require a model of possible continuations. | Its distilled HMM/classifier machinery is outside the current training-free environment. Not enabled or counted as a baseline. |

Only CAD's equation is newly implemented; its official `group_decode_fileio.py`
was inspected for shared-prefix weighted scores and KV handling. The other
papers constrain interpretation/design, not justify unimplemented features.
No external code files are vendored and no extra paper environments installed.

## Four fixed arms for each channel

Let z0 be the model with the selected channel removed, and z1 the original
model with the selected channel delivered as semantic text.

- `text`: ordinary greedy z1.
- `without_channel`: ordinary greedy z0.
- `blend`: .5 z0 + .5 z1, unchanged interpolation strength from the first run.
- `cad`: 1.5 z1 - .5 z0. The same .5 magnitude is fixed before this new run;
  it is not searched on the known bad cases. No additional plausibility filter.

Both modified decoders receive the SAME committed prefix in their two branches.
Blend zero must equal z0 exactly; CAD zero must equal z1 exactly. A mismatch
stops the experiment. Runtime failures and empty candidates are not medical
negatives or fallback successes.

The omitted-evidence confound is explicitly prevented: removing the selected
channel does not allow formerly omitted nonselected evidence to enter the
freed prompt budget. The exact remaining presented IDs must stay unchanged.
This tightening is explicit; old pilot files/code/results are not overwritten.

Classification remains **classification-conditioned language-distribution
guidance**, not direct image-classifier-to-vocabulary logit transport. Catalog
scores, scope and unknown labels retain their old semantics. Generated-text
guidance uses the cached CheXagent observation, not a ground-truth report.

## Verification, execution and costs

Entry: `scripts/run_class_text_guidance.py`. Real-resource preflight passed;
CAD equation/endpoints/shared-prefix CPU tests passed. The scheduling canary
uses the first available cached classification and generation case in each
dataset, not outcome-selected cases. Four real cases, same full manifests.
Canary must pass before full continuation, which reuses its completed rows.

Frozen output root:
`runs/class-text-guidance-v1/13a3e59cbcb99caa714fea919b3b50fb99c80ae6b9aa90e3c8b8879872e35f35`.

```bash
cd /home/dbw/merit-feddg-soft-guidance
scripts/with_gate_kb_env.sh scripts/run_class_text_guidance.py \
  --base-run runs/soft-guidance-full-v1/6ff365dc97261b29232acdab0a1060109bfd342c8dcb61d5621f7bfa07295440 \
  --output runs/class-text-guidance-v1 --canary
# Same arguments without --canary continue the full manifests.
```

Logs: `runs/soft-full-checks/class-text-canary.log`, followed by
`class-text-full.log` after successful inspection. tmux keeps jobs alive when
the client disconnects. Full completion requires exact IDs for all four
dataset/channel pairs. The pinned original ANCHOR CLOSED/open-recall scorer
then runs offline; it reports full and active-row gains/harms, old correction
and harm cohorts, paired image bootstrap and actual decoding costs.

Prefix replay is still the verified implementation. This run does NOT claim a
KV-cache speed optimization. Additional CAD cost is reported separately; model
load, controls and zero checks are recorded. No resource is called free merely
because a previously completed output was reused. No current full scores exist.

Validation outcome: all 871 CPU tests passed. Four real canary cases completed
and passed both zero endpoint checks; full continuation has been launched in
tmux `class-text-full` on physical GPU1 only. Full scores remain pending.
