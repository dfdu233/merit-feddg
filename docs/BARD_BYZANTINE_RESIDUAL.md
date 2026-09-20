# BARD v1: Byzantine-Anchored Residual Decoding

## Status

This branch implements a **training-free research candidate**, not an established
medical safety mechanism. It is designed for the harder setting where a
compatible specialist may be irrelevant, systematically wrong, or arbitrarily
misleading. It does not train a gate, use target answers, or interpret a
specialist's native confidence as a probability of correctness.

The implementation is additive to the existing matched evaluator. Existing
Generalist, packet, verifier, spatial and source experiments are left unchanged.

## Research hypothesis

Previous MERIT gates tried to estimate whether evidence was relevant, visually
used, stable, or externally preferred. Those observables do not identify
medical correctness. BARD instead asks:

> Can a candidate departure from a strong frozen Generalist survive a declared
> bounded-fault consensus test across independently conditioned receiver
> branches?

The primitive is not specialist text or specialist confidence. It is the
**effect the specialist has on the receiver's own next-token distribution**.

For exact committed prefix s:

    p0 = p_theta(. | I, q, s)
    pe = p_theta(. | I, q, Ee, s)
    re = log pe - log p0

Each residual is centered under p0 so branch-wide log-normalization offsets
cannot masquerade as a useful direction.

## Isolate first, aggregate second

BARD never concatenates several specialists into the same receiver context.
Native outputs are acquired from clean states and grouped by `expert_id`.
Multiple capabilities from one underlying expert count as **one Byzantine
node**, not several independent votes.

    expert A -> receiver A -> r_A
    expert B -> receiver B -> r_B
    expert C -> receiver C -> r_C
    expert D -> receiver D -> r_D
    no expert ----------------> p0

    r_A...r_D -> robust aggregate -> bounded-fault commit -> next token

All branches receive the **same already committed output token IDs**. The
existing `LlavaMedAnswerSession.next_scores()` replay path is reused, so BARD
does not compare freely generated branches with different prefixes.

## Robust aggregation

The main arm uses a geometric median of receiver residual vectors, computed by a
stabilized Weiszfeld iteration. Two matched ablations are also implemented:

- `isolated_mean`: isolated receiver residuals + ordinary mean;
- `isolated_geomedian`: isolated receiver residuals + geometric median;
- `bard`: geometric median + bounded-fault commit.

`joint_all` is the existing all-evidence joint-context behavior and separates
the value of isolation from the value of robust aggregation.

The geometric-median implementation follows the robust aggregation idea in RFA:
https://github.com/krishnap25/RFA/blob/40945d27a5246b2c82fd37bd453db4f5bc049952/models/model.py

RobustRAG is a second precedent: it evaluates retrieval contexts separately and
aggregates token distributions instead of concatenating all possibly corrupted
context first:
https://github.com/inspire-group/RobustRAG/blob/9bc35b2fa5fa7d1088383b2789ec19a512d316b1/src/defense.py

BARD differs from both: atoms are heterogeneous medical specialists and the
aggregated objects are **receiver residuals anchored to a frozen multimodal
Generalist**.

## Bounded-fault commit

For Generalist token a and robust-aggregate candidate b, expert-node e's margin:

    de(b,a) = [z0(b)-z0(a)] + [re(b)-re(a)]

With declared fault budget f, BARD v1 requires:

1. at least 3f+1 independent `expert_id` nodes;
2. at least n-f nodes with positive candidate margin;
3. the conservative lower support order statistic to remain positive.

Otherwise it emits the Generalist token. The default pilot declares f=1, so
fewer than four delivered expert nodes causes a **structural Generalist
fallback without evaluating the expensive expert receiver branches**.

This is conservative, but is not a theorem that the selected medical token is
correct. A uniquely correct single specialist can be rejected when independent
redundancy is absent.

The incumbent/fallback idea is related to safe policy improvement / SPIBB,
whose implementation preserves the baseline policy outside supported policy
improvement regions:
https://github.com/RomainLaroche/SPIBB/blob/833f32b642759111907d2c4d086d1ae1ce9d84cb/spibb.py

## Label-free single-fault stress test

The `bard` arm records a diagnostic stress test at every evaluated token.
Using already-computed residuals only, it replaces one expert residual at a time
with a strong sign-reversed residual and reruns the commit rule.

- no target/reference answer is read;
- no additional model forward is performed;
- corruption severity never affects production token selection;
- clean-token retention and Generalist fallback are reported separately.

This is deliberately weaker than a real wrong-patient knockoff experiment. A
future single-expert path should use metamorphic challenges and matched
wrong-patient expert outputs; they are **not** claimed as implemented here.

## Matched experiment

Configuration: `configs/matched_bard.yaml`.

| Arm | Joint context? | Aggregation | Bounded commit? |
|---|---:|---|---:|
| `generalist` | no | none | incumbent |
| `joint_all` | yes | none | no |
| `isolated_mean` | no | mean | no |
| `isolated_geomedian` | no | geometric median | no |
| `bard` | no | geometric median | yes |

The semantic packet renderer is fixed across isolated receiver branches and is
not a research variable. Old visual-contrast, NLI, external verifier and
tensor/spatial gates are disabled.

`--reuse-generalist` and `--reuse-expert-run` are supported for BARD.
Reuse remains subject to existing full-manifest, image, provenance and
request-key checks. Incompatible caches fail rather than being silently mixed.

Example:

    python -m merit_feddg.matched_evaluation \
      --protocol bard \
      --config configs/matched_bard.yaml \
      --manifest /absolute/path/to/manifest.jsonl \
      --output runs/matched-bard \
      --artifacts artifacts \
      --reuse-generalist /absolute/path/to/generalist-reuse.json \
      --reuse-expert-run /absolute/path/to/compatible-expert-run

Use source-side/canary data first. Do not use target results to change
`fault_budget`, expert count, aggregation method, or commit rules and then
report the same target as independent evaluation.

After a fully merged run, mechanism coverage can be audited without references:

    python scripts/audit_bard.py \
      --run /absolute/path/to/completed-bard-run \
      --output /absolute/path/to/bard-audit.json

The audit reports independent-node coverage, structural fallbacks, actual BARD
departures, and single-fault retention/fallback. It does not score medical
correctness.

## First falsification experiment

Before any large benchmark, report:

1. fraction of cases with 0/1/2/3/4+ independent delivered expert nodes;
2. exact Generalist parity;
3. joint-all, isolated mean, isolated geometric median and BARD task score;
4. paired Generalist improvements and harms;
5. BARD departure and structural-fallback coverage;
6. clean single-fault retention and fallback rates;
7. an offline fault-injection curve versus corrupted-node count where possible;
8. latency and actual expert/receiver forwards.

The first research question is not “does BARD beat the benchmark?” but:

> Does isolate-then-robust-aggregate preserve useful collaborative corrections
> better than joint context while making one arbitrary expert substantially less
> able to flip a strong incumbent?

If no, stop this branch rather than fitting another confidence score on target.

## Limits

- f=1 is a declared bounded-fault model, not evidence that real failures are
  independent or that only one specialist can fail.
- Specialists can share training data, architectures and biases; `expert_id`
  independence is a modeling assumption to audit.
- Fewer than four independent nodes means default BARD cannot improve the
  Generalist; this exposes where single-expert falsification is required.
- Geometric-median robustness in residual space does not imply clinical
  correctness.
- Synthetic sign reversal is a mechanism stress test, not a realistic medical
  adversary.
- No clinical non-inferiority, universal domain-generalization or SOTA claim is
  justified without fixed external evaluation.
