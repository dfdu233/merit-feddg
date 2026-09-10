# Independent uncertainty source confirmation, 2026-09-10

Verdict: **do not scale this method to target evaluation yet**. The new source
cohort confirms that evidence reaches generation, but it also contains two clear
retrieval-induced errors and one spatial-evidence-induced laterality error. Numeric
uncertainty ranges produced no range-specific medical benefit.

## Frozen run

- Config: `configs/uncertainty_source_pilot.yaml`, unchanged thresholds.
- Output: `runs/uncertainty-source-confirm-seed29/llava/e48b658050cd0748`.
- Cohort: 20 VQA-RAD source cases, seed 29, 10 per image-proxy group; 20 unique
  images/groups and zero sample, group or pixel-hash overlap with the earlier
  six-case uncertainty canary.
- Isolation: 20 source cases, zero target generations, no policy fitting,
  Block-NONE exact parity 20/20. Both domains are dataset proxy groups, not
  hospitals or independently acquired domains.
- Runtime: existing huatuo environment, LLaVA-Med, CheXagent and XRV weights;
  Hugging Face and dataset access were offline. Nothing was installed, upgraded
  or downloaded.

Input/output integrity hashes:

| Artifact | SHA-256 |
| --- | --- |
| `source.jsonl` | `48ddc71ef063030eeebfa79c96cc29a8dd1bcc6163b0a202af6c5958361d8ef5` |
| isolated `target.jsonl` | `cdd9a7188e0e9fa392283fe68167b9a9fbee38901748fa65f83a692517cc46bc` |
| `references.json` | `93def9ae9d58d351436108d7b7d4f054f3e055867137e0eda3791da9b64478f5` |
| `source-diagnostics.json` | `cb11eebf2b94c5d65fe5ac0c7ae4cd586a8cab215a0c0df8eef1698784648f7e` |
| `diagnostic-summary.json` | `d19a803f230f5b7c8d6c24adfa8f7333ce049347774f6b496e059fd31e446936` |

## Medical direction by channel

The semantic counts below are a conservative manual comparison with the supplied
references, not a blinded clinician annotation. “Benefit” requires a materially
more correct answer; lexical overlap, shorter wording and a nonempty call do not
qualify.

| Channel | Cases | Clear benefit | Clear new harm | Result |
| --- | ---: | ---: | ---: | --- |
| Related source retrieval | 20 | 4 | 2 | Mixed and unsafe for automatic admission |
| XRV findings | 3 | 0 | 0 | Nonempty scores, no clear correction |
| CheXagent description | 4 | 1 | 0 | One useful localization; too sparse |
| XRV anatomy segmentation | 1 | 0 | 1 | Correct right side flipped to left lung |

For source retrieval, uncertainty-point Token-F1 reported 8/20 improvements,
1/20 harms and 11/20 ties, with mean gain +0.0436. Medical inspection supported
only four clear corrections and found two harms. Thus the lexical result cannot
be used as the go/no-go criterion.

Clear corrections were:

- genetic/vascular question: a truncated AVM answer became `vascular`;
- image plane: wrong `coronal` became correct `axial`;
- imaging type: generic MRI became `T2-weighted MRI`;
- infarct vessel: wrong right MCA became correct `left MCA`.

Clear harms were:

- reference `psoas major muscle`: correct baseline `psoas muscle` became wrong
  `erector spinae muscle` (Token-F1 also decreased);
- reference `calcification`: correct baseline `outer rim ... calcified` became
  `located in the left frontal lobe` (Token-F1 reported a tie and missed the harm).

Several lexical improvements were not medical successes. For example, the
right-upper-lobe answer became only `right lung lobe` when the reference required
`right lower lobe`; `normal` became the non-answer `normal or abnormal`; and a
right-adrenal answer became right upper abdomen when the reference was the broader
`abdomen and pelvis`.

## Real expert examples

### CheXagent useful once

Question: `which lung lobe is the lesion present in?`

- Reference: `right lower lobe`.
- LLaVA-Med baseline: `right upper lobe`.
- CheXagent observation: `Right lower lobe`.
- Evidence-conditioned answer: `right lower lobe`.

This is a real correction, but one success among four CheXagent cases is not a
reliability estimate. The observation is recorded as unverified specialist output
with no calibrated confidence.

### Segmentation actively harmed laterality

Question: `where is the cardiac border more obscured?`

- Reference and baseline: `right`.
- XRV produced nonempty 512x512 Left Lung, Right Lung and Heart masks. Foreground
  fractions were 0.21278, 0.21526 and 0.08134; the recorded original-image boxes
  were `[0.5449,0.2308,0.9355,0.8413]`,
  `[0.0898,0.2244,0.4961,0.8637]` and
  `[0.4160,0.4920,0.7207,0.7804]`.
- Evidence-conditioned answer: `left lung`.

The real mask predictions were serialized into text memory as structure names,
boxes and foreground fractions. Every diagnostic branch recorded `visual_views=0`,
so the pixel masks themselves were not supplied to LLaVA as overlays or crops.
This proves text transport of mask-derived spatial facts and demonstrates harm;
it does not validate the visual spatial bridge.

### Findings scores did not answer localization

For the right-lower-lobe question, XRV's largest score was uncalibrated
`Infiltration=0.1449`. The answer changed from wrong `right upper lobe` to incomplete
`right lung lobe`, still missing “lower.” The score list is explicitly marked
uncalibrated for the query domain and cannot establish lesion location.

## Uncertainty and perturbation

Four real XRV calls supported perturbation measurement: three findings classifiers
and one anatomy segmenter. Gamma 0.95/1.05 probes added 12 forwards and 0.643 s.
Maximum per-label findings changes were 0.0138248, 0.0145904 and 0.00316687; anatomy
maximum nonempty-mask IoU distance was 0.0201379. Clinical invariance was not
verified, and these values do not measure correctness or applicability.

Across all matched evidence/request arms, uncertainty point and uncertainty text
generated identical answers in 31/32 pairs. The last pair differed only by an extra
word, `image`, with identical Token-F1 and no medical change. The range therefore
had 0/32 range-specific semantic effects and no demonstrated medical benefit.
Source retrieval and CheXagent had no native alternatives and correctly remained
`unknown`; they were not treated as stable or reliable.

## Cost

- 20-case diagnostic generation: 171.42 s.
- Recorded expert events: 23.83 s — retrieval 8.31 s, XRV findings 1.73 s,
  CheXagent 11.87 s and XRV anatomy 1.92 s.
- Summed measured generation plus expert work: 195.25 s, or 9.76 s/case.
- Peak PyTorch allocation: 22.68 GiB.

This is deliberately expensive diagnostic replay: 180 evidence-conditioned answer
branches reused shared tool outputs. It is not the latency of a single production
route. Baseline generation summed to 12.45 s; cold model/encoder loads account for
much of the first retrieval and CheXagent calls. A production latency claim needs a
separate warm, matched single-route benchmark.

## Decision

Do not run target or increase sample volume on the present admission logic. Before
scale-up, the minimum next evidence is a frozen source validation with independent
clinical adjudication, enough examples per expert/question scope to estimate harm,
a matched pixel-spatial arm, and a gate that rejects source/image associations that
do not entail the requested claim. These criteria are prospective; no threshold was
changed after observing this run.

Raw local files under the final run:

- `diagnostics/source-diagnostics.json`: complete per-case branches, answers,
  evidence payloads, masks and tool timings;
- `diagnostics/diagnostic-summary.json`: aggregate lexical and perturbation report;
- `diagnostics/evidence-audit.json`: 200 blind-audit records;
- `diagnostics/routing-audit.json`: image-only routing audit;
- `provenance.json`: frozen config, paths, runtime and expert provenance.
