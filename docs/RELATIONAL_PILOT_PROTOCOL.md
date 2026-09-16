# Real-image relational evidence pilot: protocol before execution

Experiment `slake-relational-evidence-dev-v1`, 2026-09-16.
Base `92955b9`; branch `experiments/medrax-relational-pilot-20260916`.

Vector: Does a fixed observer's explicit relational evidence improve a fixed actor's real-image answers across CT/MRI/X-ray and multiple anatomy strata, beyond a lossless fact table and direct vision?

This is a DEVELOPMENT mechanism diagnostic, NOT a test of unseen pretraining combinations, a new method claim, or a MedRAX reproduction. Med-MAT already studies modality-anatomy-task composition; MedAgent-Pro already implements multimodal tool workflows. Tool-mediated observation is the substrate, not the novelty. No new gate or expert fusion is involved.

- Source: official local SLAKE train.json; English VQA only. Exclude all validation/test image RGB hashes and deduplicate selected RGB hashes.
- Select up to four unique images in each predeclared stratum: CT abdomen, CT lung/chest, CT brain, MRI abdomen, MRI brain, X-ray lung. Query requires predeclared relational words (left/right/above/below/behind/front/between/largest/biggest/bigger/smaller/closest/next to/relative). Sort by SHA256(qid + image reference), not answer or model performance. Preserve each official query exactly. Brain and chest metadata groups are normalized only for sampling/reporting, never supplied to inference.
- Fixed observer and actor: local Qwen2.5-VL-7B-Instruct; BF16, deterministic greedy generation; same checkpoint in distinct calls. This is a same-backbone observation tool, not independent medical expertise.
- Observer sees the image only, no question or reference, emits up to 8 entities and 12 relations in a fixed JSON schema. Maximum 512 new tokens. Invalid JSON/unknown IDs are failures with coverage reported; no silent repair or label-guided retries.
- Actor arms: direct vision; image+observation graph JSON; image+lossless fact-table rendering; image+wrong relation graph; image+the same wrong relations in table form; graph without image. Maximum 32 answer tokens. Same instruction and question; record actual prompt tokens and latency. Layout token counts may differ; do not infer a graph algorithm effect from a format effect.
- Wrong relations: if >=2 edges with distinct tails, cyclically permute tails; keep nodes/attributes/edge labels fixed. Report whether a usable intervention exists and exact changed edge count. This deliberately changes asserted facts; it is NOT an information-equivalent transformation or new input truth.
- H1: useful relational evidence produces repeatable answer gains and sensitivity to factual relation interventions across strata. H0: effect is absent, harmful, or explained by extra text / representation format / unchanged image evidence / parser artifacts.
- Strongest simple controls: direct actor, lossless table, identical graph/table fact multiset check, wrong-relation table. If graph only beats direct but not table, graph-specific evidence is absent.
- Evaluation: offline normalized exact match (case/punctuation/articles/space); report raw outputs and exact-match limitations. No post-hoc synonym tuning. Strict graph-vs-direct and graph-vs-table rescues/harms, manipulation coverage, parse coverage, per-stratum counts, costs. These are small-development descriptive results, not generalization estimates.
- Labels are written to a separate evaluation file; inference only reads the allowlisted manifest and image bytes. No dataset triples, masks, location, modality, answers or explanations are sent to observer/actor.
- Run one canary, inspect parsing and transport, then finish the frozen cohort. Ordinary operational fixes must be recorded. Scientific changes get a new run identity.
- Stop this formulation if useful evidence is not obtained, graph fails the table comparison, or interventions have inadequate coverage. A negative result triggers a new Vector within the user's authorized multimodal/graph research scope; do not reframe it as a success.
- Conditional continuation: if evidence bottleneck is demonstrated, compare question-conditioned observation under a NEW protocol, holding observer budget fixed; it tests acquisition vs downstream representation, not graph novelty. If a robust graph mechanism survives, check nearest methods again and proceed to image/patient-disjoint held-out combinations with explicit training provenance limits.

GPU: host GPU1 UUID `GPU-3846413a-4238-d307-b1f3-10c2dfbe002c`, exposed as container GPU0, rechecked at launch. No other jobs are terminated. Future all-modality/all-anatomy claims are outside this pilot's coverage.
