# Relational acquisition development pilot v1

Frozen after observation-graph pilot, before new cohort selection/inference.
Experiment `slake-relational-acquisition-dev-v1`.

Vector: With the same frozen model and crop tool, do real relational questions contain order-stable cases where two observed regions correct an error that neither region alone corrects?

Community location: uncertainty-guided visual acquisition, compact missing-evidence recovery, nonmyopic value of information. Prior art already includes all three. Graph factorization is only justified later if it improves transfer/sample efficiency over generic pair planning. This pilot contains no new graph policy and does not claim novelty for complementarity itself.

- Dataset: English SLAKE TRAIN VQA, questions matching bigger/smaller/relative to/left of/right of/between. Up to three unique images per the six strata of the prior protocol, deterministic SHA256 sorting. Exclude evaluation RGB hashes AND every image selected in the preceding pilot. Missing strata remain missing; no replacement based on labels/results.
- The full original image is resized to a 112x112 overview using preserved aspect ratio. This deliberately restricted observation is a stress condition, not normal MedRAX performance. Four fixed quadrants of original pixels form the candidate bank; never use segmentation masks, answers, dataset triples, or answer-conditioned regions. Crop coordinates always refer to displayed image coordinates, not patient laterality.
- Source tool: deterministic pixel crop; same Qwen2.5-VL-7B-Instruct actor, BF16 greedy decode, maximum 32 answer tokens. Same question/prompt for every evidence set; image labels and coordinates preserve common reference frame. Processor limits 256*28*28 to 512*28*28 pixels per image as before.
- Exhaustive diagnostic arms: overview alone, each singleton crop (4), each pair (6) in both orders (12), all four crops (1), full original image (1), question only (1). All arms provide the overview except full-original and question-only. Total 20 answer calls/case.
- Practical baseline: plan two quadrants from overview+question in one 16-token greedy call; replay the already computed matching pair. Planner cannot read other candidate pixels. Invalid plans are failures, not repaired with gold. This is generic planning, not a learned graph method.
- Confidence baseline: rank singleton outputs by mean generated-token log probability. It reads ALL four crops and incurs four scoring/answer calls; report it as an expensive diagnostic, not an equal-two-observation deployable policy. It is UG-inspired, not an official reproduction.
- Primary defining-property count: overview wrong, all four singletons wrong, at least one pair correct in BOTH orders. This stricter criterion excludes cases with an already sufficient singleton. Report order-sensitive pairs separately.
- Secondary: best singleton oracle and best order-stable pair oracle; generic planner outcome; full-resolution and all-crops bounds; strict normalized exact match. Gold is available only to evaluation, never selection/inference/planning. Oracle ceilings are not deployable results. No post-hoc answer normalization changes.
- Record prompt/generated tokens and observed timing. Pair views may contain more pixels/tokens than singletons; that is the acquisition budget being tested, not a claim of equal compute. Generic planner is compared with oracle pairs at the same two-view budget, with planner overhead reported.
- Stop if there is no stable complementarity or no attainable pair headroom. If generic planning reaches the attainable bound, extra graph planning is not supported. If a gap survives, independently check image evidence and compare compact sequential planning before claiming a graph-specific opportunity.
- Small DEVELOPMENT samples cannot confirm unseen MAT generalization. Tool/agent training remains frozen; unknown checkpoint pretraining exposure is not certified. No synthetic questions or clinical labels are invented.

Nearest-method audit: `/home/dbw/research-notes/medrax-composition-20260916/relation_acquisition_tension.md`.
