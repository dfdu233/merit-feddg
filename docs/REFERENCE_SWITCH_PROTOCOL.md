# Frozen-candidate reference switching: capability gate

Vector: can existing fixed tools supply target regions well enough to test reference-conditioned binding, independently of free coordinate generation?

Input audit found57 groups/118 original Task7 questions with byte-identical target RGB pixels and different source references among1043 two-image questions. Offline labels confirm every group has differing target boxes. This is existing benchmark structure, not our new dataset contribution.

Freeze up to4 groups each CT→MRI, MRI→CT, MRI→US by target-pixel hash order. Exclude all source/target filename-derived patient IDs from previous28-case development pilot; deduplicate across new groups. Do not select by candidate coverage or reasoning outcomes. These remain public-benchmark DEVELOPMENT cases, and case IDs are not independently certified patient IDs.

SAM ViT-B `facebook/sam-vit-base`, revision70c1a07f894ebb5b307fd9eaaee97b9dfc16068f, supplies a target-only, question-independent automatic proposal bank:16x16 point grid,32 points/batch, zero crop layers, IoU score threshold.88, stability.95, mask threshold0, crop NMS.7. Keep all resulting masks/boxes, no label-guided prompt points, no gold masks, no answer-dependent threshold or mask merging. SAM is an existing perception baseline; its medical reliability is unproven. Generic automatic proposals are not a method contribution.

Compute proposal coverage offline only: best candidate IoU per original reference question, recall@.5, and fraction of groups with ALL reference answers covered. Also number of candidates, average best IoU, and all-source/target groups. Preserve uncovered cases. If fewer than4 groups or less than half of selected groups are fully covered, do not spend LLM calls on graph selection with this proposal tool. This is a pragmatic development feasibility rule, not a statistical finding or threshold tuned to outcomes.

If coverage passes, freeze a follow-on candidate-selection protocol comparing source/unary evidence, ordinary partial matching, same-fact relation table, and graph with a relation-permuted negative control. The key metric is correct switching of candidate under a valid source-reference change while target pixels and bank remain fixed. No graph superiority or Bit Flip is asserted by coverage. Generic matching and reference grounding already exist; the purpose is to decide whether a relation-dependent signal is even testable.

## Input-only selection correction, before proposal generation

v1 hash-greedy dedup selected3 groups. Exhaustive audit of the8 eligible groups (all CT→MRI) shows a maximum of4 mutually patient-disjoint groups. Therefore v2 uses maximum-cardinality disjoint selection, tie-broken by lexicographically ordered target hashes, with no proposal scores or labels used in optimization. Preserve v1 manifest unrun. All model/coverage settings and go/no-go rules remain unchanged. This is only a sampling correction; four small groups in one transition cannot establish cross-modality generalization.
