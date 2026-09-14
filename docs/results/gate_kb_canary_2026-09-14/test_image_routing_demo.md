# Requested single TEST-image routing demo

This is an explicitly user-requested plumbing demonstration, not a TRAIN canary,
accuracy estimate, rule-selection exercise or evidence of independent BUSI
generalization. No labels were opened; no policy was changed after the result.

Implementation: `ba23927b62c87fa389c7f55749b60caaadd2b20e`.
Input: first `test_images[0]` from the existing official BreastMNIST archive,
MD5 `750601b1f35ba3300ea97c75c52ff8f6`, native 28x28 pixels. Selection preceded
inference and did not use predictions or labels. The synthetic question was:

> In this breast ultrasound image, is the finding malignant or non-malignant (normal or benign)?

The question specifies a classification task but does not name a model or give
the sample's label. Ultrasound modality was supplied from dataset metadata;
automatic image-modality inference was NOT tested. No patient text was used.

## Real execution

Used the complete existing expert registry plus `small_medical_ready.yaml`,
restored legacy contracts, disabled already-disabled tools and source retrieval
without a source corpus. Used the existing **agent** entry, not the previous
four-arm experiment's fixed scheduling entry. Same frozen three-action budget,
shared rendering, 64 answer tokens, 48 observation tokens and purpose gate.
No expert was forcibly selected or substituted.

Initial eligible actions were U-Bench U-KAN segmentation and BreastMNIST
classification. Actual planner selections were A0, A0, A0:

1. U-Bench U-KAN, real predicted mask.
2. Real mask-derived crop and generalist local observation.
3. BreastMNIST, strict-loaded official classifier and valid native catalog.

Both specialists were actually invoked through CapabilityPool. Classification
including model initialization took 0.145 s; segmentation including initialization
took 1.064 s. Specialists used their existing default CPU configuration; LLaVA-Med
used authorized container GPU 0 (physical GPU 1). These are one-call timings, not
a GPU specialist throughput benchmark.

**Routing eligibility and eventual invocation passed. Priority-routing quality
was not established:** both initial choices selected the first list entry; at
the third step the classifier was the only remaining non-STOP action. The fixed
entry would likewise start with segmentation. No order counterfactual was run.

## Delivery failure, not correction success

Classifier output disagreed with the generalist's baseline answer. All three
purpose judgments were AUXILIARY. Thus no evidence was admitted as an answer
premise, the runner recorded `no_answer_evidence_after_gate`, final synthesis
was not invoked, no candidate was produced, and the exact baseline was retained.
All gate prompts fit the context budget, so this was not a token-budget failure.

The crop reader also called the ultrasound image a CT scan. This is a concrete
modality-description failure in this run; do not describe the generated crop
text as verified medical knowledge. The true classification label was not
read, so the disagreement cannot be labeled a good/bad correction case.
Uncalibrated softmax magnitude is not a correctness probability.

Observed calls: three planners, one crop reader, three purpose judgments, two
specialist invocations and one initial generalist answer; zero final-synthesis
calls. One image cannot establish routing accuracy, gate usefulness or novelty.

## Local artifacts (not uploaded)

Under `/home/dbw/merit-feddg-evidence-gate-kb/runs/breast-test-routing-demo/`:
`probe.py`, `run.log`, `result.json`, `test-image-0.png`, and predicted-region
`crops/`. The JSON retains the real prompts, action mapping, native expert
outputs, generation outputs, purpose decisions, delivery and costs. Patient
images/crops and raw per-image predictions are not included in this Git report.

Executed in tmux with a 300-second timeout using the existing environment:
`scripts/with_gate_kb_env.sh runs/breast-test-routing-demo/probe.py`.
The job completed. No training, dependency change, rule tuning, rerun on another
test image, or full experiment followed this demonstration.
