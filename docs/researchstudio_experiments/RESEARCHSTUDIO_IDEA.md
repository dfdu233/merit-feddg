# ResearchStudio-style idea card: receiver-mediated expert collaboration

**Status:** reviewer-defensible research hypothesis for the existing MERIT implementation. It is not an official IdeaSpark DONE artifact and is not experimentally validated.

## Motivation

Training-free medical expert collaboration has two distinct difficulties that are often conflated. First, expert models expose incompatible native outputs: a classifier score, a segmentation mask, a retrieved passage, and a specialist VLM sentence do not live in a shared prediction space. Second, once those observations influence a generalist, the system must decide whether a proposed revision is useful or harmful.

Recent work covers important pieces but leaves the heterogeneous-interface problem open. Visual Evidence Prompting converts expert outputs to prompts; CCD uses structured radiology signals for contrastive decoding; RobustRAG isolates retrieved passages before aggregation; Inference-Time Consensus aggregates next-token distributions from separately fine-tuned reference models that already share an output space. MERIT's candidate delta is to use the *same frozen receiver* as the comparison operator for task-native experts whose original outputs are not directly comparable.

## Core idea

Let specialist $e$ return native artifact $o_e$ in an arbitrary output space $\mathcal O_e$. A deterministic renderer $\rho_e$ exposes only the artifact semantics that the receiver can consume. At a shared committed prefix, the frozen receiver produces

$$
p_0(\cdot\mid I,q,y_{<t}),\qquad
p_e(\cdot\mid I,q,\rho_e(o_e),y_{<t}).
$$

The downstream common object is not $o_e$ itself but the receiver-mediated intervention

$$
\Delta_e = \log p_e - \log p_0.
$$

Thus, classification scores, masks, text findings, and retrieved passages need no cross-model projection or shared vocabulary. Source-isolated branches prevent unrelated evidence from interacting before each intervention is measured. Adaptive BARD is one source-aware commitment rule on top of this receiver-mediated interface; the interface and the commitment rule should be evaluated separately.

## Preliminary prior-art delta

- **VEP:** heterogeneous expert outputs can be symbolized as prompts, but it does not make isolated receiver-side interventions the shared aggregation object.
- **CCD:** structured radiology experts guide token decoding, but the method is specialized to radiology signals and does not define a general receiver-mediated interface for incompatible expert artifacts.
- **RobustRAG:** isolate-then-aggregate is established for retrieved text passages; MERIT's atoms are task-native medical specialists with different output semantics.
- **Inference-Time Consensus:** source-wise next-token consensus is established when source-specific reference models already share a base architecture and output vocabulary. MERIT instead creates a common distribution only *after* each heterogeneous artifact is interpreted by the same receiver.

**One-sentence delta:** Unlike consensus methods that aggregate models already expressed in a common output space, MERIT uses a frozen medical VLM as the common measurement operator that converts heterogeneous task-native specialist artifacts into comparable same-prefix receiver interventions, enabling one decoding rule without learning cross-expert alignments.

## Falsification and scope of claim

The core claim is not supported if the benefit of MERIT can be reproduced by simply concatenating the same evidence, or if isolated receiver branches provide no benefit over a matched joint-context arm once delivered evidence is held fixed.

Adaptive BARD's additional contribution is unsupported if it does not improve the rescue–harm trade-off over unprotected isolated aggregation or a strong base-relative consensus baseline using the same branch scores.

The frozen five-arm chain tests *evidence utility*, then *isolation plus merge*, then *robust aggregation*, then *anchored commit*. Because both `joint_all` and isolated arms pass through the receiver, their contrast alone does not prove that receiver mediation is superior to every non-receiver alternative. The paper should present the heterogeneous interface as a design plus measured intervention-compatibility claim, and reserve causal language for the contrasts that actually change one component.
