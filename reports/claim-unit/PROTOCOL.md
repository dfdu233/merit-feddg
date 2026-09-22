# Claim-unit ablation

Huatuo81 closed-answer source-Q cases, frozen outputs. Every baseline/candidate has unambiguous leading yes/no.15 polarity flips. No reference-based subset selection. Full81 includes same-polarity cases for denominator and controls.

Compare original full-answer propositions vs the existing decontextualize_vqa on leading yes/no only. The latter uses explicit finding negation for supported question templates and retains a question-answer proposition for other templates. Do not claim full clinical claim decomposition. Only evidence text changes; no candidate regeneration or label-guided prompt changes. Controls selected from the same81manifest using unchanged stable hash and4control count, independently of answer length. Recompute both arms for a matched comparison.

Primary diagnosis: strict yes/no correctness of original frozen answers,81denominator; count corrected and broken polarity flips. Each unqualified verifier accepts only positiveD on an actual flip. All-three accepts any positive with no available negative; missing evidence abstains. Independent proposer fault-group exclusion preserved. Decisions operate on original frozen answer choices. Neither experiment grants formal v3 authority. Previously inspected source-Q is development, not independent canary or TEST.

Only hostGPU1; unrelated jobs left running. No MUSK dependency. If no verifier improves on baseline, do not expand or tune thresholds on these cases.
