# Additional Huatuo TEST pilots: PMC-VQA and MMMU

HostGPU1 only.512fixedSHA256(id)-ordered cases per dataset,1024total. Historical baseline/candidate reused; imagepaths and candidate imagehash checked. Source controls and positiveD/no-negative joint rule unchanged. Previously failed qualification remains failed; exploratory postprocessing only. No full-dataset claims.

MCQ evidence adapter maps unambiguous bare answer letters to original option text without labels; other outputs abstain.54PMC and231MMMU pairs fail this deliberately strict mapping. Original answer strings retained for scoring; no label-aware candidate rewriting. Benchmark evaluation can parse more forms than the evidence adapter, so coverage and scoring parseability differ.

| Arm | PMC-VQA512 accuracy % | MMMU512 medical-scope accuracy % |
|---|---:|---:|
| Historical Generalist |49.4141|42.9688|
| Historical BARD |47.8516|42.7734|
| BiomedCLIP filter |49.6094|43.1641|
| CONCH filter |49.8047|42.9688|
| PLIP filter |49.2188|43.1641|
| Joint filter |49.4141|42.9688|

PMC uses existing multiple-choice parser correctness. MMMU uses pinned official parser with existing exact-option-label repair and unparseable-as-error (no random-guess credit). Outputs/scores use identical512IDs. Marginal1–2correct-answer differences do not establish superiority, especially across multiple explored arms.

Important route audit:419/512MMMU cases are outside the official five Health and Medicine subjects, yet old routes label358of thoseMRI and52CT. Native scores were initially computed wherever that route permitted, before this defect was identified. Those raw observations are preserved, not erased. A separately reported label-free subject-scope sensitivity excludes nonmedical subjects from acceptance; it is not a preregistered superiority claim. Without scope guard: BiomedCLIP43.3594,joint43.1641. With guard: BiomedCLIP43.1641,joint42.9688. Joint accepts0changes afterguard; equality is complete fallback, not successful verification.

Scope guard allows Basic Medical Science, Clinical Medicine, Diagnostics and Laboratory Medicine, Pharmacy, Public Health. It is coarse; subject membership alone does not prove each image is supported. Await properly validated input applicability rather than treating MRI-like forced routes as evidence of coverage.

PMC joint accepts5edits with zero net correct-count gain. CONCH accepts6and gains2correct answers; PLIP accepts6and loses1. MMMU scoped singleBiomedCLIP/PLIP each accept1and gain1; not robust performance evidence. No winner deployment or threshold tuning followed these TEST pilots. MUSK not used.

All artifacts under runs/test-mcq; source code and aggregate results adjacent. Evaluator invocation requires PYTHONPATH=/home/dbw/ANCHOR. Existing historical baseline paths and mapper abstentions in input-audit.json. Generic MMMU historical baseline is oldcloud20260919, not an asserted latest repaired baseline; interpret only this matched exported set.
