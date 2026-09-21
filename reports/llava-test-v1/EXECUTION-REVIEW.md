# Execution and storage review — 2026-09-21

## MMMU early EOS

The first1024 LLaVA-Med BARD cases completed with86 empty answers,967 structural fallbacks and139 answers reaching64 tokens. These are a completion-order subset, not full-corpus rates. Three empty cases (Accounting49, Agriculture1, Architecture_and_Engineering43) and one nonempty control were replayed through native generation and cached scores. Native token sequences exactly matched archived outputs; all inspected full score vectors matched bitwise. See mmmu-empty-diagnosis.json. All10500 frozen benchmark prompts exactly matched ANCHOR protocol_v2, including choices.

In the inspected Accounting49 example, the image-only route predicted MRI for a bond-pricing problem; BiomedParse returned unknown_anatomy_or_sequence and no admitted visual evidence. Only MedCPT remained. The frozen single-expert policy therefore retained Generalist, which emitted whitespace token28705 and EOS2. This identifies early native termination and an inapplicable medical expert pool; it does not prove all empty cases share one deeper cause. No forced answers, minimum-length changes, label tuning, or removal of failed answers was applied. Native replay here was bounded diagnosis, not a fresh baseline evaluation.

## Method consistency

BARD aggregation/commit/spatial/model-adapter source files remain unchanged. Existing cross-node Path probes reported exact token parity on first2cases, while this session audited deployed source/config hashes. This is bounded compatibility evidence, not proof of full-corpus bitwise hardware equivalence. Huatuo retains1024 tokens/repetition1.2/min1; LLaVA VQA64 tokens. Native expert request keys remain validated. Long-dataset Huatuo jobs explicitly reuse frozen LLaVA image-only routes and exact native caches across receivers; this is not Huatuo-specific fresh routing and is not a causal generation-matched baseline comparison.

## Disk failure and lossless repair

44450 Path workers hit ENOSPC. The benchmark session preserved223+230 completed prefixes and resumed at4511/5126. It owns those suffix recoveries; this session owns subsequent MMMU queues. Verified model shards and Path caches were relocated to system storage through unchanged-path symlinks. The additional MMMU caches are also placed on system storage on40297/44450; only assigned subsets are extracted.42865 and5090 use data storage. Extraction verifies archiveSHA and free space and preserves directory symlinks.

The runner now optionally supports --reference-native-evidence and --compress-artifacts. Exact JSON-equal native EvidenceItems are replaced by references to immutable native cache items, bound to cacheidentity, relativepath, fileSHA256 and itemindex. restore_native_evidence reconstructs every nested value and rejects mismatched cache bytes. Prediction text, token IDs, traces and decisions are unchanged. Nonmatching objects remain inline. Referenced caches must be retained locally or in a verified, resolvable master copy. Gzip changes filenames to .json.gz; both forms must be included in completion counts and exports.

The largest actual Path sample was verified: bard JSON28,200,148bytes→4,989bytes and provenance14,106,947bytes→6,597bytes, with exact full nested-value restoration. A mismatched-cache negative test passed. Completed40297/44450 legacy artifacts are being converted in place with original filenames; each replacement is checked by full restoration and originalSHA recorded in artifact-storage-migration.json.44450 snapshot572cases:8,192,102,161bytes→37,128,536bytes. No native masks/evidence were discarded. Plain output from still-running old workers may coexist with compact artifacts.

## Queue coverage and remaining dependencies

All jobs are candidate-only. LLaVA VQA-RAD remains excluded. Huatuo VQA/SLAKE and LLaVA SLAKE complete. Path active across both receivers. Full remaining MMMU is assigned through detached successors: six4090 LLaVA shards, two5090 Huatuo halves afterPath. Native/archive transfer readiness is checked before generation; an assigned queue is not proof it has begun.

PMC33430 native preparation is active onhostGPU1. MedCPT requests are partitioned exactly[0,16715)5090GPU0 and[16715,33430)hostGPU1, preserving originalcachekeys. Cloud first-half thenMIMIC retrieval queue is detached. Host native queue is PMC→MIMIC5159→Omni88995. MIMIC manifests preserve report prompts,1024-token budget and explicitreport expert task declarations; scheduledcoverage5158cases with3+groups and1with2groups, beforeactualadmission. Omni manifest includes all88995uniqueIDs/82059SHA-verifiedimages and benchmark choice prompts; freshimage-only routing is queued. PMC/MIMIC/Omni receiver staging and final evaluations are not yet complete. Do not claim the entiretwo-model all-dataset objective achieved.
