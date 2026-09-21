# LLaVA-Med Adaptive BARD source canary: complete

32 official TRAIN cases (16 VQA-RAD, 16 English SLAKE), fixed hash sampling, five arms, frozen method. Final paired scores all use cloud RTX5090 with torch2.8+cu128. Both host GPUs prepared/validated native experts and receiver paths; slow host replay was superseded by complete cloud reruns. Old TEST queues stayed paused.

| Source dataset | Generalist | Joint all | Isolated mean | Isolated geometric median | Adaptive BARD |
|---|---:|---:|---:|---:|---:|
| VQA-RAD, n=16 | 45.83 | 35.42 | 45.83 | 52.08 | 52.08 |
| SLAKE, n=16 | 65.63 | 53.13 | 53.13 | 53.13 | 65.63 |
| Combined, n=32 | 55.73 | 44.27 | 49.48 | 52.60 | 58.85 |

These percentages are the existing mixed CE correctness / OE answer-token-recall score, **not official TEST accuracy**. The source queue is small and cannot establish statistical or SOTA superiority. One SLAKE source annotation is CLOSED despite a nonbinary answer (slake-train-4466); the frozen yes/no prompt is therefore inappropriate for that case and is a protocol limitation.

BARD improves one case, harms zero and ties 31 on this score. Relative to geometric median, it removes all four harmful changes but retains only one of two helpful changes. It is a conservative safeguard, not a demonstrated general rescue mechanism.

Actual delivered groups: n=1 in 19 cases, n=2 in 3, n=3 in 8, n=4 in 2. Native spatial records reached 12 cases. Planned 21 pair / 11 robust cases was optimistic: BiomedParse rejected 20 requests for unknown anatomy/sequence, and packing can further omit evidence. MedCPT used a real 851839-document PubMed chunk, returning three documents per input.

For single-node residual sign reversal on the first eight clean BARD prefixes, clean-decision retention conditional on n>=2 was mean 216/304 (71.05%), geometric median 279/304 (91.78%), BARD 291/304 (95.72%). Conditional on n>=3, it was mean 184/256 (71.88%), geometric median 247/256 (96.48%), BARD 246/256 (96.09%). Thus robust aggregation beats mean in this limited stress test, but BARD does not improve over geometric median in the >=3-group stratum. Pooled token counts are correlated diagnostics, not independent samples or clinical error rates.

Retrieval removal changed only 2 shadow BARD token decisions in the first-eight-prefix diagnostics. The one clear rescue came from two visual branches overcoming an opposing retrieval branch; it does not establish positive retrieval value. See [mechanism analysis](method-analysis.md) and [all scores and paired cases](results.json).

No production persistent-KV switch and no full TEST expansion were performed. A host parity attempt incorrectly included a terminal EOS in the prefix, so it queried beyond the completed sequence and stopped; this is an invalid probe, not evidence of a cache-parity mismatch. Any subsequent valid parity check must exclude terminal EOS.

Final cloud replay/stress workload: 47078 language-model forwards. Interrupted/superseded host receivers used another 15193. Native expert inference, routing and technical checks are outside this count. Raw cloud outputs retain the full evidence; downloaded analysis exports omit duplicate arrays and retain each raw file SHA256.
