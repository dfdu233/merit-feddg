# Huatuo source canary complete (32/32)

Same frozen source queue as LLaVA: 16 VQA-RAD TRAIN, 16 English SLAKE TRAIN. All five Huatuo arms use historical native prompts/settings and a 1024-token budget. Scores are CE correctness plus OE answer-token recall, not plain accuracy or official TEST performance.

| Source | Generalist | Joint-all | Mean | Geometric median | BARD |
|---|---:|---:|---:|---:|---:|
| SLAKE, 16 | 65.625 | 62.500 | 62.500 | 62.500 | 65.625 |
| VQA-RAD, 16 | 34.583 | 28.750 | 42.500 | 42.500 | 42.500 |

BARD improves two VQA cases, harms two, and leaves the other twelve scores unchanged; SLAKE scores are unchanged on all sixteen cases. The VQA gain is +7.917 points, but mean and geometric median achieve exactly the same aggregate score. The PA rescue includes correction of an internally inconsistent baseline answer; do not attribute it solely to new visual understanding. Wrong changes to tracheal shift and gastric-bubble location are retained in the case report. The known nonbinary CLOSED prompt issue at slake-train-4466 is preserved and disclosed.

Delivered independent fault groups: 19 one-node, 3 two-node, 1 three-node, 9 four-node cases. Pooled adversarial retention is strongly inflated by structural one-node fallback. Restricting first-eight-prefix residual stress to 3+ nodes, same-clean-token retention is mean 110/234 (47.01%), median 220/234 (94.02%), BARD 207/234 (88.46%). This is not corrupted free-generation medical accuracy; falling back to Generalist can either help or hurt. It does not establish extra robustness over geometric median.

Conclusion: a modest source net gain on VQA, baseline preservation on SLAKE, and real harmful flips. These results do not establish a BARD-specific accuracy advantage over median or reliable receiver-independent superiority. User explicitly requested full TEST next; all method parameters remain frozen.

## Runtime and exactness

All final source efficacy outputs come from cloud 5090/eager/Torch2.8. Historical token parity passes on the cloud. Host 4090 still has one historical-token mismatch even after matching Torch2.8; its efficacy outputs are excluded. Host nevertheless validates the native spatial hook and cached-versus-replay scores.

Native KV caching matches every score in all tested ordinary/spatial prefix vectors on both hosts (including shared masked vocabulary), and all 15 complete method sequences across three cloud cases. Remaining four source cases use this validated cache. Answer budget, BARD rules and spatial operator are unchanged. KV caching is opt-in in the experiment runner.

Completed selected source cases total 110,786 language-model forwards, dominated by quadratic reference replay. This excludes interrupted duplicate work, parity checks and native experts. Interrupted source workers were bounded at 200k and 100k; exact interrupted-prefix counts were not recoverable because background workers ignored SIGINT and required SIGTERM. No cost figure is presented as a global exact total.

Full case scores/texts, paired changes and stratified stress are in results.json. Large native masks remain in original inference/cache artifacts; analysis exports omit only duplicated evidence arrays after generation.
