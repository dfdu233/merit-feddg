# Source-only bounded evidence diagnosis

Target generations: 0. Quality metric: token_f1.
Token-F1 and positive/negative groups do not establish clinical benefit or harm.

| Scope | Source domain | Strength | Arm | Groups | Mean gain |
| --- | --- | ---: | --- | ---: | ---: |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.25 | direct_context | 1 | -0.03497 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.25 | format_only | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.25 | guided | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.5 | direct_context | 1 | -0.03497 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.5 | format_only | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.5 | guided | 1 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.25 | direct_context | 1 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.25 | format_only | 1 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.25 | guided | 1 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.5 | direct_context | 1 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.5 | format_only | 1 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.5 | guided | 1 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.25 | direct_context | 2 | +0.04348 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.25 | format_only | 2 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.25 | guided | 2 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.5 | direct_context | 2 | +0.04348 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.5 | format_only | 2 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.5 | guided | 2 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.25 | direct_context | 1 | +0.07719 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.25 | format_only | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.25 | guided | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.5 | direct_context | 1 | +0.07719 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.5 | format_only | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.5 | guided | 1 | +0.00000 |

## Calibration (not a statistical safety guarantee)

- conch_tissue|pathology|open_vqa|classification|tissue_appearance: proxy_or_unverified_domains
- cxr_findings|cxr|open_vqa|classification|cxr_findings: proxy_or_unverified_domains
- chexagent_description|cxr|open_vqa|generation|cxr_description: proxy_or_unverified_domains

- Empirical independent-group confirmation is not a statistical safety bound.
- Dataset domains are not necessarily independent hospitals.
- No LODO or native multi-view stability is implemented in this profile.
- Token-F1 is not clinical factuality.

Source timings reuse native evidence and are not end-to-end method latency.
