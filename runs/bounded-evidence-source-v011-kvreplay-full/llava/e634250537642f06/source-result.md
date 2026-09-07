# Source-only bounded evidence diagnosis

Target generations: 0. Quality metric: token_f1.
Token-F1 and positive/negative groups do not establish clinical benefit or harm.

| Scope | Source domain | Strength | Arm | Groups | Mean gain |
| --- | --- | ---: | --- | ---: | ---: |
| biomed_anatomy/ct/open_vqa/classification/major_anatomy | vqarad-image-proxy-1 | 0.25 | direct_context | 1 | +0.00000 |
| biomed_anatomy/ct/open_vqa/classification/major_anatomy | vqarad-image-proxy-1 | 0.25 | format_only | 1 | +0.00000 |
| biomed_anatomy/ct/open_vqa/classification/major_anatomy | vqarad-image-proxy-1 | 0.25 | guided | 1 | +0.00000 |
| biomed_anatomy/ct/open_vqa/classification/major_anatomy | vqarad-image-proxy-1 | 0.5 | direct_context | 1 | +0.00000 |
| biomed_anatomy/ct/open_vqa/classification/major_anatomy | vqarad-image-proxy-1 | 0.5 | format_only | 1 | +0.00000 |
| biomed_anatomy/ct/open_vqa/classification/major_anatomy | vqarad-image-proxy-1 | 0.5 | guided | 1 | +0.00000 |
| biomed_anatomy/cxr/open_vqa/classification/major_anatomy | vqarad-image-proxy-0 | 0.25 | direct_context | 1 | +0.00000 |
| biomed_anatomy/cxr/open_vqa/classification/major_anatomy | vqarad-image-proxy-0 | 0.25 | format_only | 1 | +0.00000 |
| biomed_anatomy/cxr/open_vqa/classification/major_anatomy | vqarad-image-proxy-0 | 0.25 | guided | 1 | +0.00000 |
| biomed_anatomy/cxr/open_vqa/classification/major_anatomy | vqarad-image-proxy-0 | 0.5 | direct_context | 1 | +0.00000 |
| biomed_anatomy/cxr/open_vqa/classification/major_anatomy | vqarad-image-proxy-0 | 0.5 | format_only | 1 | +0.00000 |
| biomed_anatomy/cxr/open_vqa/classification/major_anatomy | vqarad-image-proxy-0 | 0.5 | guided | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | pathvqa-train-proxy-0 | 0.25 | direct_context | 1 | +0.08696 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | pathvqa-train-proxy-0 | 0.25 | format_only | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | pathvqa-train-proxy-0 | 0.25 | guided | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | pathvqa-train-proxy-0 | 0.5 | direct_context | 1 | +0.08696 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | pathvqa-train-proxy-0 | 0.5 | format_only | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | pathvqa-train-proxy-0 | 0.5 | guided | 1 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.25 | direct_context | 6 | +0.01798 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.25 | format_only | 6 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.25 | guided | 6 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.5 | direct_context | 6 | +0.01798 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.5 | format_only | 6 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-0 | 0.5 | guided | 6 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-1 | 0.25 | direct_context | 4 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-1 | 0.25 | format_only | 4 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-1 | 0.25 | guided | 4 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-1 | 0.5 | direct_context | 4 | +0.00000 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-1 | 0.5 | format_only | 4 | -0.06667 |
| chexagent_description/cxr/open_vqa/generation/cxr_description | vqarad-image-proxy-1 | 0.5 | guided | 4 | -0.06667 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.25 | direct_context | 8 | +0.00326 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.25 | format_only | 8 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.25 | guided | 8 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.5 | direct_context | 8 | +0.00326 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.5 | format_only | 8 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-0 | 0.5 | guided | 8 | +0.00000 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.25 | direct_context | 9 | +0.00190 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.25 | format_only | 9 | -0.00033 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.25 | guided | 9 | -0.00033 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.5 | direct_context | 9 | +0.00190 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.5 | format_only | 9 | -0.00033 |
| conch_tissue/pathology/open_vqa/classification/tissue_appearance | pathvqa-train-proxy-1 | 0.5 | guided | 9 | -0.00033 |
| cxr_anatomy/cxr/open_vqa/segmentation/thoracic_anatomy | vqarad-image-proxy-0 | 0.25 | direct_context | 1 | +0.00000 |
| cxr_anatomy/cxr/open_vqa/segmentation/thoracic_anatomy | vqarad-image-proxy-0 | 0.25 | format_only | 1 | +0.00000 |
| cxr_anatomy/cxr/open_vqa/segmentation/thoracic_anatomy | vqarad-image-proxy-0 | 0.25 | guided | 1 | +0.00000 |
| cxr_anatomy/cxr/open_vqa/segmentation/thoracic_anatomy | vqarad-image-proxy-0 | 0.5 | direct_context | 1 | +0.00000 |
| cxr_anatomy/cxr/open_vqa/segmentation/thoracic_anatomy | vqarad-image-proxy-0 | 0.5 | format_only | 1 | +0.00000 |
| cxr_anatomy/cxr/open_vqa/segmentation/thoracic_anatomy | vqarad-image-proxy-0 | 0.5 | guided | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | pathvqa-train-proxy-0 | 0.25 | direct_context | 1 | +0.05405 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | pathvqa-train-proxy-0 | 0.25 | format_only | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | pathvqa-train-proxy-0 | 0.25 | guided | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | pathvqa-train-proxy-0 | 0.5 | direct_context | 1 | +0.05405 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | pathvqa-train-proxy-0 | 0.5 | format_only | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | pathvqa-train-proxy-0 | 0.5 | guided | 1 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.25 | direct_context | 5 | -0.02456 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.25 | format_only | 5 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.25 | guided | 5 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.5 | direct_context | 5 | -0.02456 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.5 | format_only | 5 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-0 | 0.5 | guided | 5 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-1 | 0.25 | direct_context | 4 | +0.03333 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-1 | 0.25 | format_only | 4 | +0.00000 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-1 | 0.25 | guided | 4 | -0.06667 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-1 | 0.5 | direct_context | 4 | +0.03333 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-1 | 0.5 | format_only | 4 | -0.06667 |
| cxr_findings/cxr/open_vqa/classification/cxr_findings | vqarad-image-proxy-1 | 0.5 | guided | 4 | -0.06667 |

## Calibration (not a statistical safety guarantee)

- conch_tissue|pathology|open_vqa|classification|tissue_appearance: proxy_or_unverified_domains
- cxr_findings|cxr|open_vqa|classification|cxr_findings: proxy_or_unverified_domains
- chexagent_description|cxr|open_vqa|generation|cxr_description: proxy_or_unverified_domains
- biomed_anatomy|ct|open_vqa|classification|major_anatomy: proxy_or_unverified_domains
- biomed_anatomy|cxr|open_vqa|classification|major_anatomy: proxy_or_unverified_domains
- cxr_anatomy|cxr|open_vqa|segmentation|thoracic_anatomy: proxy_or_unverified_domains

- Empirical independent-group confirmation is not a statistical safety bound.
- Dataset domains are not necessarily independent hospitals.
- No LODO or native multi-view stability is implemented in this profile.
- Token-F1 is not clinical factuality.

Source timings reuse native evidence and are not end-to-end method latency.
