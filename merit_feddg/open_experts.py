"""Lazy native evidence plugins and cached contrastive image features per case."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .experts.base import load_rgb, null_image_like
from .extract import _expert_from_spec
from .med_defer import NativeEvidence


class OpenExpertPool:
    def __init__(self, specs: dict, artifacts):
        self.specs, self.artifacts = specs, artifacts
        self.models, self.features = {}, {}

    def reset_case(self):
        self.features.clear()

    def evidence_function(self, name: str, image):
        spec = self.specs[name]

        def infer(claim, prefix):
            if name not in self.models:
                if spec.get("checkpoint_path"):
                    local_spec = {**spec, "id": str(Path(spec["checkpoint_path"]).resolve())}
                    self.models[name] = _expert_from_spec(local_spec, None)
                else:
                    self.models[name] = _expert_from_spec(spec, self.artifacts)
            model = self.models[name]
            capability = claim.required_capabilities[0]
            if hasattr(model, "infer_claims"):
                # True segmentation/detection/retrieval/generation integrations use
                # this path; no forced conversion of a mask into a diagnosis label.
                return model.infer_claims(image=image, claim=claim, generated_prefix=prefix)
            if capability != "classification":
                raise ValueError("non-classification adapters must implement infer_claims")
            queries = [p.proposition for p in claim.propositions]
            if spec.get("adapter") in {"contrastive_conch", "contrastive_biomedclip"}:
                if name not in self.features:
                    rgb = load_rgb(image)
                    self.features[name] = model._image_embedding(rgb) - model._image_embedding(
                        null_image_like(rgb)
                    )
                with model.torch.inference_mode():
                    if spec["adapter"] == "contrastive_conch":
                        tokens = model.tokenize(model.tokenizer, queries).to(model.device)
                        text = model.model.encode_text(tokens, normalize=True)
                    else:
                        text = model._text_embeddings(queries)
                    # Fixed temperature, recorded in config; never target fitted.
                    scores = (self.features[name] @ text.T).squeeze(0).float().cpu().numpy()
                    scores = scores / float(spec.get("temperature", 0.07))
            else:
                scores = model.score_claims(image, claim.question, prefix, queries)
            scores = np.asarray(scores, dtype=float)
            if scores.shape != (len(queries),) or not np.isfinite(scores).all():
                raise ValueError("invalid expert candidate scores")
            return NativeEvidence(
                expert_id=name,
                capability=capability,
                concept_scores=dict(zip(queries, scores.tolist(), strict=True)),
                confidence=1.0,
                provenance={
                    "score_semantics": "logit",
                    "prefix": prefix,
                    "adapter": spec.get("adapter"),
                    "image_minus_null": True,
                },
            )

        return infer
