"""Frozen semantic state/action features, without labels or target fitting."""

from __future__ import annotations

import json

import numpy as np

from .capabilities import scoped_key
from .capability_routing import question_type
from .native_evidence import compile_evidence
from .open_data import INFERENCE_FIELDS


def action_key(row, descriptor):
    return scoped_key(
        descriptor["expert"], row["modality"], row["task"],
        descriptor["capability"], descriptor["scope"],
    )


def state_kind(state):
    if state.prefix:
        return "continuation"
    return "after_tool" if state.history else "initial"


class ValueStateEncoder:
    """Reuse an existing frozen BiomedCLIP; only small ridge heads are fitted.

    A fixed random projection is data-independent. Native cosine alignment and
    projected state-by-tool interactions avoid arbitrary numeric action IDs.
    This shared encoder is NOT advertised as expert-native OOD measurement.
    """

    def __init__(self, pool, spec, *, dimensions=16, seed=17, max_tokens=96, max_calls=2):
        if dimensions < 1 or max_tokens < 1 or max_calls < 1:
            raise ValueError("feature dimensions and budgets must be positive")
        self.pool, self.spec = pool, spec
        self.dimensions, self.seed = dimensions, seed
        self.max_tokens, self.max_calls = max_tokens, max_calls
        self.projections = {}

    def _project(self, vector):
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if not np.isfinite(vector).all() or not vector.size:
            raise ValueError("encoder returned invalid features")
        size = vector.size
        if size not in self.projections:
            self.projections[size] = np.random.default_rng(self.seed).normal(
                size=(size, self.dimensions)
            ) / np.sqrt(size)
        value = vector @ self.projections[size]
        return value / max(np.linalg.norm(value), 1e-12)

    def candidates(self, row, state, descriptors, prefix_text):
        if set(row) != INFERENCE_FIELDS:
            raise ValueError("value encoder accepts label-free inference rows only")
        if not descriptors:
            return []
        image = self._project(self.pool._image_feature(self.spec, row["image"]))
        memory = compile_evidence(state.items, row["question"], max_chars=900)
        texts = [
            ("question", row["question"]),
            ("state", f"{prefix_text[-600:]} Observed: {json.dumps(memory, ensure_ascii=False)}"),
            ("context", f"{row['modality']} {row['task']} {question_type(row['question'])}"),
            *[(f"tool-{i}", f"{d['capability']} {d['scope']} {d['description']}")
              for i, d in enumerate(descriptors)],
        ]
        vectors = [self._project(v) for v in self.pool._text_vectors(self.spec, tuple(texts))]
        query, observed, context = vectors[:3]
        results = []
        for descriptor, tool in zip(descriptors, vectors[3:], strict=True):
            features = np.concatenate([
                image, query, observed, context, tool,
                image * tool, query * tool, observed * tool,
                [len(state.prefix) / self.max_tokens, len(state.history) / self.max_calls,
                 float(bool(state.items)), float(image @ tool), float(query @ tool)],
            ]).tolist()
            results.append({
                "action_key": action_key(row, descriptor), "features": features,
                "state_kind": state_kind(state), "history_actions": list(state.history),
            })
        return results
