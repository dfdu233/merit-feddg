"""Deterministic capability subrequests and a lossy, auditable evidence bridge.

This is not a diagnostic predictor or a learned clinical ontology. Unknown
questions stay unknown; no answer, negative finding or ROI is invented.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from .capabilities import CAPABILITIES
from .capability_routing import question_type
from .native_evidence import _relevance, _tokens, compile_evidence


@dataclass(frozen=True)
class EvidenceNeed:
    capability: str
    scope: str
    question_type: str
    query: str
    region: None = None
    source: str = "question_only_template_v1"


def evidence_need(question, descriptor):
    """Resolve WHAT evidence is requested, without treating generated text as truth.

    Fixed-catalog/segmentation adapters may ignore query; that limitation remains
    explicit in their native provenance. This does not make them open-vocabulary.
    """
    capability = descriptor["capability"]
    if capability not in CAPABILITIES or not isinstance(question, str) or not question.strip():
        raise ValueError("a real question and registered capability are required")
    instruction = {
        "generation": "Describe only visible observations relevant to the question in at most two "
                      "short sentences. Do not infer unobservable history or give unrelated findings.",
        "classification": "Return native visual matches relevant to the question, with original "
                          "score semantics. Do not turn scores into confirmed diagnoses.",
        "segmentation": "Localize supported visible anatomical structures relevant to the question. "
                        "A predicted anatomy region is not a lesion or diagnosis.",
        "detection": "Return supported object locations relevant to the question. "
                     "Do not infer absence from a missing detection.",
        # Retrieval embeds the question, not these instructions: boilerplate can
        # dominate question similarity and is not a meaningful search query.
        "retrieval": "",
    }[capability]
    query = (instruction + "\nQuestion: " + question) if instruction else question
    return EvidenceNeed(capability, descriptor["scope"], question_type(question), query)


def scoped_items(items, question, *, top_k=2, retrieval_answers=False):
    """Prune only presentation copies. Raw scores/text/masks remain unchanged.

    Explicit question overlap takes precedence over native rank. Without a
    match retain native top-k; never infer a clinical threshold. Whole generated
    observations are retained (no substring truncation that could lose negation).
    Retrieval answers are hidden by default to test the label-copying confound;
    retrieval is then an intentionally weak question/similarity control, not RAG.
    """
    if type(top_k) is not int or top_k < 1 or type(retrieval_answers) is not bool:
        raise ValueError("top_k must be positive and retrieval_answers must be boolean")
    query = _tokens(question)
    output = []
    for item in items:
        payload = copy.deepcopy(item.payload)
        if not isinstance(payload, dict):
            continue
        for name in ("catalog", "findings", "structures", "detections", "boxes", "objects", "references"):
            entries = payload.get(name)
            if not isinstance(entries, list):
                continue
            # Native order is stable on relevance ties. No softmax, re-scaling,
            # label replacement, or disease conclusion is introduced here.
            entries = sorted(entries, key=lambda entry: -_relevance(entry, query))[:top_k]
            if name == "references" and not retrieval_answers:
                entries = [{key: value for key, value in entry.items() if key not in {
                    "source_reference", "source_answer", "answer", "reference", "references",
                }} for entry in entries if isinstance(entry, dict)]
                payload["source_answers_included"] = False
            payload[name] = entries
        output.append(replace(item, payload=payload))
    return tuple(output)


def presentation_items(items, question, config):
    if config.evidence_style == "focused":
        from .request_scope import focused_items

        items = focused_items(items)
    if config.evidence_style == "native":
        return tuple(items)
    return scoped_items(items, question, top_k=config.evidence_top_k,
                        retrieval_answers=config.retrieval_answer_context)


def evidence_memory(items, question, config):
    presented = presentation_items(items, question, config)
    if config.evidence_style == "graph":
        from .structured_evidence import compile_typed_evidence

        return compile_typed_evidence(presented, question, config.max_evidence_chars)
    return compile_evidence(presented, question, config.max_evidence_chars)
