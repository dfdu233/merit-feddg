"""Unified atomic-claim transactions for VQA and report generation.

Cross-task semantics follow the atomic-fact/decontextualized-claim literature.
Radiology reports can optionally use RadGraph-XL grounding, but the transaction
engine itself is domain agnostic.

Crucially, the incumbent text is immutable: rejected transactions cannot alter
its decoding trajectory or untouched spans.
"""

from dataclasses import dataclass, field
from itertools import pairwise
import re
from typing import Any


_SENTENCE = re.compile(r"[^\n.!?]+(?:[.!?]+|$)")


@dataclass(frozen=True)
class AtomicClinicalClaim:
    claim_id: str
    proposition: str
    source_span: tuple[int, int] | None
    grounding: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.claim_id or not self.proposition.strip():
            raise ValueError("atomic claim id/proposition cannot be empty")
        if self.source_span is not None:
            start, end = self.source_span
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise ValueError("source_span must be a valid half-open character range")


@dataclass(frozen=True)
class ClaimTransaction:
    transaction_id: str
    operation: str
    proposed_claim: AtomicClinicalClaim
    replacement_text: str
    baseline_claim_id: str | None = None
    proposer_expert_id: str | None = None

    def __post_init__(self):
        if self.operation not in {"ADD", "DELETE", "REPLACE"}:
            raise ValueError("transaction operation must be ADD/DELETE/REPLACE")
        if not self.transaction_id:
            raise ValueError("transaction id cannot be empty")
        if self.operation in {"DELETE", "REPLACE"} and not self.baseline_claim_id:
            raise ValueError("DELETE/REPLACE require a baseline claim")
        if self.operation == "ADD" and self.baseline_claim_id is not None:
            raise ValueError("ADD cannot target a baseline claim")
        if self.operation != "DELETE" and not self.replacement_text.strip():
            raise ValueError("ADD/REPLACE require replacement text")


@dataclass(frozen=True)
class TransactionDecision:
    transaction_id: str
    commit: bool
    reason: str
    verifier_fault_groups: tuple[str, ...] = ()
    patient_specific_support: bool = False
    source_qualified_support: bool = False
    differential_effect: float | None = None

    def __post_init__(self):
        if not self.transaction_id or not self.reason:
            raise ValueError("transaction decision identity/reason cannot be empty")


def decontextualize_vqa(question, answer):
    """Turn a short answer into a self-contained proposition without target labels."""
    q = " ".join(str(question).strip().rstrip("?.! ").split())
    a = " ".join(str(answer).strip().rstrip("?.! ").split())
    if not q or not a:
        raise ValueError("question and answer cannot be empty")
    key = a.casefold()
    finding_patterns = (
        re.compile(r"^is there\s+(?:any\s+)?(.+)$", re.IGNORECASE),
        re.compile(r"^are there\s+(?:any\s+)?(.+)$", re.IGNORECASE),
        re.compile(r"^does (?:the|this) (?:image|scan|study|radiograph) show\s+(.+)$", re.IGNORECASE),
    )
    for pattern in finding_patterns:
        match = pattern.match(q)
        if match and key in {"yes", "no"}:
            finding = match.group(1)
            verb = "shows" if key == "yes" else "does not show"
            return f"The image {verb} {finding}."
    if q.casefold().startswith(
        ("what finding", "what abnormality", "what diagnosis", "what disease",
         "which finding", "which abnormality", "which diagnosis", "which disease")
    ):
        return f"The image shows {a}."
    if q.casefold().startswith(("where", "which location", "what location", "which region")):
        return f"The finding is located in {a}."
    return f'For the clinical question "{q}", the image-supported answer is "{a}".'


def claimize_vqa(question, answer):
    answer_text = str(answer)
    if not answer_text.strip():
        raise ValueError("VQA baseline answer cannot be empty")
    return (
        AtomicClinicalClaim(
            claim_id="answer-0",
            proposition=decontextualize_vqa(question, answer_text),
            source_span=(0, len(answer_text)),
            grounding=None,
            metadata={"task": "vqa", "atomicity": "single_answer_proposition"},
        ),
    )


def sentence_spans(text):
    spans = []
    for match in _SENTENCE.finditer(text):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            spans.append((match.start() + left, match.start() + right))
    return spans


def _radgraph_proposition(annotation):
    observation = str(annotation.get("observation", "")).strip()
    if not observation:
        raise ValueError("RadGraph observation cannot be empty")
    tags = [str(value).casefold() for value in annotation.get("tags", [])]
    tag = tags[0] if tags else "definitely present"
    if "absent" in tag:
        sentence = f"The image does not show {observation}"
    elif "uncertain" in tag:
        sentence = f"The image may show {observation}"
    else:
        sentence = f"The image shows {observation}"
    located = [str(value).strip() for value in annotation.get("located_at", []) if str(value).strip()]
    if located:
        sentence += " at " + ", ".join(located)
    suggested = [str(value).strip() for value in annotation.get("suggestive_of", []) if str(value).strip()]
    if suggested:
        sentence += "; this observation is suggestive of " + ", ".join(suggested)
    return sentence.rstrip(".") + "."


class RadGraphClaimizer:
    """Optional RadGraph-XL grounding for radiology report atomic claims."""

    def __init__(self, model_type="modern-radgraph-xl"):
        try:
            from radgraph import RadGraph, get_radgraph_processed_annotations
        except ImportError as exc:
            raise RuntimeError(
                "Radiology report claimization requires the official radgraph package"
            ) from exc
        self.model = RadGraph(model_type=model_type)
        self.process = get_radgraph_processed_annotations
        self.model_type = model_type

    def __call__(self, report):
        if not isinstance(report, str) or not report.strip():
            raise ValueError("radiology report cannot be empty")
        claims = []
        claim_index = 0
        for sentence_index, (start, end) in enumerate(sentence_spans(report)):
            sentence = report[start:end]
            annotations = self.model([sentence])
            processed = self.process(annotations).get("processed_annotations", [])
            for annotation in processed:
                observation = str(annotation.get("observation", "")).strip()
                if not observation:
                    continue
                claims.append(
                    AtomicClinicalClaim(
                        claim_id=f"report-{claim_index}",
                        proposition=_radgraph_proposition(annotation),
                        source_span=(start, end),
                        grounding={
                            "schema": "radgraph-xl",
                            "model_type": self.model_type,
                            "observation": observation,
                            "located_at": list(annotation.get("located_at", [])),
                            "suggestive_of": list(annotation.get("suggestive_of", [])),
                            "tags": list(annotation.get("tags", [])),
                        },
                        metadata={
                            "task": "report_generation",
                            "sentence_index": sentence_index,
                            "atomicity": "observation_centered_radgraph_subgraph",
                        },
                    )
                )
                claim_index += 1
        return tuple(claims)


def apply_transactions(baseline_text, baseline_claims, transactions, decisions):
    """Apply only committed non-overlapping edits to an immutable baseline."""
    baseline_text = str(baseline_text)
    baseline_claims = tuple(baseline_claims)
    transactions = tuple(transactions)
    decisions = tuple(decisions)
    claims = {claim.claim_id: claim for claim in baseline_claims}
    if len(claims) != len(baseline_claims):
        raise ValueError("baseline claim ids must be unique")
    decision_map = {decision.transaction_id: decision for decision in decisions}
    if len(decision_map) != len(decisions):
        raise ValueError("transaction decisions must be unique")

    edits = []
    committed = []
    for transaction in transactions:
        decision = decision_map.get(transaction.transaction_id)
        if decision is None or not decision.commit:
            continue
        if not (
            decision.patient_specific_support
            and decision.source_qualified_support
            and decision.verifier_fault_groups
        ):
            raise ValueError(
                "committed transaction requires patient-specific, source-qualified support"
            )
        if transaction.operation == "ADD":
            position = len(baseline_text)
            replacement = (
                ("" if not baseline_text or baseline_text.endswith((" ", "\n")) else " ")
                + transaction.replacement_text.strip()
            )
            edits.append((position, position, replacement, transaction.transaction_id))
        else:
            claim = claims.get(transaction.baseline_claim_id)
            if claim is None or claim.source_span is None:
                raise ValueError("transaction target has no patchable baseline span")
            start, end = claim.source_span
            replacement = "" if transaction.operation == "DELETE" else transaction.replacement_text
            edits.append((start, end, replacement, transaction.transaction_id))
        committed.append(transaction.transaction_id)

    edits.sort(key=lambda value: (value[0], value[1]))
    for left, right in pairwise(edits):
        if right[0] < left[1]:
            raise ValueError(
                "committed claim edits overlap; merge same-sentence report claims first"
            )

    result = baseline_text
    for start, end, replacement, _transaction_id in reversed(edits):
        result = result[:start] + replacement + result[end:]
    return {
        "text": result,
        "baseline_text": baseline_text,
        "committed_transactions": committed,
        "fallback_exact": not committed and result == baseline_text,
        "untouched_baseline_preserved": all(
            result[:start] == baseline_text[:start]
            for start, _end, _replacement, _id in edits[:1]
        ) if edits else True,
    }
