from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .med_defer import SUPPORTED_CAPABILITIES

_YES = frozenset({"yes", "y", "true", "present", "positive"})
_NO = frozenset({"no", "n", "false", "absent", "negative"})
_BINARY_QUESTIONS = (
    (re.compile(r"^is there\s+(?:any\s+)?(.+)$", re.IGNORECASE), "finding"),
    (re.compile(r"^are there\s+(?:any\s+)?(.+)$", re.IGNORECASE), "finding"),
    (
        re.compile(
            r"^does (?:the|this) (?:image|scan|study|radiograph) show\s+(.+)$", re.IGNORECASE
        ),
        "finding",
    ),
    (
        re.compile(
            r"^is (?:the|this) "
            r"(tissue|lesion|mass|tumou?r|nodule|heart|lung|liver|kidney|retina|patient) "
            r"(.+)$",
            re.IGNORECASE,
        ),
        "subject",
    ),
    (re.compile(r"^is this\s+(.+)$", re.IGNORECASE), "image_description"),
)


def _clean(text: str) -> str:
    return " ".join(str(text).strip().rstrip("?.! ").split())


def _answer_key(answer: str) -> str:
    return _clean(answer).casefold()


def _ensure_period(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _binary_statements(question: str) -> tuple[str, str]:
    """Turn a yes/no question into affirmative and negative clinical propositions."""

    clean = _clean(question)
    for pattern, kind in _BINARY_QUESTIONS:
        match = pattern.match(clean)
        if not match:
            continue
        if kind == "finding":
            finding = match.group(1)
            return (
                _ensure_period(f"The image shows {finding}"),
                _ensure_period(f"The image does not show {finding}"),
            )
        if kind == "image_description":
            description = match.group(1)
            return (
                _ensure_period(f"The image is {description}"),
                _ensure_period(f"The image is not {description}"),
            )
        subject, predicate = match.groups()
        return (
            _ensure_period(f"The {subject} is {predicate}"),
            _ensure_period(f"The {subject} is not {predicate}"),
        )

    # The fallback still contains the complete medical question.  It never sends
    # the semantically empty strings ``yes`` or ``no`` to a specialist.
    return (
        _ensure_period(f'The image supports an affirmative answer to: "{clean}"'),
        _ensure_period(f'The image supports a negative answer to: "{clean}"'),
    )


def _categorical_statement(question: str, answer: str) -> str:
    clean_question = _clean(question)
    clean_answer = _clean(answer)
    lowered = clean_question.casefold()
    finding_stems = (
        "what finding",
        "what abnormality",
        "what diagnosis",
        "what disease",
        "which finding",
        "which abnormality",
        "which diagnosis",
        "which disease",
    )
    if lowered.startswith(finding_stems):
        return _ensure_period(f"The image shows {clean_answer}")
    if lowered.startswith(("where", "which location", "what location", "which region")):
        return _ensure_period(f"The finding is located in {clean_answer}")
    if lowered.startswith(("what modality", "which modality", "what imaging")):
        return _ensure_period(f"The image modality is {clean_answer}")
    if lowered.startswith(("how many", "what number", "what count")):
        return _ensure_period(f"The image supports a count of {clean_answer}")
    return _ensure_period(
        f'For the clinical question "{clean_question}", the image-supported answer is '
        f'"{clean_answer}"'
    )


@dataclass(frozen=True)
class CandidateProposition:
    """A dataset answer paired with the medical proposition shown to experts."""

    candidate_id: str
    answer: str
    proposition: str
    polarity: str = "categorical"

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.answer.strip() or not self.proposition.strip():
            raise ValueError("candidate id, answer, and proposition cannot be empty")
        if self.polarity not in {"affirmed", "negated", "categorical", "open"}:
            raise ValueError(f"unsupported proposition polarity: {self.polarity}")


@dataclass(frozen=True)
class ClaimSpec:
    """Label-free description of one clinical decision or open clinical claim.

    ``propositions`` are the only strings that should be sent to an expert.
    Dataset answer strings are retained solely to map the result back to the VQA
    output space.  No target label is represented by this type.
    """

    claim_id: str
    question: str
    modality: str
    required_capabilities: tuple[str, ...]
    propositions: tuple[CandidateProposition, ...]
    closed_set: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.question.strip() or not self.modality.strip():
            raise ValueError("claim_id, question, and modality cannot be empty")
        if not self.required_capabilities:
            raise ValueError("a claim needs at least one required capability")
        unsupported = set(self.required_capabilities) - SUPPORTED_CAPABILITIES
        if unsupported:
            raise ValueError(f"unsupported capabilities: {sorted(unsupported)}")
        if self.closed_set and len(self.propositions) < 2:
            raise ValueError("a closed-set claim needs at least two candidates")
        if not self.closed_set and len(self.propositions) != 1:
            raise ValueError("an open claim must contain exactly one proposition")
        ids = [item.candidate_id for item in self.propositions]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")

    @property
    def expert_queries(self) -> tuple[str, ...]:
        """Medical propositions for specialist inference, never bare VQA answers."""

        return tuple(item.proposition for item in self.propositions)

    @property
    def candidate_answers(self) -> tuple[str, ...]:
        return tuple(item.answer for item in self.propositions)

    @classmethod
    def from_vqa(
        cls,
        *,
        claim_id: str,
        question: str,
        candidates: tuple[str, ...] | list[str],
        modality: str,
        required_capabilities: tuple[str, ...] = ("classification",),
        metadata: dict[str, Any] | None = None,
    ) -> ClaimSpec:
        """Build semantic propositions from a real VQA row without its label."""

        answers = tuple(_clean(value) for value in candidates)
        if len(answers) < 2 or any(not answer for answer in answers):
            raise ValueError("VQA candidates must contain at least two non-empty answers")
        keys = tuple(_answer_key(answer) for answer in answers)
        is_binary = len(answers) == 2 and set(keys) == {_answer_key("yes"), _answer_key("no")}
        propositions: list[CandidateProposition] = []
        if is_binary:
            affirmative, negative = _binary_statements(question)
            for index, (answer, key) in enumerate(zip(answers, keys)):
                positive = key in _YES
                propositions.append(
                    CandidateProposition(
                        candidate_id=f"candidate-{index}",
                        answer=answer,
                        proposition=affirmative if positive else negative,
                        polarity="affirmed" if positive else "negated",
                    )
                )
        else:
            for index, answer in enumerate(answers):
                propositions.append(
                    CandidateProposition(
                        candidate_id=f"candidate-{index}",
                        answer=answer,
                        proposition=_categorical_statement(question, answer),
                    )
                )
        return cls(
            claim_id=claim_id,
            question=_clean(question),
            modality=modality,
            required_capabilities=required_capabilities,
            propositions=tuple(propositions),
            closed_set=True,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_open_claim(
        cls,
        *,
        claim_id: str,
        claim: str,
        modality: str,
        required_capabilities: tuple[str, ...],
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ClaimSpec:
        """Describe a generated clinical claim that needs evidential support."""

        clean_claim = _ensure_period(_clean(claim))
        combined_metadata = dict(metadata or {})
        if context:
            combined_metadata["generation_context"] = context
        return cls(
            claim_id=claim_id,
            question=clean_claim,
            modality=modality,
            required_capabilities=required_capabilities,
            propositions=(
                CandidateProposition(
                    candidate_id="open-claim",
                    answer=clean_claim,
                    proposition=clean_claim,
                    polarity="open",
                ),
            ),
            closed_set=False,
            metadata=combined_metadata,
        )
