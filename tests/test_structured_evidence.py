"""Protocol tests for typed evidence presentation and pre-commit verification."""

import pytest

from merit_feddg.capabilities import EvidenceItem
from merit_feddg.claim_verification import (
    ClaimCommitVerifier,
    EvidenceCertificate,
    VerificationConfig,
)
from merit_feddg.claims import ClaimSpec
from merit_feddg.evidence_bridges import EvidenceBridgeRegistry
from merit_feddg.med_defer import NativeEvidence
from merit_feddg.structured_evidence import compile_typed_evidence


def item(capability, payload, *, expert="expert", confidence=None):
    return EvidenceItem(
        "evidence-1", expert, capability, "scope", payload, confidence=confidence
    )


def test_classification_graph_preserves_semantics_without_inventing_absence():
    result = compile_typed_evidence(
        [
            item(
                "classification",
                {
                    "findings": [
                        {"finding": "Pleural Effusion", "score": 0.8},
                        {"finding": "Mass", "score": 0.1},
                    ],
                    "score_semantics": "uncalibrated_independent_sigmoid",
                },
            )
        ],
        "Is there pleural effusion?",
        5000,
    )
    observations = result[0]["payload"]["observations"]
    assert observations[0]["observation"] == "Pleural Effusion"
    assert observations[0]["polarity"] == "unknown"
    assert "uncalibrated" in observations[0]["score_semantics"]
    assert "absent" not in str(observations).lower()
    assert any("does not establish absence" in value for value in result[0]["payload"]["limitations"])


def test_spatial_graph_keeps_anatomy_distinct_from_lesion():
    result = compile_typed_evidence(
        [
            item(
                "segmentation",
                {
                    "structures": [
                        {
                            "anatomical_structure": "Right Lung",
                            "bbox_xyxy_normalized_original_image": [0.5, 0.1, 0.9, 0.9],
                            "foreground_fraction_of_crop": 0.3,
                        }
                    ]
                },
            )
        ],
        "Where is the opacity?",
        5000,
    )
    node = result[0]["payload"]["observations"][0]
    assert node["kind"] == "anatomy_region"
    assert node["bbox_xyxy_normalized"] == [0.5, 0.1, 0.9, 0.9]
    assert any("do not establish a diagnosis" in value for value in result[0]["payload"]["limitations"])


def test_detection_graph_preserves_native_box_without_calling_it_truth():
    result = compile_typed_evidence(
        [
            item(
                "detection",
                {
                    "detections": [
                        {
                            "finding": "opacity",
                            "score": 0.7,
                            "bbox_xyxy_normalized": [0.2, 0.2, 0.6, 0.7],
                        }
                    ],
                    "score_semantics": "uncalibrated_detector_score",
                },
            )
        ],
        "Where is the opacity?",
        5000,
    )
    node = result[0]["payload"]["observations"][0]
    assert node["kind"] == "detected_region"
    assert node["polarity"] == "predicted_region"
    assert node["score_semantics"] == "uncalibrated_detector_score"


def test_retrieval_graph_never_relabels_source_case_as_query_evidence():
    result = compile_typed_evidence(
        [
            item(
                "retrieval",
                {
                    "references": [
                        {
                            "source_id": "source-1",
                            "source_question": "What finding?",
                            "source_reference": "pneumonia",
                            "similarity": 0.9,
                        }
                    ]
                },
            )
        ],
        "What finding?",
        5000,
    )
    node = result[0]["payload"]["observations"][0]
    assert node["applies_to"] == "source_image_only"
    assert node["polarity"] == "not_query_evidence"


def _claim(*, negative=False, spatial=False):
    candidates = ["no", "yes"] if negative else ["yes", "no"]
    claim = ClaimSpec.from_vqa(
        claim_id="claim-1",
        question="Is there a pleural effusion?",
        candidates=candidates,
        modality="cxr",
        metadata={"spatial_required": spatial},
    )
    candidate = next(
        proposition for proposition in claim.propositions
        if proposition.polarity == ("negated" if negative else "affirmed")
    )
    return claim, candidate


def _converted(claim, *, support=0.9, contradiction=0.1, spatial=False, exhaustive=False):
    candidate = claim.propositions[0]
    other = claim.propositions[1]
    evidence = NativeEvidence(
        expert_id="cxr-expert",
        capability="classification",
        concept_scores={candidate.proposition: support, other.proposition: contradiction},
        confidence=1.0,
        boxes=((0.1, 0.1, 0.8, 0.8),) if spatial else (),
        provenance={
            "score_semantics": "probability",
            "scope": "cxr_findings",
            "coverage_semantics": "exhaustive" if exhaustive else "local",
            "evidence_id": "evidence-1",
        },
    )
    return EvidenceBridgeRegistry().convert(claim, evidence)


def certificate(qualified=True):
    return EvidenceCertificate(
        expert_id="cxr-expert",
        capability="classification",
        scope="cxr_findings",
        qualified=qualified,
        source_only=True,
        lower_content_gain=0.05 if qualified else 0.0,
        domains=("hospital-a", "hospital-b"),
        domain_kinds=("hospital",),
    )


def test_precommit_requires_certificate_and_spatial_support_when_requested():
    claim, candidate = _claim(spatial=True)
    evidence = _converted(claim)
    verifier = ClaimCommitVerifier()
    no_card = verifier.verify(claim, candidate.candidate_id, [evidence])
    assert no_card.action == "ABSTAIN" and "qualified" in no_card.reason
    no_region = verifier.verify(
        claim, candidate.candidate_id, [evidence], {certificate().key: certificate()}
    )
    assert no_region.action == "ABSTAIN" and no_region.reason == "missing-spatial-support"


def test_precommit_commits_supported_claim_and_revises_contradiction():
    claim, candidate = _claim()
    card = certificate()
    verifier = ClaimCommitVerifier(VerificationConfig(min_support=0.5))
    supported = verifier.verify(
        claim, candidate.candidate_id, [_converted(claim)], {card.key: card}
    )
    assert supported.action == "COMMIT"
    contradicted = verifier.verify(
        claim,
        candidate.candidate_id,
        [_converted(claim, support=0.1, contradiction=0.9)],
        {card.key: card},
    )
    assert contradicted.action == "REVISE"


def test_negative_claim_needs_exhaustive_coverage_not_missing_detection():
    claim, candidate = _claim(negative=True)
    card = certificate()
    local = ClaimCommitVerifier().verify(
        claim, candidate.candidate_id, [_converted(claim)], {card.key: card}
    )
    assert local.action == "ABSTAIN"
    assert local.reason == "negative-claim-without-exhaustive-coverage"
    exhaustive = ClaimCommitVerifier().verify(
        claim,
        candidate.candidate_id,
        [_converted(claim, exhaustive=True)],
        {card.key: card},
    )
    assert exhaustive.action == "COMMIT"


def test_proxy_domain_cannot_form_qualified_certificate():
    with pytest.raises(ValueError, match="real source domains"):
        EvidenceCertificate(
            expert_id="expert",
            capability="classification",
            scope="scope",
            qualified=True,
            source_only=True,
            lower_content_gain=0.1,
            domains=("proxy-a", "proxy-b"),
            domain_kinds=("proxy",),
        )
