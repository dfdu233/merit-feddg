import pytest

from merit_feddg.claims import ClaimSpec
from merit_feddg.evidence_bridges import EvidenceBridgeRegistry, semantic_score_map
from merit_feddg.med_defer import NativeEvidence


def _binary_claim():
    return ClaimSpec.from_vqa(
        claim_id="path-1",
        question="Is this tissue malignant?",
        candidates=["yes", "no"],
        modality="pathology",
    )


def _evidence(claim, capability="classification", **kwargs):
    scores = {
        claim.propositions[0].proposition: 3.0,
        claim.propositions[-1].proposition: -2.0,
    }
    return NativeEvidence(
        expert_id="specialist",
        capability=capability,
        concept_scores=scores,
        confidence=0.9,
        **kwargs,
    )


def test_binary_vqa_exposes_medical_propositions_not_bare_yes_no():
    claim = _binary_claim()
    assert claim.candidate_answers == ("yes", "no")
    assert all(query.casefold() not in {"yes", "no"} for query in claim.expert_queries)
    assert "tissue is malignant" in claim.expert_queries[0].casefold()
    assert "tissue is not malignant" in claim.expert_queries[1].casefold()
    assert not hasattr(claim, "label")


def test_binary_candidate_order_is_preserved_without_using_a_label():
    claim = ClaimSpec.from_vqa(
        claim_id="cxr-1",
        question="Is there a pleural effusion?",
        candidates=["no", "yes"],
        modality="cxr",
    )
    assert claim.candidate_answers == ("no", "yes")
    assert "does not show a pleural effusion" in claim.expert_queries[0].casefold()
    assert "shows a pleural effusion" in claim.expert_queries[1].casefold()


def test_is_this_article_question_produces_a_well_formed_image_proposition():
    claim = ClaimSpec.from_vqa(
        claim_id="modality-1",
        question="Is this a chest x-ray?",
        candidates=["yes", "no"],
        modality="cxr",
    )
    assert claim.expert_queries == (
        "The image is a chest x-ray.",
        "The image is not a chest x-ray.",
    )


def test_nonbinary_real_vqa_candidates_become_semantic_propositions():
    claim = ClaimSpec.from_vqa(
        claim_id="cxr-2",
        question="What abnormality is visible?",
        candidates=["pneumothorax", "pleural effusion", "pulmonary edema"],
        modality="cxr",
    )
    assert len(claim.propositions) == 3
    assert claim.expert_queries == (
        "The image shows pneumothorax.",
        "The image shows pleural effusion.",
        "The image shows pulmonary edema.",
    )


def test_open_claim_supports_nonclassification_capabilities():
    claim = ClaimSpec.from_open_claim(
        claim_id="report-claim-3",
        claim="A 7 mm nodule is present in the right upper lobe",
        modality="ct",
        required_capabilities=("detection", "segmentation"),
        context="Findings:",
    )
    assert not claim.closed_set
    assert claim.expert_queries == ("A 7 mm nodule is present in the right upper lobe.",)
    assert claim.metadata["generation_context"] == "Findings:"


def test_classification_bridge_maps_semantic_scores_to_claim_support():
    claim = _binary_claim()
    converted = EvidenceBridgeRegistry().convert(claim, _evidence(claim))
    assert not converted.abstained
    assert converted.candidates[0].support > converted.candidates[1].support
    assert converted.candidates[0].signed_support > 0
    scores = converted.concept_scores()
    assert set(scores) == set(claim.expert_queries)


@pytest.mark.parametrize(
    ("capability", "kwargs", "expected_reason"),
    [
        (
            "retrieval",
            {"generated_text": "A source-domain report supports malignancy."},
            "retrieval-evidence",
        ),
        ("segmentation", {"masks": ({"area_fraction": 0.2},)}, "segmentation-evidence"),
        ("detection", {"boxes": ((0.1, 0.2, 0.7, 0.8),)}, "detection-evidence"),
        (
            "generation",
            {"generated_text": "The specialist describes malignant glands."},
            "generation-evidence",
        ),
    ],
)
def test_heterogeneous_bridges_preserve_native_provenance(capability, kwargs, expected_reason):
    base = _binary_claim()
    claim = ClaimSpec(
        claim_id=base.claim_id,
        question=base.question,
        modality=base.modality,
        required_capabilities=(capability,),
        propositions=base.propositions,
        closed_set=True,
    )
    converted = EvidenceBridgeRegistry().convert(claim, _evidence(claim, capability, **kwargs))
    assert not converted.abstained
    assert converted.reason == expected_reason
    if capability in {"segmentation", "detection"}:
        assert converted.candidates[0].spatial
    if capability in {"retrieval", "generation"}:
        assert converted.candidates[0].rationale


def test_missing_bridge_and_unrequested_capability_fail_closed():
    claim = _binary_claim()
    evidence = _evidence(claim)
    no_bridges = EvidenceBridgeRegistry(include_defaults=False)
    missing = no_bridges.convert(claim, evidence)
    assert missing.abstained
    assert missing.reason == "no-evidence-bridge:classification"
    assert set(missing.concept_scores().values()) == {0.0}

    generation = NativeEvidence(
        expert_id="generator",
        capability="generation",
        concept_scores={query: 1.0 for query in claim.expert_queries},
        confidence=0.8,
        generated_text="A report.",
    )
    mismatch = EvidenceBridgeRegistry().convert(claim, generation)
    assert mismatch.abstained
    assert mismatch.reason == "capability-not-requested"


def test_bare_answer_scores_are_rejected_instead_of_silently_bridged():
    claim = _binary_claim()
    evidence = NativeEvidence(
        expert_id="unsafe-adapter",
        capability="classification",
        concept_scores={"yes": 10.0, "no": -10.0},
        confidence=1.0,
    )
    converted = EvidenceBridgeRegistry().convert(claim, evidence)
    assert converted.abstained
    assert converted.reason == "no-semantic-proposition-score"
    assert set(semantic_score_map(claim, evidence).values()) == {0.0}


def test_spatial_bridges_fail_closed_when_native_output_is_missing():
    base = _binary_claim()
    claim = ClaimSpec(
        claim_id=base.claim_id,
        question=base.question,
        modality=base.modality,
        required_capabilities=("segmentation",),
        propositions=base.propositions,
        closed_set=True,
    )
    evidence = _evidence(claim, "segmentation")
    converted = EvidenceBridgeRegistry().convert(claim, evidence)
    assert converted.abstained
    assert converted.reason == "segmentation-without-mask"


def test_native_spatial_output_can_exist_without_fake_classification_scores():
    claim = ClaimSpec.from_open_claim(
        claim_id="seg-1",
        claim="The lesion occupies the right lobe",
        modality="ct",
        required_capabilities=("segmentation",),
    )
    evidence = NativeEvidence(
        expert_id="segmenter",
        capability="segmentation",
        concept_scores={},
        confidence=0.8,
        masks=({"rle": "native-mask"},),
    )
    converted = EvidenceBridgeRegistry().convert(claim, evidence)
    assert converted.abstained
    assert converted.reason == "no-semantic-proposition-score"


def test_spatial_adapter_can_link_native_mask_to_a_claim_without_fake_logits():
    claim = ClaimSpec.from_open_claim(
        claim_id="seg-2",
        claim="The lesion occupies the right lobe",
        modality="ct",
        required_capabilities=("segmentation",),
    )
    evidence = NativeEvidence(
        expert_id="segmenter",
        capability="segmentation",
        concept_scores={},
        confidence=0.8,
        masks=({"rle": "native-mask"},),
        provenance={
            "candidate_support": {"open-claim": 0.9},
            "score_semantics": "probability",
        },
    )
    converted = EvidenceBridgeRegistry().convert(claim, evidence)
    assert not converted.abstained
    assert converted.candidates[0].support == pytest.approx(0.9)
    assert converted.candidates[0].spatial[0].kind == "mask"


def test_closed_set_bridge_rejects_partial_candidate_coverage():
    claim = _binary_claim()
    evidence = NativeEvidence(
        expert_id="partial",
        capability="classification",
        concept_scores={claim.expert_queries[0]: 2.0},
        confidence=0.9,
    )
    converted = EvidenceBridgeRegistry().convert(claim, evidence)
    assert converted.abstained
    assert converted.reason == "incomplete-semantic-coverage"


@pytest.mark.parametrize("semantics", ["probabilty", "signed", "unknown"])
def test_bridge_rejects_unknown_score_semantics(semantics):
    claim = _binary_claim()
    evidence = NativeEvidence(
        expert_id="invalid",
        capability="classification",
        concept_scores={query: 0.5 for query in claim.expert_queries},
        confidence=0.9,
        provenance={"score_semantics": semantics},
    )
    converted = EvidenceBridgeRegistry().convert(claim, evidence)
    assert converted.abstained
    assert converted.reason == "invalid-native-evidence"


def test_bridge_rejects_out_of_range_probability_and_invalid_box():
    claim = _binary_claim()
    invalid_probability = NativeEvidence(
        expert_id="invalid",
        capability="classification",
        concept_scores={query: 2.0 for query in claim.expert_queries},
        confidence=0.9,
        provenance={"score_semantics": "probability"},
    )
    assert EvidenceBridgeRegistry().convert(claim, invalid_probability).abstained

    detection_claim = ClaimSpec(
        claim_id=claim.claim_id,
        question=claim.question,
        modality=claim.modality,
        required_capabilities=("detection",),
        propositions=claim.propositions,
        closed_set=True,
    )
    invalid_box = NativeEvidence(
        expert_id="invalid",
        capability="detection",
        concept_scores={query: 0.5 for query in claim.expert_queries},
        confidence=0.9,
        boxes=((float("nan"), 0.0, 1.0, 1.0),),
        provenance={"score_semantics": "probability"},
    )
    converted = EvidenceBridgeRegistry().convert(detection_claim, invalid_box)
    assert converted.abstained
    assert converted.reason == "invalid-native-evidence"
