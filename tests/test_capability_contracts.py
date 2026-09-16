from merit_feddg.capabilities import tool_descriptors
from merit_feddg.capability_contracts import (
    assess_authority,
    authority_contract,
    question_semantics,
)


def finding_spec():
    return {
        "id": "xrv/findings",
        "modalities": ["cxr"],
        "tasks": ["open_vqa"],
        "capabilities": ["classification"],
        "scope": "cxr_findings",
        "authority_contract": {
            "schema": "native-authority-v1",
            "native_variable": {
                "entity_type": "chest_xray_finding",
                "attribute": "presence",
                "values": ["present", "absent"],
                "output_semantics": "independent_sigmoid_score",
            },
            "supports": ["finding_presence"],
            "requires_entity_match_for": ["finding_presence"],
            "entity_aliases": {
                "Pneumothorax": [],
                "Effusion": ["pleural effusion"],
                "Cardiomegaly": ["enlarged heart"],
            },
            "forbids": ["laterality", "location", "measurement", "finding_identity"],
        },
    }


def anatomy_spec():
    return {
        "id": "xrv/anatomy",
        "modalities": ["cxr"],
        "tasks": ["open_vqa"],
        "capabilities": ["segmentation"],
        "scope": "thoracic_anatomy",
        "authority_contract": {
            "schema": "native-authority-v1",
            "native_variable": {
                "entity_type": "thoracic_anatomy",
                "attribute": "spatial_extent",
                "values": ["soft_mask"],
                "output_semantics": "per_structure_soft_mask",
            },
            "supports": ["anatomy_identity", "location", "laterality", "relative_extent"],
            "requires_entity_match_for": ["location", "laterality", "relative_extent"],
            "entity_aliases": {
                "Left Lung": [],
                "Right Lung": [],
                "Heart": ["cardiac silhouette"],
            },
            "forbids": ["finding_presence", "finding_identity", "measurement"],
        },
    }


def row(question):
    return {"question": question, "modality": "cxr", "task": "open_vqa"}


def test_contract_roundtrip_is_machine_readable():
    contract = authority_contract("cxr_findings", finding_spec(), "classification")
    data = contract.to_json()
    assert data["schema"] == "native-authority-v1"
    assert data["native_variable"]["attribute"] == "presence"
    assert data["supports"] == ["finding_presence"]
    assert data["requires_entity_match_for"] == ["finding_presence"]
    assert data["entity_aliases"]["Effusion"] == ["pleural effusion"]


def test_question_semantics_can_be_multidimensional():
    assert question_semantics("Is there a left pneumothorax?") == frozenset(
        {"finding_presence", "laterality"}
    )


def test_classifier_exact_presence_is_safe_for_global_transport():
    audit = assess_authority(
        "Is there a pneumothorax?", finding_spec(), "classification", expert_id="xrv"
    )
    assert audit["status"] == "exact"
    assert audit["authorized_dimensions"] == ["finding_presence"]
    assert audit["matched_entities"] == ["Pneumothorax"]
    assert audit["global_transport_allowed"]
    assert audit["local_intervention_allowed"]
    assert not audit["reliability_estimated"]


def test_classifier_alias_matches_native_entity():
    audit = assess_authority(
        "Is there a pleural effusion?", finding_spec(), "classification", expert_id="xrv"
    )
    assert audit["status"] == "exact"
    assert audit["matched_entities"] == ["Effusion"]


def test_classifier_rejects_unmodeled_finding_even_for_presence_question():
    audit = assess_authority(
        "Is there tuberculosis?", finding_spec(), "classification", expert_id="xrv"
    )
    assert audit["status"] == "denied"
    assert audit["reason"] == "native_entity_not_declared"
    assert audit["entity_unmatched_dimensions"] == ["finding_presence"]


def test_classifier_partial_laterality_cannot_enter_global_text_channel():
    audit = assess_authority(
        "Is there a left pneumothorax?", finding_spec(), "classification", expert_id="xrv"
    )
    assert audit["status"] == "partial"
    assert audit["authorized_dimensions"] == ["finding_presence"]
    assert audit["unsupported_dimensions"] == ["laterality"]
    assert not audit["global_transport_allowed"]
    assert audit["local_intervention_allowed"]


def test_classifier_location_question_is_denied():
    audit = assess_authority(
        "Where is the pneumothorax located?", finding_spec(), "classification", expert_id="xrv"
    )
    assert audit["status"] == "denied"
    assert not audit["local_intervention_allowed"]


def test_anatomy_mask_can_support_location_and_laterality():
    audit = assess_authority(
        "Where is the left lung located?", anatomy_spec(), "segmentation", expert_id="anatomy"
    )
    assert audit["status"] == "exact"
    assert set(audit["authorized_dimensions"]) == {"laterality", "location"}
    assert audit["matched_entities"] == ["Left Lung"]
    assert audit["global_transport_allowed"]


def test_anatomy_mask_rejects_location_for_unmodeled_structure():
    audit = assess_authority(
        "Where is the liver located?", anatomy_spec(), "segmentation", expert_id="anatomy"
    )
    assert audit["status"] == "denied"
    assert audit["reason"] == "native_entity_not_declared"


def test_anatomy_identity_can_use_declared_catalog_without_named_entity():
    audit = assess_authority(
        "What organ is visible?", anatomy_spec(), "segmentation", expert_id="anatomy"
    )
    assert audit["status"] == "exact"
    assert audit["authorized_dimensions"] == ["anatomy_identity"]


def test_anatomy_mask_cannot_establish_disease_presence():
    audit = assess_authority(
        "Is there pneumonia?", anatomy_spec(), "segmentation", expert_id="anatomy"
    )
    assert audit["status"] == "denied"
    assert not audit["global_transport_allowed"]


def test_anatomy_mask_does_not_claim_physical_measurement():
    audit = assess_authority(
        "How large is the heart in centimeters?", anatomy_spec(), "segmentation", expert_id="anatomy"
    )
    assert audit["status"] == "denied"


def test_tool_descriptors_enforce_declared_global_authority():
    specs = {"finding": finding_spec(), "anatomy": anatomy_spec()}
    exact = tool_descriptors(specs, row("Is there a pneumothorax?"))
    assert [item["expert"] for item in exact] == ["finding"]
    assert exact[0]["authority"]["status"] == "exact"

    composite = tool_descriptors(specs, row("Is there a left pneumothorax?"))
    assert composite == []

    spatial = tool_descriptors(specs, row("Where is the left lung located?"))
    assert [item["expert"] for item in spatial] == ["anatomy"]

    unsupported = tool_descriptors(specs, row("Is there tuberculosis?"))
    assert unsupported == []


def test_undeclared_legacy_expert_keeps_historical_routing():
    spec = {
        "id": "legacy",
        "modalities": ["cxr"],
        "tasks": ["open_vqa"],
        "capabilities": ["generation"],
        "scope": "description",
    }
    descriptors = tool_descriptors({"legacy": spec}, row("Describe the image."))
    assert len(descriptors) == 1
    assert descriptors[0]["authority"]["status"] == "undeclared"
