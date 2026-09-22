from merit_feddg.transactional_claims import _radgraph_proposition


def test_official_null_suggestive_relation():
    annotation = {"observation": "effusion", "located_at": [],
                  "tags": ["definitely absent"], "suggestive_of": None}
    assert _radgraph_proposition(annotation) == "The image does not show effusion."


def test_official_relation_is_already_a_complete_phrase():
    annotation = {"observation": "consolidation", "located_at": [],
                  "tags": ["definitely present"],
                  "suggestive_of": ["consolidation suggestive of pneumonia"]}
    text = _radgraph_proposition(annotation)
    assert text == "The image shows consolidation; consolidation suggestive of pneumonia."


def test_uncertainty_is_preserved_when_present_in_native_tags():
    annotation = {"observation": "pneumonia", "located_at": [],
                  "tags": ["uncertain"], "suggestive_of": None}
    assert _radgraph_proposition(annotation) == "The image may show pneumonia."
