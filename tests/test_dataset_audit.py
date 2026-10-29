from alterra_trialgpt.dataset_audit import audit_annotations

annotation_5 = {
    "annotation_id": 5,
    "patient_id": 156,
    "criterion_type": "inclusion",
    "expert_eligibility": "included",
    "gpt4_eligibility": "included",
}

annotation_6 = {
    "annotation_id": 6,
    "patient_id": 157,
    "criterion_type": "inclusion",
    "expert_eligibility": "excluded",
    "gpt4_eligibility": "included",
}

annotation_7 = {
    "annotation_id": 7,
    "patient_id": None,
    "criterion_type": "inclusion",
    "expert_eligibility": "not enough information",
    "gpt4_eligibility": "included",
}


def test_audit_collects_invalid_annotations() -> None:
    annotations = [
        annotation_5,
        annotation_6,
        annotation_7,
    ]

    actual = audit_annotations(annotations)

    assert len(actual) == 2


def test_audit_collects_valid_annotations() -> None:
    annotation = annotation_5

    actual = audit_annotations([annotation])

    assert len(actual) == 0
