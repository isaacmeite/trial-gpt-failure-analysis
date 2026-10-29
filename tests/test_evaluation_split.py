from alterra_trialgpt.evaluation_split import (
    audit_evaluation_split,
    split_annotations_by_patient,
)


def test_split_prevents_patient_leakage() -> None:
    annotations = [
        {"annotation_id": 1, "patient_id": "patient_a"},
        {"annotation_id": 2, "patient_id": "patient_a"},
        {"annotation_id": 3, "patient_id": "patient_b"},
        {"annotation_id": 4, "patient_id": "patient_c"},
        {"annotation_id": 5, "patient_id": "patient_d"},
        {"annotation_id": 6, "patient_id": "patient_e"},
    ]

    development, holdout = split_annotations_by_patient(annotations)

    development_ids = {annotation["patient_id"] for annotation in development}

    holdout_ids = {annotation["patient_id"] for annotation in holdout}

    assert development_ids.isdisjoint(holdout_ids)
    assert len(development) + len(holdout) == len(annotations)


def test_split_is_reproducible() -> None:
    annotations = [
        {"annotation_id": i, "patient_id": f"patient_{i}"} for i in range(20)
    ]

    first_development, first_holdout = split_annotations_by_patient(annotations)

    second_development, second_holdout = split_annotations_by_patient(annotations)

    assert first_development == second_development
    assert first_holdout == second_holdout


def test_audit_detects_patient_overlap() -> None:
    development = [
        {"annotation_id": 1, "patient_id": "patient_a"},
        {"annotation_id": 2, "patient_id": "patient_b"},
    ]

    holdout = [
        {"annotation_id": 3, "patient_id": "patient_b"},
        {"annotation_id": 4, "patient_id": "patient_c"},
    ]

    audit = audit_evaluation_split(development, holdout)

    assert audit["overlapping_patients"] == 1
    assert audit["patient_disjoint"] is False
