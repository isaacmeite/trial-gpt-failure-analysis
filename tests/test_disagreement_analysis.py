import pytest

from alterra_trialgpt.disagreement_analysis import (
    count_disagreements,
    count_label_transitions,
)


def test_count_disagreements() -> None:
    annotations = [
        {
            "annotation_id": 1,
            "patient_id": 101,
            "criterion_type": "inclusion",
            "expert_eligibility": "included",
            "gpt4_eligibility": "included",
        },
        {
            "annotation_id": 2,
            "patient_id": 102,
            "criterion_type": "exclusion",
            "expert_eligibility": "excluded",
            "gpt4_eligibility": "not excluded",
        },
        {
            "annotation_id": 3,
            "patient_id": 103,
            "criterion_type": "inclusion",
            "expert_eligibility": "not enough information",
            "gpt4_eligibility": "included",
        },
    ]

    disagreement_count = count_disagreements(annotations)

    assert disagreement_count == 2


def test_count_disagreements_empty_input() -> None:
    assert count_disagreements([]) == 0


def test_count_disagreements_rejects_missing_labels() -> None:
    annotations = [
        {
            "annotation_id": 3,
            "patient_id": 103,
            "criterion_type": "inclusion",
            "expert_eligibility": "not enough information",
            "gpt4_eligibility": None,
        }
    ]

    with pytest.raises(ValueError):
        count_disagreements(annotations)


def test_count_transitions() -> None:
    annotations = [
        {
            "criterion_type": "inclusion",
            "expert_eligibility": "included",
            "gpt4_eligibility": "not enough information",
        },
        {
            "criterion_type": "inclusion",
            "expert_eligibility": "included",
            "gpt4_eligibility": "not enough information",
        },
        {
            "criterion_type": "inclusion",
            "expert_eligibility": "included",
            "gpt4_eligibility": "included",
        },
    ]

    transitions = count_label_transitions(annotations)

    assert transitions[("inclusion", "included", "not enough information")] == 2
    assert transitions[("inclusion", "included", "included")] == 1
