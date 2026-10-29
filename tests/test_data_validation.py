import pytest

from alterra_trialgpt.data_validation import validate_annotation
from alterra_trialgpt.validation import is_valid_eligibility_label


@pytest.mark.parametrize(
    "criterion_type, label, expected",
    [
        ("inclusion", "included", True),
        ("inclusion", "excluded", False),
        ("exclusion", "excluded", True),
        ("exclusion", "included", False),
        ("inclusion", "not enough information", True),
        ("exclusion", "not applicable", True),
        ("inclusion", "unknown label", False),
    ],
)
def test_is_valid_eligibility_label(
    criterion_type: str, label: str, expected: bool
) -> None:
    actual = is_valid_eligibility_label(criterion_type, label)
    assert actual is expected


def test_unknown_criterion_type() -> None:
    with pytest.raises(ValueError):
        is_valid_eligibility_label("unknown", "included")


def test_valid_annotation_has_no_violations() -> None:
    annotation_1 = {
        "annotation_id": 1,
        "patient_id": 100,
        "criterion_type": "inclusion",
        "expert_eligibility": "included",
        "gpt4_eligibility": "included",
    }

    violations_1 = validate_annotation(annotation_1)
    assert violations_1 == []


def test_invalid_annotation_has_violations() -> None:

    annotation_2 = {
        "annotation_id": 2,
        "patient_id": 50,
        "criterion_type": "inclusion",
        "expert_eligibility": "excluded",
        "gpt4_eligibility": "included",
    }

    violations_2 = validate_annotation(annotation_2)
    assert violations_2 == ["invalid_expert_label"]


def test_unknown_annotation_criterion_type() -> None:
    annotation_3 = {
        "annotation_id": 3,
        "patient_id": 100,
        "criterion_type": "unknown",
        "expert_eligibility": "included",
        "gpt4_eligibility": "included",
    }

    violations_3 = validate_annotation(annotation_3)

    assert violations_3 == ["invalid_criterion_type"]


def test_annotation_reports_multiple_violations() -> None:
    annotation_4 = {
        "annotation_id": None,
        "patient_id": None,
        "criterion_type": "inclusion",
        "expert_eligibility": "excluded",
        "gpt4_eligibility": "included",
    }

    violations_4 = validate_annotation(annotation_4)

    assert set(violations_4) == {
        "invalid_expert_label",
        "missing_patient_id",
        "missing_annotation_id",
    }
