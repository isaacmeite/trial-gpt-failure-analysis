from alterra_trialgpt.run_f1_evaluation import (
    job_key,
    validate_response,
)


def test_job_key_distinguishes_conditions() -> None:
    baseline = {
        "annotation_id": 10,
        "prompt_version": "trialgpt_adapted_v1",
        "repetition": 1,
    }
    intervention = {
        **baseline,
        "prompt_version": "trialgpt_no_applicability_v1",
    }

    assert job_key(baseline) != job_key(intervention)


def test_valid_response_passes_validation() -> None:
    raw = (
        '{"eligibility_label": "not excluded", '
        '"explanation": "The criterion is not met."}'
    )

    prediction, violations = validate_response(raw)

    assert violations == []
    assert prediction["eligibility_label"] == "not excluded"


def test_invalid_response_is_flagged() -> None:
    prediction, violations = validate_response(
        '{"eligibility_label": "maybe", "explanation": ""}'
    )

    assert prediction is not None
    assert "invalid_eligibility_label" in violations
    assert "invalid_explanation" in violations
