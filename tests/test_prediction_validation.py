from alterra_trialgpt.prediction_validation import validate_prediction


def test_valid_prediction_has_no_violations() -> None:
    prediction = {
        "eligibility_label": "not excluded",
        "explanation": "The patient is 58 years old.",
    }

    violations = validate_prediction(prediction)

    assert violations == []


def test_invalid_label_produces_violation() -> None:

    prediction = {
        "eligibility_label": "maybe eligible",
        "explanation": "The patient is 16 years old.",
    }

    violations = validate_prediction(prediction)

    assert violations == ["invalid eligibility label"]


def test_missing_label_produces_violation() -> None:
    prediction = {
        "explanation": "The patient is 58 years old.",
    }

    violations = validate_prediction(prediction)

    assert violations == ["missing eligibility label"]


def test_blank_explanation_produces_violation() -> None:
    prediction = {
        "eligibility_label": "not excluded",
        "explanation": "   ",
    }

    violations = validate_prediction(prediction)

    assert violations == ["invalid explanation"]
