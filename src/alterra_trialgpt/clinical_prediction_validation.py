from typing import Any

VALID_CLINICAL_STATUSES = {"present", "absent", "unknown"}


def validate_clinical_prediction(
    prediction: dict[str, Any],
) -> list[str]:
    "Validate a clinical-status response."

    violations = []

    status = prediction.get("clinical_status")
    explanation = prediction.get("explanation")

    if status is None:
        violations.append("missing_clinical_status")
    elif not isinstance(status, str) or status not in VALID_CLINICAL_STATUSES:
        violations.append("invalid_clinical_status")

    if not isinstance(explanation, str) or not explanation.strip():
        violations.append("invalid_explanation")

    return violations
