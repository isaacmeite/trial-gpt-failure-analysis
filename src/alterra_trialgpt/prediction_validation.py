from typing import Any

from alterra_trialgpt.validation import VALID_LABELS


def validate_prediction(
    prediction: dict[str, Any],
) -> list[str]:
    """Return violations in one model-generated exclusion prediction"""

    exclusion_labels = VALID_LABELS["exclusion"]
    violations = []

    label = prediction.get("eligibility_label")

    if label is None:
        violations.append("missing eligibility label")

    elif label not in exclusion_labels:
        violations.append("invalid eligibility label")

    explanation = prediction.get("explanation")

    if not isinstance(explanation, str):
        violations.append("explanation must be a string")

    elif not explanation.strip():
        violations.append("invalid explanation")

    return violations
