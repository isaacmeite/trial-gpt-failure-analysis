from typing import Any

from alterra_trialgpt.validation import (
    VALID_LABELS,
    is_valid_eligibility_label,
)


def validate_annotation(
    annotation: dict[str, Any],
) -> list[str]:
    """
     Identify schema violations in one TrialGPT annotation.

    Args:
        annotation (dict): A TrialGPT annotation dictionary.

    Returns:
        list[str]: A list of error messages for any schema violations found withinn annotation dictionary.
    """

    violations = []

    if annotation.get("criterion_type") is None:
        violations.append("missing_criterion_type")

    else:
        criterion_type = annotation["criterion_type"]

        if criterion_type not in VALID_LABELS:
            violations.append("invalid_criterion_type")
        else:
            if annotation.get("expert_eligibility") is None:
                violations.append("missing_expert_eligibility")
            else:
                is_valid_expert_label = is_valid_eligibility_label(
                    criterion_type,
                    annotation["expert_eligibility"],
                )

                if not is_valid_expert_label:
                    violations.append("invalid_expert_label")

            if annotation.get("gpt4_eligibility") is None:
                violations.append("missing_gpt4_eligibility")
            else:
                is_valid_gpt4_label = is_valid_eligibility_label(
                    criterion_type,
                    annotation["gpt4_eligibility"],
                )

                if not is_valid_gpt4_label:
                    violations.append("invalid_gpt4_label")

    if annotation.get("patient_id") is None:
        violations.append("missing_patient_id")

    if annotation.get("annotation_id") is None:
        violations.append("missing_annotation_id")

    return violations
