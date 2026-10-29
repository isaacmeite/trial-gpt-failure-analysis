from typing import Any

from alterra_trialgpt.data_validation import validate_annotation


def audit_annotations(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect validation violations across annotations."""
    audit_results = []

    for annotation in annotations:
        violations = validate_annotation(annotation)
        if violations:
            audit_results.append(
                {
                    "annotation_id": annotation.get("annotation_id"),
                    "violations": violations,
                }
            )

    return audit_results
