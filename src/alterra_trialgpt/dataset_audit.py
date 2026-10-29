from typing import Any

from alterra_trialgpt.data_validation import validate_annotation


def audit_annotations(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect validation violations across annotations"""

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


def find_duplicate_annotation_ids(
    annotations: list[dict[str, Any]],
) -> list[int]:
    """Return annotation IDs that occur more than once"""
    observed_ids = set()
    duplicate_ids = set()

    for annotation in annotations:
        annotation_id = annotation.get("annotation_id")
        if annotation_id in observed_ids:
            duplicate_ids.add(annotation_id)
        else:
            observed_ids.add(annotation_id)

    return list(duplicate_ids)
