import random
from typing import Any

SPLIT_SEED = 20261027
HOLDOUT_FRACTION = 0.20


def split_annotations_by_patient(
    annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split annotations into patient-disjoint development and holdout sets"""

    patient_ids = sorted({annotation["patient_id"] for annotation in annotations})

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(patient_ids)

    holdout_count = round(len(patient_ids) * HOLDOUT_FRACTION)
    holdout_patient_ids = set(patient_ids[:holdout_count])

    development = [
        annotation
        for annotation in annotations
        if annotation["patient_id"] not in holdout_patient_ids
    ]

    holdout = [
        annotation
        for annotation in annotations
        if annotation["patient_id"] in holdout_patient_ids
    ]

    return development, holdout


def audit_evaluation_split(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize a patient-disjoint evaluation split."""

    development_patients = {annotation["patient_id"] for annotation in development}

    holdout_patients = {annotation["patient_id"] for annotation in holdout}

    overlap = development_patients & holdout_patients

    return {
        "development_annotations": len(development),
        "holdout_annotations": len(holdout),
        "development_patients": len(development_patients),
        "holdout_patients": len(holdout_patients),
        "overlapping_patients": len(overlap),
        "patient_disjoint": len(overlap) == 0,
        "holdout_patient_ids": sorted(holdout_patients),
    }
