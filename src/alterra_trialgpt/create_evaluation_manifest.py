import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

from alterra_trialgpt.evaluation_split import (
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    audit_evaluation_split,
    split_annotations_by_patient,
)

DATASET_NAME = "ncbi/TrialGPT-Criterion-Annotations"
MANIFEST_PATH = Path("results/evaluation_manifest_v1.json")
AUDIT_PATH = Path("results/evaluation_split_audit_v1.json")


def annotation_ids(rows: list[dict[str, Any]]) -> list[int]:
    """Return sorted annotation identifiers"""
    return sorted(row["annotation_id"] for row in rows)


def main() -> None:
    """Freeze and audit the patient-disjoint evaluation split"""

    if MANIFEST_PATH.exists() or AUDIT_PATH.exists():
        raise FileExistsError("Evaluation manifest already exists")

    # Resolve an immutable dataset revision
    dataset_revision = HfApi().dataset_info(DATASET_NAME).sha

    if not dataset_revision:
        raise ValueError("Unable to resolve dataset revision.")

    # Load the exact revision recorded in the manifest
    dataset = load_dataset(
        DATASET_NAME,
        revision=dataset_revision,
    )["train"]

    rows = [dict(row) for row in dataset]

    development, holdout = split_annotations_by_patient(rows)
    audit = audit_evaluation_split(development, holdout)

    if not audit["patient_disjoint"]:
        raise ValueError("Patient leakage detected.")

    if len(development) + len(holdout) != len(rows):
        raise ValueError("Split lost annotation records.")

    all_ids = annotation_ids(rows)
    development_ids = annotation_ids(development)
    holdout_ids = annotation_ids(holdout)

    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate annotation IDs in source dataset.")

    if set(development_ids) & set(holdout_ids):
        raise ValueError("Annotation overlap detected.")

    if sorted(development_ids + holdout_ids) != all_ids:
        raise ValueError("Split does not preserve all annotations.")

    # Removing cases already inspected during exploratory research
    previously_inspected_annotation_ids = {44, 108, 109}

    inspected_in_holdout = sorted(
        set(holdout_ids) & previously_inspected_annotation_ids
    )

    manifest = {
        "manifest_version": "v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": DATASET_NAME,
        "dataset_split": "train",
        "dataset_revision": dataset_revision,
        "split_seed": SPLIT_SEED,
        "holdout_fraction_target": HOLDOUT_FRACTION,
        "development_annotation_ids": development_ids,
        "holdout_annotation_ids": holdout_ids,
        "development_patient_ids": sorted({row["patient_id"] for row in development}),
        "holdout_patient_ids": sorted({row["patient_id"] for row in holdout}),
        "previously_inspected_annotation_ids": sorted(
            previously_inspected_annotation_ids
        ),
        "previously_inspected_in_holdout": inspected_in_holdout,
        "holdout_caveat": ("Partition created after exploratory aggregate analysis "),
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset revision: {dataset_revision}")
    print(f"Total annotations: {len(rows)}")
    print(f"Development patients: {audit['development_patients']}")
    print(f"Holdout patients: {audit['holdout_patients']}")
    print(f"Development annotations: {len(development)}")
    print(f"Holdout annotations: {len(holdout)}")
    print(f"Patient overlap: {audit['overlapping_patients']}")
    print(
        "Previously inspected annotations in holdout:",
        inspected_in_holdout,
    )
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print(f"Audit saved to: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
