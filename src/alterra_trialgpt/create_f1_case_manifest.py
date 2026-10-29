import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SOURCE_MANIFEST = Path("results/evaluation_manifest_v1.json")
OUTPUT_PATH = Path("results/f1_case_manifest_v1.json")

PREVIOUSLY_INSPECTED_IDS = {44, 108, 109}


def main() -> None:
    """Freeze exclusion-criterion cases for F1 evaluation"""

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "F1 manifest already exists. Do not overwrite a frozen evaluation"
        )

    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    dataset = load_dataset(
        source["dataset"],
        revision=source["dataset_revision"],
    )["train"]

    rows = list(dataset)
    holdout_ids = set(source["holdout_annotation_ids"])

    inspected_patients = {
        row["patient_id"]
        for row in rows
        if row["annotation_id"] in PREVIOUSLY_INSPECTED_IDS
    }

    eligible_cases = sorted(
        (
            row
            for row in rows
            if row["annotation_id"] in holdout_ids
            and row["patient_id"] not in inspected_patients
            and row["criterion_type"] == "exclusion"
        ),
        key=lambda row: row["annotation_id"],
    )

    label_counts = Counter(row["expert_eligibility"] for row in eligible_cases)

    manifest = {
        "version": "f1_v1",
        "source_manifest": str(SOURCE_MANIFEST),
        "dataset_revision": source["dataset_revision"],
        "criterion_type": "exclusion",
        "excluded_exploratory_patient_ids": sorted(inspected_patients),
        "annotation_ids": [row["annotation_id"] for row in eligible_cases],
        "patient_ids": sorted({row["patient_id"] for row in eligible_cases}),
        "reference_label_counts": dict(sorted(label_counts.items())),
        "selection_rule": (
            "All exclusion annotations in the frozen patient holdout, "
            "excluding every patient linked to the three explicitly "
            "documented previously inspected annotations."
        ),
        "limitations": [
            "The original dataset was examined during exploration.",
            "Historical expert labels require clinical interpretation.",
            (
                "This is a reserved evaluation partition, not a pristine "
                "prospective external test set."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("F1 evaluation manifest frozen")
    print("Eligible annotations:", len(eligible_cases))
    print("Eligible patients:", len(manifest["patient_ids"]))
    print("Reference label counts:", dict(label_counts))
    print("Previously inspected patients:", sorted(inspected_patients))
    print("Manifest:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
