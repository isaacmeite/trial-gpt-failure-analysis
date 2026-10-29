import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

MANIFEST_PATH = Path("results/f1_case_manifest_v1.json")
PROTOCOL_PATH = Path("results/f1_evaluation_protocol_v1.json")


def build_f1_evaluation_plan(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one planned model call per case and condition."""

    selected_ids = set(manifest["annotation_ids"])

    if len(selected_ids) != len(manifest["annotation_ids"]):
        raise ValueError("Duplicate annotation IDs in manifest.")

    rows_by_id = {
        row["annotation_id"]: row
        for row in annotations
        if row["annotation_id"] in selected_ids
    }

    if set(rows_by_id) != selected_ids:
        raise ValueError("Dataset does not match frozen annotation IDs.")

    if len(rows_by_id) != len(
        [row for row in annotations if row["annotation_id"] in selected_ids]
    ):
        raise ValueError("Duplicate selected annotation IDs in dataset.")

    plan = []

    for annotation_id in sorted(selected_ids):
        row = rows_by_id[annotation_id]

        if row["criterion_type"] != "exclusion":
            raise ValueError("Non-exclusion annotation in F1 manifest.")

        if row["patient_id"] not in manifest["patient_ids"]:
            raise ValueError("Patient ID does not match frozen manifest.")

        for repetition in range(
            1,
            protocol["repetitions_per_condition"] + 1,
        ):
            for prompt_version in protocol["experimental_conditions"]:
                plan.append(
                    {
                        "annotation_id": annotation_id,
                        "patient_id": row["patient_id"],
                        "criterion_type": row["criterion_type"],
                        "prompt_version": prompt_version,
                        "repetition": repetition,
                        "expert_eligibility": row["expert_eligibility"],
                    }
                )

    expected_calls = (
        len(selected_ids)
        * len(protocol["experimental_conditions"])
        * protocol["repetitions_per_condition"]
    )

    if len(plan) != expected_calls:
        raise ValueError("Evaluation plan has incorrect call count.")

    return plan


def main() -> None:
    """Print the frozen F1 evaluation schedule without API calls."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    dataset = load_dataset(
        "ncbi/TrialGPT-Criterion-Annotations",
        revision=manifest["dataset_revision"],
    )["train"]

    plan = build_f1_evaluation_plan(
        manifest,
        protocol,
        [dict(row) for row in dataset],
    )

    print("F1 EVALUATION DRY RUN")
    print("Annotations:", len(manifest["annotation_ids"]))
    print("Patients:", len(manifest["patient_ids"]))
    print("Prompt conditions:", protocol["experimental_conditions"])
    print("Repetitions:", protocol["repetitions_per_condition"])
    print("Planned API calls:", len(plan))

    print("\nFirst four planned calls:")
    for job in plan[:4]:
        print(job)

    print("\nNo OpenAI API calls were made.")


if __name__ == "__main__":
    main()
