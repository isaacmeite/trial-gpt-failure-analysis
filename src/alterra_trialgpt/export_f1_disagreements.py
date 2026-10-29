import csv
import json
from pathlib import Path

PREDICTIONS_PATH = Path("results/f1_evaluation_predictions_v1.jsonl")
MANIFEST_PATH = Path("results/f1_case_manifest_v1.json")

ALL_OUTPUT = Path("results/f1_disagreements_v1.csv")
PRIORITY_OUTPUT = Path("results/f1_review_priority_v1.csv")

BASELINE = "trialgpt_adapted_v1"
ABLATION = "trialgpt_no_applicability_v1"

COLUMNS = [
    "annotation_id",
    "patient_id",
    "expert_label",
    "baseline_prediction",
    "ablation_prediction",
    "baseline_correct",
    "ablation_correct",
    "review_category",
    "priority",
    "criterion_text",
    "baseline_explanation",
    "ablation_explanation",
    "reviewer_decision",
    "reviewer_notes",
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def classify_case(
    expert: str,
    baseline: str,
    ablation: str,
) -> tuple[str, int]:
    """Assign a reproducible review category and priority"""

    if (
        expert == "not excluded"
        and baseline == "not applicable"
        and ablation != "not applicable"
    ):
        return "targeted_false_na_corrected", 1

    if (
        expert == "not excluded"
        and baseline != "not applicable"
        and ablation == "not applicable"
    ):
        return "new_false_na_error", 1

    if (
        expert == "not applicable"
        and baseline == "not applicable"
        and ablation != "not applicable"
    ):
        return "genuine_na_lost", 1

    if (
        expert == "not applicable"
        and baseline != "not applicable"
        and ablation == "not applicable"
    ):
        return "genuine_na_recovered", 1

    baseline_correct = baseline == expert
    ablation_correct = ablation == expert

    if not baseline_correct and ablation_correct:
        return "other_error_corrected", 2

    if baseline_correct and not ablation_correct:
        return "other_new_error", 2

    if not baseline_correct and not ablation_correct:
        return "both_incorrect", 3

    return "both_correct", 4


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    predictions = load_jsonl(PREDICTIONS_PATH)

    expected_ids = set(manifest["annotation_ids"])

    by_annotation = {}

    for record in predictions:
        annotation_id = record["annotation_id"]
        prompt_version = record["prompt_version"]

        if annotation_id not in expected_ids:
            raise ValueError(f"Unexpected annotation ID: {annotation_id}")

        if prompt_version not in {BASELINE, ABLATION}:
            raise ValueError(f"Unexpected prompt version: {prompt_version}")

        if record["repetition"] != 1:
            raise ValueError("Expected one repetition per condition.")

        if record["validation_violations"]:
            raise ValueError(f"Invalid prediction: {annotation_id}")

        pair = by_annotation.setdefault(annotation_id, {})

        if prompt_version in pair:
            raise ValueError(f"Duplicate prediction: {annotation_id}, {prompt_version}")

        pair[prompt_version] = record

    if set(by_annotation) != expected_ids:
        raise ValueError("Predictions do not cover frozen F1 cases.")

    rows = []

    for annotation_id in sorted(by_annotation):
        pair = by_annotation[annotation_id]

        if set(pair) != {BASELINE, ABLATION}:
            raise ValueError(f"Incomplete paired prediction: {annotation_id}")

        base = pair[BASELINE]
        ablated = pair[ABLATION]

        if (
            base["patient_id"] != ablated["patient_id"]
            or base["expert_eligibility"] != ablated["expert_eligibility"]
        ):
            raise ValueError(f"Inconsistent paired metadata: {annotation_id}")

        expert = base["expert_eligibility"]
        base_label = base["prediction"]["eligibility_label"]
        ablated_label = ablated["prediction"]["eligibility_label"]

        category, priority = classify_case(
            expert,
            base_label,
            ablated_label,
        )

        # Every case is included if at least one condition is wrong
        if category == "both_correct":
            continue

        rows.append(
            {
                "annotation_id": annotation_id,
                "patient_id": base["patient_id"],
                "expert_label": expert,
                "baseline_prediction": base_label,
                "ablation_prediction": ablated_label,
                "baseline_correct": base_label == expert,
                "ablation_correct": ablated_label == expert,
                "review_category": category,
                "priority": priority,
                # Filled from pinned dataset in a subsequent
                # read-only enrichment step if desired.
                "criterion_text": "",
                "baseline_explanation": (base["prediction"]["explanation"]),
                "ablation_explanation": (ablated["prediction"]["explanation"]),
                "reviewer_decision": "",
                "reviewer_notes": "",
            }
        )

    rows.sort(
        key=lambda row: (
            row["priority"],
            row["review_category"],
            row["annotation_id"],
        )
    )

    priority_rows = [row for row in rows if row["priority"] == 1]

    write_csv(ALL_OUTPUT, rows)
    write_csv(PRIORITY_OUTPUT, priority_rows)

    print("\nF1 DISAGREEMENT EXPORT")
    print("=" * 45)
    print("Frozen annotations:", len(expected_ids))
    print("Cases with at least one error:", len(rows))
    print("Priority 1 cases:", len(priority_rows))

    print("\nPriority cases:")
    for row in priority_rows:
        print(
            f"Annotation {row['annotation_id']}: "
            f"{row['review_category']} | "
            f"expert={row['expert_label']} | "
            f"baseline={row['baseline_prediction']} | "
            f"ablation={row['ablation_prediction']}"
        )

    print("\nSaved:")
    print(ALL_OUTPUT)
    print(PRIORITY_OUTPUT)


if __name__ == "__main__":
    main()
