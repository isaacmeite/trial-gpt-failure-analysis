import json
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST_PATH = Path("results/f1_case_manifest_v1.json")
PROTOCOL_PATH = Path("results/f1_evaluation_protocol_v1.json")
PREDICTIONS_PATH = Path("results/f1_evaluation_predictions_v1.jsonl")
OUTPUT_PATH = Path("results/f1_analysis_v1.json")

LABELS = [
    "excluded",
    "not excluded",
    "not applicable",
    "not enough information",
]

BASELINE = "trialgpt_adapted_v1"
ABLATION = "trialgpt_no_applicability_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> list[dict]:
    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}") from error

    return records


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def build_confusion_matrix(records: list[dict]) -> dict:
    matrix = {
        reference: {prediction: 0 for prediction in LABELS} for reference in LABELS
    }

    for record in records:
        reference = record["expert_eligibility"]
        prediction = record["prediction"]["eligibility_label"]
        matrix[reference][prediction] += 1

    return matrix


def calculate_class_metrics(matrix: dict) -> dict:
    metrics = {}

    for label in LABELS:
        true_positive = matrix[label][label]

        support = sum(matrix[label].values())
        predicted_count = sum(matrix[reference][label] for reference in LABELS)

        precision = safe_divide(true_positive, predicted_count)
        recall = safe_divide(true_positive, support)

        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        metrics[label] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics


def analyze_condition(records: list[dict]) -> dict:
    matrix = build_confusion_matrix(records)

    correct = sum(
        record["prediction"]["eligibility_label"] == record["expert_eligibility"]
        for record in records
    )

    not_excluded_cases = [
        record for record in records if record["expert_eligibility"] == "not excluded"
    ]

    false_na_count = sum(
        record["prediction"]["eligibility_label"] == "not applicable"
        for record in not_excluded_cases
    )

    genuine_na_cases = [
        record for record in records if record["expert_eligibility"] == "not applicable"
    ]

    genuine_na_correct = sum(
        record["prediction"]["eligibility_label"] == "not applicable"
        for record in genuine_na_cases
    )

    return {
        "total": len(records),
        "correct": correct,
        "accuracy": safe_divide(correct, len(records)),
        "false_na_count": false_na_count,
        "false_na_denominator": len(not_excluded_cases),
        "false_na_rate": safe_divide(false_na_count, len(not_excluded_cases)),
        "genuine_na_correct": genuine_na_correct,
        "genuine_na_denominator": len(genuine_na_cases),
        "genuine_na_recall": safe_divide(genuine_na_correct, len(genuine_na_cases)),
        "confusion_matrix": matrix,
        "class_metrics": calculate_class_metrics(matrix),
    }


def main() -> None:
    manifest = load_json(MANIFEST_PATH)
    protocol = load_json(PROTOCOL_PATH)
    records = load_predictions(PREDICTIONS_PATH)

    expected_ids = set(manifest["annotation_ids"])

    if len(expected_ids) != len(manifest["annotation_ids"]):
        raise ValueError("Duplicate IDs in frozen case manifest.")

    if protocol["experimental_conditions"] != [
        BASELINE,
        ABLATION,
    ]:
        raise ValueError("Unexpected frozen prompt conditions.")

    if protocol["repetitions_per_condition"] != 1:
        raise ValueError("This analysis requires one repetition per condition.")

    expected_keys = {
        (annotation_id, condition, 1)
        for annotation_id in expected_ids
        for condition in [BASELINE, ABLATION]
    }

    actual_keys = [
        (
            record["annotation_id"],
            record["prompt_version"],
            record["repetition"],
        )
        for record in records
    ]

    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("Duplicate prediction records detected.")

    if set(actual_keys) != expected_keys:
        missing = expected_keys - set(actual_keys)
        extra = set(actual_keys) - expected_keys
        raise ValueError(
            f"Evaluation incomplete or inconsistent. "
            f"Missing={len(missing)}, extra={len(extra)}"
        )

    if any(record["validation_violations"] for record in records):
        raise ValueError(
            "Some model responses failed validation. "
            "Review them before running this analysis."
        )

    if any(
        record["requested_model"] != protocol["model"]
        or record["response_model"] != protocol["model"]
        for record in records
    ):
        raise ValueError("Model version does not match protocol.")

    if any(
        record["expert_eligibility"] not in LABELS
        or record["prediction"]["eligibility_label"] not in LABELS
        for record in records
    ):
        raise ValueError("Unexpected eligibility label found.")

    patients_by_annotation = defaultdict(set)
    references_by_annotation = defaultdict(set)
    for record in records:
        patients_by_annotation[record["annotation_id"]].add(record["patient_id"])
        references_by_annotation[record["annotation_id"]].add(
            record["expert_eligibility"]
        )

    if any(len(values) != 1 for values in patients_by_annotation.values()):
        raise ValueError("Inconsistent patient ID within a paired case.")

    if any(len(values) != 1 for values in references_by_annotation.values()):
        raise ValueError("Inconsistent expert label within a paired case.")

    patient_ids = {next(iter(values)) for values in patients_by_annotation.values()}

    if patient_ids != set(manifest["patient_ids"]):
        raise ValueError("Prediction patients do not match frozen manifest.")

    observed_labels = Counter(
        next(iter(values)) for values in references_by_annotation.values()
    )

    if dict(observed_labels) != manifest["reference_label_counts"]:
        raise ValueError("Expert label counts do not match manifest.")

    by_condition = {
        condition: [
            record for record in records if record["prompt_version"] == condition
        ]
        for condition in [BASELINE, ABLATION]
    }

    baseline = analyze_condition(by_condition[BASELINE])
    ablation = analyze_condition(by_condition[ABLATION])

    paired = defaultdict(dict)

    for record in records:
        paired[record["annotation_id"]][record["prompt_version"]] = record

    corrected = 0
    newly_wrong = 0
    unchanged = 0

    for pair in paired.values():
        baseline_record = pair[BASELINE]
        ablation_record = pair[ABLATION]

        baseline_correct = (
            baseline_record["prediction"]["eligibility_label"]
            == baseline_record["expert_eligibility"]
        )

        ablation_correct = (
            ablation_record["prediction"]["eligibility_label"]
            == ablation_record["expert_eligibility"]
        )

        if not baseline_correct and ablation_correct:
            corrected += 1
        elif baseline_correct and not ablation_correct:
            newly_wrong += 1
        else:
            unchanged += 1

    results = {
        "experiment_id": "f1_evaluation_v1",
        "model": protocol["model"],
        "total_annotations": len(expected_ids),
        "total_predictions": len(records),
        "total_patients": len(patient_ids),
        "reference_label_counts": dict(observed_labels),
        "baseline": baseline,
        "ablation": ablation,
        "paired_comparison": {
            "corrected_cases": corrected,
            "newly_wrong_cases": newly_wrong,
            "unchanged_correctness_cases": unchanged,
            "accuracy_improvement": (ablation["accuracy"] - baseline["accuracy"]),
            "false_na_rate_reduction": (
                baseline["false_na_rate"] - ablation["false_na_rate"]
            ),
            "genuine_na_recall_change": (
                ablation["genuine_na_recall"] - baseline["genuine_na_recall"]
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nF1 EVALUATION RESULTS")
    print("=" * 50)
    print("Annotations:", len(expected_ids))
    print("Patients:", len(patient_ids))
    print("Predictions:", len(records))

    for name, analysis in [
        ("ADAPTED PROMPT", baseline),
        ("ABLATION PROMPT", ablation),
    ]:
        print(f"\n{name}")
        print(
            "Overall accuracy:",
            f"{analysis['correct']}/{analysis['total']}",
            f"({analysis['accuracy']:.1%})",
        )
        print(
            "False-NA rate:",
            f"{analysis['false_na_count']}/{analysis['false_na_denominator']}",
            f"({analysis['false_na_rate']:.1%})",
        )
        print(
            "Genuine NA recall:",
            f"{analysis['genuine_na_correct']}/{analysis['genuine_na_denominator']}",
            f"({analysis['genuine_na_recall']:.1%})",
        )

        print("\nPer-class metrics:")
        for label, metrics in analysis["class_metrics"].items():
            print(
                f"  {label}: "
                f"precision={metrics['precision']}, "
                f"recall={metrics['recall']}, "
                f"f1={metrics['f1']}"
            )

        print("\nConfusion matrix (rows=expert, columns=model):")
        print("Label order:", LABELS)
        for label in LABELS:
            print(
                label,
                [
                    analysis["confusion_matrix"][label][predicted]
                    for predicted in LABELS
                ],
            )

    print("\nPAIRED COMPARISON")
    print("Incorrect -> correct:", corrected)
    print("Correct -> incorrect:", newly_wrong)
    print("Unchanged correctness:", unchanged)
    print(
        "Accuracy improvement:",
        f"{results['paired_comparison']['accuracy_improvement']:.1%}",
    )
    print(
        "False-NA reduction:",
        f"{results['paired_comparison']['false_na_rate_reduction']:.1%}",
    )

    print("\nSaved analysis:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
