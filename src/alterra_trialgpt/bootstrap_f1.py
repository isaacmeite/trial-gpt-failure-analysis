import json
import random
from collections import defaultdict
from pathlib import Path

PREDICTIONS_PATH = Path("results/f1_evaluation_predictions_v1.jsonl")
ANALYSIS_PATH = Path("results/f1_analysis_v1.json")
OUTPUT_PATH = Path("results/f1_bootstrap_v1.json")

BASELINE = "trialgpt_adapted_v1"
ABLATION = "trialgpt_no_applicability_v1"

BOOTSTRAP_SEED = 20261028
N_BOOTSTRAPS = 10000


def load_predictions() -> list[dict]:
    return [
        json.loads(line)
        for line in PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], probability: float) -> float:
    """Calculate a percentile with linear interpolation."""
    values = sorted(values)

    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return values[lower] * (1 - fraction) + values[upper] * fraction


def summarize_interval(values: list[float]) -> dict:
    return {
        "lower_95": percentile(values, 0.025),
        "upper_95": percentile(values, 0.975),
        "valid_bootstrap_samples": len(values),
    }


def count_metrics(records: list[dict]) -> dict:
    """Calculate the numerators and denominators for one prompt."""
    correct = 0
    false_na = 0
    not_excluded_count = 0
    genuine_na_correct = 0
    genuine_na_count = 0

    for record in records:
        reference = record["expert_eligibility"]
        predicted = record["prediction"]["eligibility_label"]

        if predicted == reference:
            correct += 1

        if reference == "not excluded":
            not_excluded_count += 1
            if predicted == "not applicable":
                false_na += 1

        if reference == "not applicable":
            genuine_na_count += 1
            if predicted == "not applicable":
                genuine_na_correct += 1

    return {
        "correct": correct,
        "total": len(records),
        "false_na": false_na,
        "not_excluded_count": not_excluded_count,
        "genuine_na_correct": genuine_na_correct,
        "genuine_na_count": genuine_na_count,
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_effects(
    baseline_records: list[dict],
    ablation_records: list[dict],
) -> dict:
    base = count_metrics(baseline_records)
    ablated = count_metrics(ablation_records)

    baseline_accuracy = safe_rate(base["correct"], base["total"])
    ablated_accuracy = safe_rate(ablated["correct"], ablated["total"])

    baseline_false_na = safe_rate(base["false_na"], base["not_excluded_count"])
    ablated_false_na = safe_rate(ablated["false_na"], ablated["not_excluded_count"])

    baseline_na_recall = safe_rate(base["genuine_na_correct"], base["genuine_na_count"])
    ablated_na_recall = safe_rate(
        ablated["genuine_na_correct"], ablated["genuine_na_count"]
    )

    return {
        "accuracy_improvement": (
            ablated_accuracy - baseline_accuracy
            if baseline_accuracy is not None and ablated_accuracy is not None
            else None
        ),
        "false_na_rate_reduction": (
            baseline_false_na - ablated_false_na
            if baseline_false_na is not None and ablated_false_na is not None
            else None
        ),
        "genuine_na_recall_change": (
            ablated_na_recall - baseline_na_recall
            if baseline_na_recall is not None and ablated_na_recall is not None
            else None
        ),
    }


def main() -> None:
    records = load_predictions()

    # Require the complete, previously validated analysis.
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))

    if len(records) != analysis["total_predictions"]:
        raise ValueError("Prediction count does not match F1 analysis.")

    if len(records) != 296:
        raise ValueError("Expected exactly 296 predictions.")

    # Group all model responses by patient, preserving pairing.
    by_patient = defaultdict(list)

    for record in records:
        if record["validation_violations"]:
            raise ValueError("Invalid response in bootstrap dataset.")

        by_patient[record["patient_id"]].append(record)

    patient_ids = sorted(by_patient)

    if len(patient_ids) != analysis["total_patients"]:
        raise ValueError("Patient count does not match F1 analysis.")

    if len(patient_ids) != 10:
        raise ValueError("Expected exactly 10 patients.")

    # Independently verify one prediction per condition per annotation.
    by_annotation = defaultdict(set)

    for record in records:
        key = (record["annotation_id"], record["repetition"])
        condition = record["prompt_version"]

        if condition in by_annotation[key]:
            raise ValueError(f"Duplicate condition for {key}.")

        by_annotation[key].add(condition)

    if any(conditions != {BASELINE, ABLATION} for conditions in by_annotation.values()):
        raise ValueError("One or more annotations lack paired prompts.")

    baseline_records = [
        record for record in records if record["prompt_version"] == BASELINE
    ]

    ablation_records = [
        record for record in records if record["prompt_version"] == ABLATION
    ]

    observed = compute_effects(
        baseline_records,
        ablation_records,
    )

    rng = random.Random(BOOTSTRAP_SEED)

    bootstrap_effects = {name: [] for name in observed}

    undefined_counts = {name: 0 for name in observed}

    for _ in range(N_BOOTSTRAPS):
        sampled_patients = rng.choices(
            patient_ids,
            k=len(patient_ids),
        )

        # Retain every annotation and both model conditions
        # for each sampled patient. Repeatedly sampled patients
        # contribute their records repeatedly.
        sampled_records = []

        for patient_id in sampled_patients:
            sampled_records.extend(by_patient[patient_id])

        sampled_baseline = [
            record for record in sampled_records if record["prompt_version"] == BASELINE
        ]

        sampled_ablation = [
            record for record in sampled_records if record["prompt_version"] == ABLATION
        ]

        effects = compute_effects(
            sampled_baseline,
            sampled_ablation,
        )

        for name, effect in effects.items():
            if effect is None:
                undefined_counts[name] += 1
            else:
                bootstrap_effects[name].append(effect)

    intervals = {}

    for name, values in bootstrap_effects.items():
        if not values:
            raise ValueError(f"No valid bootstrap samples for {name}.")

        intervals[name] = {
            "observed_effect": observed[name],
            **summarize_interval(values),
            "undefined_bootstrap_samples": undefined_counts[name],
        }

    output = {
        "experiment_id": "f1_evaluation_v1",
        "method": "paired_patient_cluster_percentile_bootstrap",
        "seed": BOOTSTRAP_SEED,
        "bootstrap_samples": N_BOOTSTRAPS,
        "patient_clusters": len(patient_ids),
        "annotations": len(by_annotation),
        "confidence_level": 0.95,
        "intervals": intervals,
        "limitations": [
            "Only 10 patient clusters.",
            "Cluster sizes are unequal.",
            "Rare reference classes can be absent from resamples.",
            "Percentile intervals may have poor coverage with few clusters.",
            "Results describe this benchmark and adapted prompts.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nF1 PATIENT-CLUSTER BOOTSTRAP")
    print("=" * 52)
    print("Patients:", len(patient_ids))
    print("Annotations:", len(by_annotation))
    print("Bootstrap samples:", N_BOOTSTRAPS)

    for name, result in intervals.items():
        print(f"\n{name}")
        print(f"Observed effect: {result['observed_effect']:+.2%}")
        print(
            "95% CI:",
            f"[{result['lower_95']:+.2%}, {result['upper_95']:+.2%}]",
        )
        print(
            "Undefined resamples:",
            result["undefined_bootstrap_samples"],
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
