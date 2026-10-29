import json
from collections import Counter
from pathlib import Path

MANIFEST_PATH = Path("results/f2_case_manifest_v1.json")
PREDICTIONS_PATH = Path("results/f2_evaluation_predictions_v1.jsonl")
OUTPUT_PATH = Path("results/f2_analysis_v1.json")

CONDITIONS = ("clinical_neutral_v1", "clinical_inference_v1")
VALID_STATUSES = {"present", "absent", "unknown"}


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    cases = {case["case_id"]: case for case in manifest["cases"]}
    if len(cases) != 20:
        raise ValueError("Expected 20 unique cases.")

    records = [
        json.loads(line)
        for line in PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    predictions = {}

    for record in records:
        key = (record["case_id"], record["prompt_version"])

        if key in predictions:
            raise ValueError(f"Duplicate prediction: {key}")

        if record["case_id"] not in cases:
            raise ValueError(f"Unknown case: {key}")

        if record["prompt_version"] not in CONDITIONS:
            raise ValueError(f"Unknown prompt: {key}")

        if record["repetition"] != 1:
            raise ValueError(f"Unexpected repetition: {key}")

        if record["validation_violations"]:
            raise ValueError(f"Invalid response: {key}")

        status = record["prediction"]["clinical_status"]

        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid clinical status: {key}")

        if record["expected_status"] != cases[key[0]]["expected_status"]:
            raise ValueError(f"Reference label mismatch: {key}")

        predictions[key] = status

    expected_keys = {
        (case_id, condition) for case_id in cases for condition in CONDITIONS
    }

    if set(predictions) != expected_keys:
        raise ValueError(
            f"Missing jobs: {expected_keys - set(predictions)}; "
            f"unexpected jobs: {set(predictions) - expected_keys}"
        )

    results = {}

    for condition in CONDITIONS:
        by_reference = {}

        for expected in ("unknown", "present", "absent"):
            selected = [
                (case_id, predictions[(case_id, condition)])
                for case_id, case in cases.items()
                if case["expected_status"] == expected
            ]

            counts = Counter(status for _, status in selected)

            by_reference[expected] = {
                "n": len(selected),
                "correct": counts[expected],
                "predicted_status_counts": dict(sorted(counts.items())),
            }

        unknown = by_reference["unknown"]
        unsupported_absent = unknown["predicted_status_counts"].get("absent", 0)

        correct = sum(group["correct"] for group in by_reference.values())

        results[condition] = {
            "by_reference": by_reference,
            "unsupported_absent_count": unsupported_absent,
            "unknown_case_count": unknown["n"],
            "unsupported_absent_rate": (unsupported_absent / unknown["n"]),
            "overall_correct": correct,
            "overall_n": len(cases),
            "overall_accuracy": correct / len(cases),
        }

    unknown_case_ids = [
        case_id
        for case_id, case in cases.items()
        if case["expected_status"] == "unknown"
    ]

    changed_cases = []

    for case_id in unknown_case_ids:
        neutral = predictions[(case_id, CONDITIONS[0])]
        inference = predictions[(case_id, CONDITIONS[1])]

        if neutral != inference:
            changed_cases.append(
                {
                    "case_id": case_id,
                    "condition": cases[case_id]["condition"],
                    "neutral_prediction": neutral,
                    "inference_prediction": inference,
                }
            )

    neutral_rate = results[CONDITIONS[0]]["unsupported_absent_rate"]
    inference_rate = results[CONDITIONS[1]]["unsupported_absent_rate"]

    analysis = {
        "experiment_id": "f2_evaluation_v1",
        "model": manifest["model"],
        "unique_cases": len(cases),
        "total_responses": len(records),
        "results": results,
        "inference_minus_neutral_unsupported_absent_rate": (
            inference_rate - neutral_rate
        ),
        "unknown_cases_with_different_predictions": changed_cases,
        "limitations": [
            "Researcher-authored synthetic cases.",
            "One response per case per prompt.",
            "Small sample with no real-patient validation.",
            "Cases were not sampled to represent clinical practice.",
            (
                "Prompt comparison is descriptive, not causal proof "
                "of general model behavior."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    print("F2 ANALYSIS COMPLETE")
    print("Unique cases:", len(cases))
    print("Validated responses:", len(records))

    for condition in CONDITIONS:
        result = results[condition]
        print(
            condition,
            "unsupported absent:",
            f"{result['unsupported_absent_count']}/{result['unknown_case_count']}",
            "overall correct:",
            f"{result['overall_correct']}/{result['overall_n']}",
        )

    print(
        "Risk difference:",
        f"{100 * (inference_rate - neutral_rate):+.1f}",
        "percentage points",
    )
    print("Changed unknown cases:", changed_cases)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
