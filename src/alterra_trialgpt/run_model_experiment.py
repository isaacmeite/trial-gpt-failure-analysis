import json
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from alterra_trialgpt.mechanism_cases import (
    build_generalization_cases,
)
from alterra_trialgpt.prediction_validation import validate_prediction
from alterra_trialgpt.prompt_builder import (
    build_exclusion_prompt,
    build_trialgpt_adapted_prompt,
    build_trialgpt_no_applicability_prompt,
)

MODEL_ID = "gpt-5-mini-2025-08-07"

PROMPT_BUILDERS = {
    "clarified_v1": build_exclusion_prompt,
    "trialgpt_adapted_v1": build_trialgpt_adapted_prompt,
    "trialgpt_no_applicability_v1": build_trialgpt_no_applicability_prompt,
}

EXPERIMENT_ID = "applicability_generalization_v1"
REPETITIONS = 2

RESULTS_PATH = Path("results/applicability_generalization_v1.jsonl")


def main() -> None:
    """Run a paired prompt-sensitivity experiment on the age-58 case."""

    cases = build_generalization_cases()

    client = OpenAI()
    run_id = datetime.now(UTC).isoformat()

    prompt_versions = [
        "trialgpt_adapted_v1",
        "trialgpt_no_applicability_v1",
    ]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for repetition in range(REPETITIONS):
            # Alternate prompt order between repetitions
            versions = (
                prompt_versions
                if repetition % 2 == 0
                else list(reversed(prompt_versions))
            )

            for prompt_version in versions:
                prompt = PROMPT_BUILDERS[prompt_version](case)

                print(
                    f"\nRepetition {repetition + 1}/{REPETITIONS} | "
                    f"Prompt: {prompt_version}"
                )

                response = client.responses.create(
                    model=MODEL_ID,
                    input=prompt,
                    text={"format": {"type": "json_object"}},
                    store=False,
                )

                raw_output = response.output_text
                prediction = None
                violations = []

                try:
                    prediction = json.loads(raw_output)
                except (json.JSONDecodeError, TypeError):
                    violations.append("invalid_json")
                else:
                    if not isinstance(prediction, dict):
                        violations.append("invalid_prediction_object")
                    else:
                        violations.extend(validate_prediction(prediction))

                label_match = None

                if not violations and isinstance(prediction, dict):
                    label_match = (
                        prediction["eligibility_label"] == case["expected_label"]
                    )

                result = {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": run_id,
                    "repetition": repetition + 1,
                    "case_id": case["case_id"],
                    "source_type": case["source_type"],
                    "prompt_version": prompt_version,
                    "prompt": prompt,
                    "requested_model": MODEL_ID,
                    "model": response.model,
                    "response_id": response.id,
                    "response_format": "json_object",
                    "expected_label": case["expected_label"],
                    "raw_output": raw_output,
                    "prediction": prediction,
                    "validation_violations": violations,
                    "label_match": label_match,
                }

                with RESULTS_PATH.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(result) + "\n")

                print(f"Case: {case['case_id']}")
                print(f"Model: {response.model}")
                print(f"Prediction: {prediction}")
                print(f"Validation violations: {violations}")
                print(f"Expected label: {case['expected_label']}")
                print(f"Labels agree: {label_match}")

    print(f"\nExperiment finished. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
