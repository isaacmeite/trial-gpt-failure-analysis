import json
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from alterra_trialgpt.clinical_prediction_validation import (
    validate_clinical_prediction,
)
from alterra_trialgpt.mechanism_cases import (
    build_naturalistic_shock_cases,
)
from alterra_trialgpt.prompt_builder import (
    build_inference_clinical_prompt,
    build_neutral_clinical_prompt,
)

MODEL_ID = "gpt-5-mini-2025-08-07"
EXPERIMENT_ID = "clinical_naturalistic_generalization_v1"
REPETITIONS = 2

RESULTS_PATH = Path("results/clinical_naturalistic_generalization_v1.jsonl")

PROMPT_BUILDERS = {
    "clinical_neutral_v1": build_neutral_clinical_prompt,
    "clinical_inference_v1": build_inference_clinical_prompt,
}


def main() -> None:
    """Evaluate clinical evidence reasoning under two prompt conditions"""

    cases = build_naturalistic_shock_cases()
    client = OpenAI()

    run_id = datetime.now(UTC).isoformat()
    prompt_versions = list(PROMPT_BUILDERS)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for repetition in range(REPETITIONS):
            versions = (
                prompt_versions
                if repetition % 2 == 0
                else list(reversed(prompt_versions))
            )
            for prompt_version in versions:
                prompt = PROMPT_BUILDERS[prompt_version](case)

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
                        violations.extend(validate_clinical_prediction(prediction))

                status_match = None
                unsupported_definitive_claim = None

                if not violations:
                    predicted_status = prediction["clinical_status"]

                    status_match = predicted_status == case["expected_clinical_status"]

                    if case["expected_clinical_status"] == "unknown":
                        unsupported_definitive_claim = predicted_status in {
                            "present",
                            "absent",
                        }

                result = {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": run_id,
                    "case_id": case["case_id"],
                    "source_type": case["source_type"],
                    "repetition": repetition + 1,
                    "prompt_version": prompt_version,
                    "prompt": prompt,
                    "request_model": MODEL_ID,
                    "model": response.model,
                    "response_id": response.id,
                    "response_format": "json_object",
                    "expected_clinical_status": case["expected_clinical_status"],
                    "raw_output": raw_output,
                    "prediction": prediction,
                    "validation_violations": violations,
                    "status_match": status_match,
                    "unsupported_definitive_claim": unsupported_definitive_claim,
                }

                with RESULTS_PATH.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(result) + "\n")

                print(
                    f"\nCase: {case['case_id']} | "
                    f"Repetition: {repetition + 1}/{REPETITIONS} | "
                    f"Prompt: {prompt_version}"
                )
                print(f"Prediction: {prediction}")
                print(f"Validation violations: {violations}")
                print(f"Expected status: {case['expected_clinical_status']}")
                print(f"Status match: {status_match}")
    print(f"\nExperiment completed. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
