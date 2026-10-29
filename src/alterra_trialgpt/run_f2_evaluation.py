import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from alterra_trialgpt.prompt_builder import (
    build_inference_clinical_prompt,
    build_neutral_clinical_prompt,
)

MANIFEST_PATH = Path("results/f2_case_manifest_v1.json")
OUTPUT_PATH = Path("results/f2_evaluation_predictions_v1.jsonl")

PROMPT_BUILDERS = {
    "clinical_neutral_v1": build_neutral_clinical_prompt,
    "clinical_inference_v1": build_inference_clinical_prompt,
}

VALID_STATUSES = {"present", "absent", "unknown"}


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def job_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        record["case_id"],
        record["prompt_version"],
        record["repetition"],
    )


def build_plan(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = manifest["conditions"]

    if conditions != [
        "clinical_neutral_v1",
        "clinical_inference_v1",
    ]:
        raise ValueError("Unexpected frozen F2 prompt conditions.")

    if manifest["repetitions_per_condition"] != 1:
        raise ValueError("F2 protocol requires one repetition.")

    cases = manifest["cases"]

    if len(cases) != manifest["case_count"]:
        raise ValueError("Case count does not match manifest.")

    case_ids = [case["case_id"] for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate F2 case IDs.")

    plan = []

    for case in cases:
        if case["expected_status"] not in VALID_STATUSES:
            raise ValueError("Unexpected reference status.")

        for condition in conditions:
            plan.append(
                {
                    "case_id": case["case_id"],
                    "prompt_version": condition,
                    "repetition": 1,
                    "expected_status": case["expected_status"],
                }
            )

    return plan


def load_existing() -> dict[tuple[str, str, int], dict[str, Any]]:
    existing = {}

    if not OUTPUT_PATH.exists():
        return existing

    lines = OUTPUT_PATH.read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid saved JSONL line {line_number}") from error

        key = job_key(record)

        if key in existing:
            raise ValueError(f"Duplicate saved job: {key}")

        existing[key] = record

    return existing


def prepare_case(case: dict[str, Any]) -> dict[str, Any]:
    """Provide aliases expected by different prompt-builder interfaces."""
    return {
        **case,
        "patient_note": case["note"],
        "clinical_note": case["note"],
        "clinical_condition": case["condition"],
        "target_condition": case["condition"],
    }


def validate_output(raw: str) -> tuple[dict | None, list[str]]:
    try:
        prediction = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, ["invalid_json"]

    if not isinstance(prediction, dict):
        return None, ["invalid_prediction_object"]

    violations = []

    if prediction.get("clinical_status") not in VALID_STATUSES:
        violations.append("invalid_clinical_status")

    explanation = prediction.get("explanation")

    if not isinstance(explanation, str) or not explanation.strip():
        violations.append("invalid_explanation")

    return prediction, violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make API calls; otherwise perform a dry run.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum number of new calls.",
    )
    args = parser.parse_args()

    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive.")

    manifest = load_manifest()
    plan = build_plan(manifest)

    existing = load_existing()

    planned_keys = {job_key(job) for job in plan}

    if len(planned_keys) != len(plan):
        raise ValueError("Duplicate jobs in F2 plan.")

    if not set(existing).issubset(planned_keys):
        raise ValueError("Saved predictions include unexpected jobs.")

    pending = [job for job in plan if job_key(job) not in existing]

    print("F2 EVALUATION")
    print("Frozen cases:", len(manifest["cases"]))
    print("Planned API calls:", len(plan))
    print("Previously recorded:", len(existing))
    print("Pending:", len(pending))

    cases_by_id = {case["case_id"]: case for case in manifest["cases"]}

    jobs_to_run = pending[: args.max_calls]

    if args.max_calls is None:
        jobs_to_run = pending

    prepared = []

    for job in jobs_to_run:
        case = cases_by_id[job["case_id"]]
        builder = PROMPT_BUILDERS[job["prompt_version"]]

        prompt = builder(prepare_case(case))

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty prompt: {job_key(job)}")

        if case["note"] not in prompt:
            raise ValueError(f"Patient note missing from prompt: {job_key(job)}")

        if case["condition"] not in prompt:
            raise ValueError(f"Target condition missing from prompt: {job_key(job)}")

        prepared.append((job, prompt))

    print("Selected prompts validated:", len(prepared))

    if not args.execute:
        print("DRY RUN: No OpenAI API calls made.")
        return

    client = OpenAI()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for job, prompt in prepared:
        response = client.responses.create(
            model=manifest["model"],
            input=prompt,
            text={"format": {"type": "json_object"}},
            store=False,
        )

        raw_output = response.output_text
        prediction, violations = validate_output(raw_output)

        result = {
            **job,
            "experiment_id": "f2_evaluation_v1",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "requested_model": manifest["model"],
            "response_model": response.model,
            "response_id": response.id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output": raw_output,
            "prediction": prediction,
            "validation_violations": violations,
        }

        # Persist each completed response immediately
        with OUTPUT_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")
            file.flush()

        status = (
            prediction.get("clinical_status") if isinstance(prediction, dict) else None
        )

        print(
            f"Saved {job['case_id']} | "
            f"{job['prompt_version']} | "
            f"status={status} | "
            f"violations={violations}"
        )

    print("\nResults saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
