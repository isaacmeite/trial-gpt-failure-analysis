import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI

from alterra_trialgpt.f1_evaluation_plan import (
    build_f1_evaluation_plan,
)
from alterra_trialgpt.prompt_builder import (
    build_trialgpt_adapted_prompt,
    build_trialgpt_no_applicability_prompt,
)

MANIFEST_PATH = Path("results/f1_case_manifest_v1.json")
PROTOCOL_PATH = Path("results/f1_evaluation_protocol_v1.json")
RESULTS_PATH = Path("results/f1_evaluation_predictions_v1.jsonl")

PROMPT_BUILDERS = {
    "trialgpt_adapted_v1": build_trialgpt_adapted_prompt,
    "trialgpt_no_applicability_v1": build_trialgpt_no_applicability_prompt,
}

VALID_LABELS = {
    "excluded",
    "not excluded",
    "not applicable",
    "not enough information",
}


def job_key(job: dict[str, Any]) -> tuple[int, str, int]:
    return (
        job["annotation_id"],
        job["prompt_version"],
        job["repetition"],
    )


def load_existing_results() -> dict[tuple[int, str, int], dict[str, Any]]:
    """Read saved predictions so completed jobs are not repeated."""
    completed = {}

    if not RESULTS_PATH.exists():
        return completed

    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        result = json.loads(line)
        key = job_key(result)

        if key in completed:
            raise ValueError(f"Duplicate saved result: {key}")

        completed[key] = result

    return completed


def build_case(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt a dataset record to our existing prompt-builder interface."""
    return {
        "patient_note": row["note"],
        "note": row["note"],
        "criterion_text": row["criterion_text"],
        "criterion_type": row["criterion_type"],
        "annotation_id": row["annotation_id"],
    }


def validate_response(raw_output: str) -> tuple[Any, list[str]]:
    """Parse and validate the expected exclusion-label response."""
    try:
        prediction = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None, ["invalid_json"]

    if not isinstance(prediction, dict):
        return prediction, ["invalid_prediction_object"]

    violations = []

    if prediction.get("eligibility_label") not in VALID_LABELS:
        violations.append("invalid_eligibility_label")

    explanation = prediction.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        violations.append("invalid_explanation")

    return prediction, violations


def prepare_evaluation() -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    dataset = load_dataset(
        "ncbi/TrialGPT-Criterion-Annotations",
        revision=manifest["dataset_revision"],
    )["train"]

    annotations = [dict(row) for row in dataset]
    plan = build_f1_evaluation_plan(manifest, protocol, annotations)

    rows_by_id = {
        row["annotation_id"]: row
        for row in annotations
        if row["annotation_id"] in set(manifest["annotation_ids"])
    }

    conditions = set(protocol["experimental_conditions"])
    if conditions != set(PROMPT_BUILDERS):
        raise ValueError("Frozen conditions do not match prompt builders.")

    return plan, rows_by_id, protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the OpenAI API; otherwise dry-run.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum new API calls to make in this invocation.",
    )
    args = parser.parse_args()

    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive.")

    plan, rows_by_id, protocol = prepare_evaluation()
    completed = load_existing_results()

    planned_keys = {job_key(job) for job in plan}
    unexpected = set(completed) - planned_keys
    if unexpected:
        raise ValueError(
            f"Results file contains jobs outside frozen plan: {unexpected}"
        )

    pending = [job for job in plan if job_key(job) not in completed]

    print("Frozen planned calls:", len(plan))
    print("Previously recorded calls:", len(completed))
    print("Pending calls:", len(pending))

    if not args.execute:
        print("DRY RUN: No OpenAI API calls will be made.")
        return

    limit = args.max_calls or len(pending)
    jobs_to_run = pending[:limit]

    # Validating prompts before starting paid model calls
    prepared = []
    for job in jobs_to_run:
        row = rows_by_id[job["annotation_id"]]
        case = build_case(row)

        prompt = PROMPT_BUILDERS[job["prompt_version"]](case)

        if not prompt.strip():
            raise ValueError("Empty prompt generated.")

        if row["note"] not in prompt:
            raise ValueError(f"Patient note missing from prompt: {job_key(job)}")

        if row["criterion_text"] not in prompt:
            raise ValueError(f"Criterion missing from prompt: {job_key(job)}")

        prepared.append((job, prompt))

    client = OpenAI()
    invocation_id = datetime.now(UTC).isoformat()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for job, prompt in prepared:
        response = client.responses.create(
            model=protocol["model"],
            input=prompt,
            text={"format": {"type": "json_object"}},
            store=False,
        )

        raw_output = response.output_text
        prediction, violations = validate_response(raw_output)

        result = {
            **job,
            "experiment_id": "f1_evaluation_v1",
            "invocation_id": invocation_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "requested_model": protocol["model"],
            "response_model": response.model,
            "response_id": response.id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output": raw_output,
            "prediction": prediction,
            "validation_violations": violations,
        }

        with RESULTS_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")
            file.flush()

        print(
            f"Saved annotation={job['annotation_id']} "
            f"prompt={job['prompt_version']} "
            f"prediction={prediction.get('eligibility_label') if isinstance(prediction, dict) else None} "
            f"violations={violations}"
        )

    print(f"\nRecorded results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
