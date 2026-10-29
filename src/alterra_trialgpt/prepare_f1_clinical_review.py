import csv
import json
from pathlib import Path

from datasets import load_dataset

MANIFEST_PATH = Path("results/f1_case_manifest_v1.json")
REVIEW_PATH = Path("results/f1_review_priority_enriched_v1.csv")
OUTPUT_PATH = Path("results/f1_clinical_review_v1.md")


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Review document already exists: {OUTPUT_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    with REVIEW_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        review_rows = list(csv.DictReader(file))

    dataset = load_dataset(
        "ncbi/TrialGPT-Criterion-Annotations",
        revision=manifest["dataset_revision"],
    )["train"]

    source_by_id = {row["annotation_id"]: dict(row) for row in dataset}

    sections = [
        "# F1 Targeted Clinical-Evidence Review",
        "",
        "Researcher review; not independent clinician adjudication.",
        "",
        "## Review rubric",
        "",
        "- `clear_model_error`: Original evidence supports the expert label.",
        "- `reference_label_ambiguous`: Multiple interpretations are defensible.",
        "- `possible_reference_error`: Expert label appears inconsistent.",
        "- `insufficient_context`: Cannot determine from available evidence.",
        "",
        "Do not infer that an undocumented condition is clinically absent.",
        "Distinguish a wrong eligibility label from an unsupported explanation.",
        "",
    ]

    for review in review_rows:
        annotation_id = int(review["annotation_id"])
        source = source_by_id[annotation_id]

        if annotation_id not in manifest["annotation_ids"]:
            raise ValueError("Case outside frozen evaluation.")

        if source["patient_id"] != review["patient_id"]:
            raise ValueError("Patient mismatch.")

        sections.extend(
            [
                f"## Annotation {annotation_id}",
                "",
                f"**Review category:** {review['review_category']}",
                "",
                f"**Criterion:** {source['criterion_text']}",
                "",
                f"**Expert label:** `{review['expert_label']}`",
                "",
                f"**Adapted prediction:** `{review['baseline_prediction']}`",
                "",
                f"**Ablated prediction:** `{review['ablation_prediction']}`",
                "",
                "### Original patient note",
                "",
                "```text",
                source["note"],
                "```",
                "",
                "### Adapted-prompt explanation",
                "",
                review["baseline_explanation"],
                "",
                "### Ablated-prompt explanation",
                "",
                review["ablation_explanation"],
                "",
                "### Reviewer assessment",
                "",
                "**Decision:** TODO",
                "",
                "**Supporting note evidence:** TODO",
                "",
                "**Clinical relevance:** TODO",
                "",
                "**Remaining uncertainty:** TODO",
                "",
                "---",
                "",
            ]
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )

    print("Created:", OUTPUT_PATH)
    print("Cases prepared:", len(review_rows))
    print("No API calls made.")


if __name__ == "__main__":
    main()
