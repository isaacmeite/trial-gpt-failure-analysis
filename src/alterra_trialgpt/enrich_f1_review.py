import csv
import json
from pathlib import Path

from datasets import load_dataset

MANIFEST_PATH = Path("results/f1_case_manifest_v1.json")

INPUT_FILES = {
    Path("results/f1_disagreements_v1.csv"): Path(
        "results/f1_disagreements_enriched_v1.csv"
    ),
    Path("results/f1_review_priority_v1.csv"): Path(
        "results/f1_review_priority_enriched_v1.csv"
    ),
}


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError(f"Missing CSV columns: {path}")

        rows = list(reader)

    return fieldnames, rows


def enrich_rows(
    rows: list[dict],
    source_by_id: dict[int, dict],
    allowed_ids: set[int],
) -> list[dict]:
    enriched = []
    seen_ids = set()

    for row in rows:
        annotation_id = int(row["annotation_id"])

        if annotation_id in seen_ids:
            raise ValueError(f"Duplicate review annotation: {annotation_id}")
        seen_ids.add(annotation_id)

        if annotation_id not in allowed_ids:
            raise ValueError(f"Annotation outside frozen manifest: {annotation_id}")

        if annotation_id not in source_by_id:
            raise ValueError(f"Annotation missing from dataset: {annotation_id}")

        source = source_by_id[annotation_id]

        if row["patient_id"] != source["patient_id"]:
            raise ValueError(f"Patient mismatch for annotation {annotation_id}")

        if row["expert_label"] != source["expert_eligibility"]:
            raise ValueError(f"Expert label mismatch for annotation {annotation_id}")

        if source["criterion_type"] != "exclusion":
            raise ValueError(f"Non-exclusion criterion: {annotation_id}")

        enriched_row = dict(row)
        enriched_row["criterion_text"] = source["criterion_text"]

        enriched.append(enriched_row)

    return enriched


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    allowed_ids = set(manifest["annotation_ids"])

    dataset = load_dataset(
        "ncbi/TrialGPT-Criterion-Annotations",
        revision=manifest["dataset_revision"],
    )["train"]

    source_by_id = {
        row["annotation_id"]: dict(row)
        for row in dataset
        if row["annotation_id"] in allowed_ids
    }

    if set(source_by_id) != allowed_ids:
        raise ValueError("Frozen annotations missing from pinned dataset.")

    for input_path, output_path in INPUT_FILES.items():
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}. Preserve frozen review exports."
            )

        fieldnames, rows = read_csv(input_path)

        if "criterion_text" not in fieldnames:
            raise ValueError(f"Criterion column missing from {input_path}")

        if any(row["criterion_text"].strip() for row in rows):
            raise ValueError(f"Criterion column is already populated: {input_path}")

        enriched = enrich_rows(
            rows,
            source_by_id,
            allowed_ids,
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(enriched)

        print(f"Enriched {len(enriched)} rows: {output_path}")

    print("\nAll criterion texts retrieved from pinned dataset.")


if __name__ == "__main__":
    main()
