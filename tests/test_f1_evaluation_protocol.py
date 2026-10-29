import json
from pathlib import Path


def test_f1_protocol_matches_frozen_manifest() -> None:
    manifest = json.loads(
        Path("results/f1_case_manifest_v1.json").read_text(encoding="utf-8")
    )

    protocol = json.loads(
        Path("results/f1_evaluation_protocol_v1.json").read_text(encoding="utf-8")
    )

    assert len(manifest["annotation_ids"]) == 148
    assert len(manifest["patient_ids"]) == 10

    assert manifest["reference_label_counts"]["not excluded"] == 105
    assert manifest["reference_label_counts"]["not applicable"] == 11

    assert (
        protocol["primary_population"]["expected_count"]
        == manifest["reference_label_counts"]["not excluded"]
    )

    assert protocol["repetitions_per_condition"] == 1
    assert len(protocol["experimental_conditions"]) == 2
