import pytest

from alterra_trialgpt.f1_evaluation_plan import (
    build_f1_evaluation_plan,
)


def test_plan_pairs_every_case_with_both_prompts() -> None:

    manifest = {
        "annotation_ids": [10, 20],
        "patient_ids": ["patient_a", "patient_b"],
    }

    protocol = {
        "experimental_conditions": [
            "trialgpt_adapted_v1",
            "trialgpt_no_applicability_v1",
        ],
        "repetitions_per_condition": 1,
    }

    annotations = [
        {
            "annotation_id": 10,
            "patient_id": "patient_a",
            "criterion_type": "exclusion",
            "expert_eligibility": "not excluded",
        },
        {
            "annotation_id": 20,
            "patient_id": "patient_b",
            "criterion_type": "exclusion",
            "expert_eligibility": "not applicable",
        },
    ]

    plan = build_f1_evaluation_plan(
        manifest,
        protocol,
        annotations,
    )

    assert len(plan) == 4

    for annotation_id in [10, 20]:
        conditions = {
            job["prompt_version"]
            for job in plan
            if job["annotation_id"] == annotation_id
        }

        assert conditions == {
            "trialgpt_adapted_v1",
            "trialgpt_no_applicability_v1",
        }


def test_plan_rejects_missing_annotations() -> None:
    manifest = {
        "annotation_ids": [10],
        "patient_ids": ["patient_a"],
    }

    protocol = {
        "experimental_conditions": ["trialgpt_adapted_v1"],
        "repetitions_per_condition": 1,
    }

    with pytest.raises(ValueError, match="Dataset does not match"):
        build_f1_evaluation_plan(
            manifest,
            protocol,
            annotations=[],
        )
