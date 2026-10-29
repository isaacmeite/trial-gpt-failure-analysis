from alterra_trialgpt.mechanism_cases import (
    build_age_experiment,
    build_septic_shock_cases,
)
from alterra_trialgpt.prompt_builder import (
    build_exclusion_prompt,
    build_inference_clinical_prompt,
    build_neutral_clinical_prompt,
    build_trialgpt_adapted_prompt,
    build_trialgpt_no_applicability_prompt,
)


def test_prompt_contains_evidence_but_not_expected_label() -> None:

    case = build_age_experiment()[0]

    prompt = build_exclusion_prompt(case)

    assert case["patient_note"] in prompt
    assert case["criterion_text"] in prompt
    assert "expected_label" not in prompt
    assert "expected_label" not in prompt
    assert "eligibility_label" in prompt
    assert "explanation" in prompt


def test_adapted_prompt_contains_case_evidence() -> None:
    case = build_age_experiment()[0]

    prompt = build_trialgpt_adapted_prompt(case)

    assert case["patient_note"] in prompt
    assert case["criterion_text"] in prompt
    assert "expected_label" not in prompt


def test_adapted_prompt_contains_historical_decision_rules() -> None:
    case = build_age_experiment()[0]

    prompt = build_trialgpt_adapted_prompt(case)

    assert "premise" in prompt
    assert "direct evidence" in prompt
    assert "well-documented" in prompt


def test_ablation_removes_only_applicability_instruction() -> None:
    case = build_age_experiment()[0]

    original = build_trialgpt_adapted_prompt(case)
    ablated = build_trialgpt_no_applicability_prompt(case)

    removed_instruction = (
        "1. First consider whether the criterion genuinely does not apply\n"
        "   because its premise is absent. This should be uncommon.\n"
    )

    assert original.replace(removed_instruction, "", 1) == ablated
    assert removed_instruction not in ablated

    assert case["patient_note"] in ablated
    assert case["criterion_text"] in ablated
    assert "expected_label" not in ablated


def test_neutral_clinical_prompt_has_no_reference_leakage() -> None:
    case = build_septic_shock_cases()[2]
    case["clinical_condition"] = "Septic shock"

    prompt = build_neutral_clinical_prompt(case)

    assert case["patient_note"] in prompt
    assert "clinical_status" in prompt
    assert "expected_clinical_status" not in prompt
    assert "If such an omission would be implausible" not in prompt


def test_inference_prompt_preserves_neutral_prompt() -> None:
    case = build_septic_shock_cases()[2]
    case["clinical_condition"] = "Septic shock"

    neutral = build_neutral_clinical_prompt(case)
    inference = build_inference_clinical_prompt(case)

    assert inference.startswith(neutral)
    assert "infer that the condition" in inference
    assert case["patient_note"] in inference
