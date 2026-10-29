from alterra_trialgpt.prompt_builder import (
    build_inference_clinical_prompt,
    build_neutral_clinical_prompt,
)


def test_f2_prompts_use_case_specific_condition() -> None:
    case = {
        "patient_note": "A patient attends a dermatology appointment.",
        "clinical_condition": "chronic kidney disease",
    }

    neutral = build_neutral_clinical_prompt(case)
    inference = build_inference_clinical_prompt(case)

    for prompt in (neutral, inference):
        assert "chronic kidney disease" in prompt
        assert "A patient attends a dermatology appointment." in prompt
        assert "Clinical condition:\nSeptic shock" not in prompt


def test_inference_instruction_is_only_added_to_inference_prompt() -> None:
    case = {
        "patient_note": "The relevant history is not documented.",
        "clinical_condition": "deep vein thrombosis",
    }

    neutral = build_neutral_clinical_prompt(case)
    inference = build_inference_clinical_prompt(case)

    assert "If such an omission would be implausible" not in neutral
    assert "If such an omission would be implausible" in inference
