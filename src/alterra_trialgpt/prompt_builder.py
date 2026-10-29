from typing import Any


def build_exclusion_prompt(case: dict[str, Any]) -> str:
    """Build a classification prompt without revealing the expected label (leakage)"""

    prompt = f""" Evaluate whether the patient meets this clinical trial exclusion criterion

    Patient information
    {case["patient_note"]}

    Exclusion criterion:
    {case["criterion_text"]}

    Choose one label:

    - excluded: The evidence establishes that the exclusion criterion is met.
    - not excluded: The evidence establishes that the exclusion criterion is not met.
    - not enough information: The available evidence is insufficient to decide.
    - not applicable: The criterion's premise does not apply to this patient.

    Return a JSON object containing:
    - eligibility_label
    - explanation

"""
    return prompt


def build_trialgpt_adapted_prompt(case: dict[str, Any]) -> str:
    """Build a single-criterion prompt adapted from TrialGPT's rules"""

    return f"""
You are assisting with clinical trial recruitment.
Evaluate the patient's eligibility for the exclusion criterion below.

For this exclusion criterion:
1. First consider whether the criterion genuinely does not apply
   because its premise is absent. This should be uncommon.
2. Examine direct evidence in the patient note.
3. If direct evidence is missing, consider whether a well-documented
   clinical note could reasonably omit the condition.
4. If such an omission would be implausible, infer that the condition
   is absent. Otherwise, classify the evidence as insufficient.

Use exactly one exclusion label:
- excluded: The patient meets the exclusion criterion.
- not excluded: The patient does not meet the exclusion criterion.
- not enough information: The evidence cannot establish either outcome.
- not applicable: The criterion's premise does not apply to the patient.

Patient note:
{case["patient_note"]}

Exclusion criterion:
{case["criterion_text"]}

Return only a JSON object with:
- eligibility_label
- explanation
"""


def build_trialgpt_no_applicability_prompt(
    case: dict[str, Any],
) -> str:
    """Remove the initial applicability check from the adapted prompt"""

    original_prompt = build_trialgpt_adapted_prompt(case)

    removed_instruction = (
        "1. First consider whether the criterion genuinely does not apply\n"
        "   because its premise is absent. This should be uncommon.\n"
    )

    if original_prompt.count(removed_instruction) != 1:
        raise ValueError("Expected applicability instruction not found exactly once")

    return original_prompt.replace(removed_instruction, "", 1)


def build_neutral_clinical_prompt(case: dict[str, Any]) -> str:
    """Build a neutral clinical evidence-assessment prompt"""

    return f"""
Review the patient note and determine whether the specified
clinical condition is present.

Patient note:
{case["patient_note"]}

Clinical condition:
{case["clinical_condition"]}

Choose exactly one clinical_status:
- present
- absent
- unknown

Return only a JSON object containing:
- clinical_status
- explanation
"""


def build_inference_clinical_prompt(case: dict[str, Any]) -> str:
    """Add a missing-information inference instruction to the neutral prompt"""

    neutral_prompt = build_neutral_clinical_prompt(case)

    inference_instruction = """
When direct evidence is missing, consider whether a good clinical
note could plausibly omit this condition if it were present.
If such an omission would be implausible, infer that the condition
is absent. Otherwise, there is not enough information.
"""

    return neutral_prompt + inference_instruction
