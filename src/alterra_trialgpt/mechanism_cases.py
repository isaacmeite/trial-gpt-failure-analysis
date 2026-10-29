from typing import Any


def build_age_experiment() -> list[dict[str, Any]]:
    """Build age-based counterfactual exclusion cases"""

    case_a = {
        "case_id": "age_58",
        "patient_note": "Patient is 58 years old.",
        "criterion_type": "exclusion",
        "criterion_text": "Participant is under 18 years old.",
        "expected_label": "not excluded",
        "source_type": "synthetic",
    }

    case_b = {
        "case_id": "age_missing",
        "patient_note": "Patient's age is not documented.",
        "criterion_type": "exclusion",
        "criterion_text": "Participant is under 18 years old.",
        "expected_label": "not enough information",
        "source_type": "synthetic",
    }

    case_c = {
        "case_id": "age_16",
        "patient_note": "Patient is 16 years old.",
        "criterion_type": "exclusion",
        "criterion_text": "Participant is under 18 years old.",
        "expected_label": "excluded",
        "source_type": "synthetic",
    }

    return [case_a, case_b, case_c]


def build_original_age_case(
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an experimental case from original TrialGPT annotation 44"""

    for annotation in annotations:
        if annotation["annotation_id"] == 44:
            return {
                "case_id": "original_age_44",
                "annotation_id": annotation["annotation_id"],
                "patient_id": annotation["patient_id"],
                "patient_note": annotation["note"],
                "criterion_type": annotation["criterion_type"],
                "criterion_text": annotation["criterion_text"],
                "expected_label": annotation["expert_eligibility"],
                "source_type": "original_trialgpt",
            }

    raise ValueError("Annotation 44 not found in dataset")


def build_generalization_cases() -> list[dict[str, Any]]:
    """Build prespecified cross-criterion applicability probes"""

    examples = [
        (
            "smoking_negative",
            "The patient has never smoked tobacco.",
            "Current tobacco smoking.",
        ),
        (
            "diabetes_negative",
            "The patient has no history of diabetes mellitus.",
            "History of diabetes mellitus.",
        ),
        (
            "transplant_negative",
            "The patient has never received an organ transplant.",
            "Prior solid-organ transplantation.",
        ),
        (
            "age_45_under_21",
            "The patient is 45 years old.",
            "Participant is under 21 years old.",
        ),
    ]

    return [
        {
            "case_id": case_id,
            "patient_note": patient_note,
            "criterion_type": "exclusion",
            "criterion_text": criterion_text,
            "expected_label": "not excluded",
            "source_type": "synthetic",
        }
        for case_id, patient_note, criterion_text in examples
    ]


def build_septic_shock_cases() -> list[dict[str, Any]]:
    """Build synthetic cases testing clinical evidence sufficiency."""

    criterion = "Patients with septic shock from any source."

    examples = [
        (
            "shock_positive",
            "The treating physician explicitly documents septic shock.",
            "present",
        ),
        (
            "shock_negative",
            "The treating physician explicitly rules out septic shock.",
            "absent",
        ),
        (
            "shock_unknown",
            (
                "The patient has diabetes and a worsening, "
                "oozing lesion on the lower leg. "
                "No shock assessment or hemodynamic measurements "
                "are provided."
            ),
            "unknown",
        ),
    ]

    return [
        {
            "case_id": case_id,
            "patient_note": patient_note,
            "criterion_type": "exclusion",
            "criterion_text": criterion,
            "expected_clinical_status": expected_status,
            "source_type": "synthetic",
        }
        for case_id, patient_note, expected_status in examples
    ]


def build_naturalistic_shock_cases() -> list[dict[str, Any]]:
    """Build new unknown-evidence cases for generalization testing"""

    criterion = "Patients with septic shock from any source."

    examples = [
        (
            "shock_skin_lesion_naturalistic",
            (
                "A 64-year-old woman with diabetes has a painful "
                "skin lesion on her lower leg. The lesion has worsened "
                "and is now oozing despite topical treatment."
            ),
        ),
        (
            "shock_possible_infection_naturalistic",
            (
                "A 52-year-old patient presents with fever, "
                "tachycardia, and a suspected bacterial infection. "
                "Further clinical evaluation is being arranged."
            ),
        ),
    ]

    return [
        {
            "case_id": case_id,
            "patient_note": patient_note,
            "criterion_type": "exclusion",
            "criterion_text": criterion,
            "expected_clinical_status": "unknown",
            "source_type": "synthetic",
        }
        for case_id, patient_note in examples
    ]
