from alterra_trialgpt.mechanism_cases import (
    build_age_experiment,
    build_generalization_cases,
    build_naturalistic_shock_cases,
    build_original_age_case,
    build_septic_shock_cases,
)


def test_age_experiment_has_three_unique_cases() -> None:

    cases = build_age_experiment()

    assert len(cases) == 3
    assert len({case["case_id"] for case in cases}) == 3


def test_age_expected_labels_for_each_cases() -> None:

    cases = build_age_experiment()

    expected_labels = {
        "age_58": "not excluded",
        "age_missing": "not enough information",
        "age_16": "excluded",
    }

    for case in cases:
        case_id = case["case_id"]
        actual_label = case["expected_label"]

        assert actual_label == expected_labels[case_id]
        assert {case["case_id"] for case in cases} == set(expected_labels)


def test_age_experiment_keeps_criterion_constant() -> None:

    cases = build_age_experiment()

    criterion_texts = {case["criterion_text"] for case in cases}

    assert len(criterion_texts) == 1
    assert criterion_texts == {"Participant is under 18 years old."}


def test_age_experiment_changes_patient_evidence() -> None:

    cases = build_age_experiment()

    notes_by_id = {case["case_id"]: case["patient_note"] for case in cases}

    assert notes_by_id["age_58"] == "Patient is 58 years old."
    assert notes_by_id["age_missing"] == "Patient's age is not documented."
    assert notes_by_id["age_16"] == "Patient is 16 years old."


def test_age_experiment_marks_synthetic_cases() -> None:

    cases = build_age_experiment()

    assert all(case["source_type"] == "synthetic" for case in cases)


def test_original_age_case() -> None:

    annotations = [
        {
            "annotation_id": 44,
            "patient_id": "sigir-20143",
            "note": "The patient is 58 years old.",
            "criterion_type": "exclusion",
            "criterion_text": "Participant is under 18 years old.",
            "expert_eligibility": "not excluded",
        }
    ]

    case = build_original_age_case(annotations)

    assert case["annotation_id"] == 44
    assert case["expected_label"] == "not excluded"
    assert case["source_type"] == "original_trialgpt"
    assert case["criterion_text"] == "Participant is under 18 years old."


def test_generalization_cases_are_unique() -> None:

    cases = build_generalization_cases()

    case_ids = {case["case_id"] for case in cases}

    assert len(cases) == 4
    assert len(case_ids) == 4


def test_generalization_cases_have_expected_metadata() -> None:

    cases = build_generalization_cases()

    for case in cases:
        assert case["criterion_type"] == "exclusion"
        assert case["expected_label"] == "not excluded"
        assert case["source_type"] == "synthetic"
        assert case["patient_note"].strip()
        assert case["criterion_text"].strip()


def test_septic_shock_cases_have_three_evidence_states() -> None:

    cases = build_septic_shock_cases()

    statuses = {case["expected_clinical_status"] for case in cases}

    assert len(cases) == 3
    assert statuses == {"present", "absent", "unknown"}


def test_septic_shock_cases_keep_criterion_constant() -> None:

    cases = build_septic_shock_cases()

    criteria = {case["criterion_text"] for case in cases}
    case_ids = {case["case_id"] for case in cases}

    assert criteria == {"Patients with septic shock from any source."}
    assert len(case_ids) == 3
    assert all(case["source_type"] == "synthetic" for case in cases)


def test_naturalistic_shock_cases_have_expected_status() -> None:
    cases = build_naturalistic_shock_cases()

    assert len(cases) == 2
    assert {case["case_id"] for case in cases} == {
        "shock_skin_lesion_naturalistic",
        "shock_possible_infection_naturalistic",
    }
    assert all(case["expected_clinical_status"] == "unknown" for case in cases)


def test_naturalistic_shock_cases_keep_criterion_constant() -> None:
    cases = build_naturalistic_shock_cases()

    assert {case["criterion_text"] for case in cases} == {
        "Patients with septic shock from any source."
    }

    assert all(case["source_type"] == "synthetic" for case in cases)
