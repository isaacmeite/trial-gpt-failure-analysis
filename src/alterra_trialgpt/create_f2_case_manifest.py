import json
from collections import Counter
from pathlib import Path

OUTPUT_PATH = Path("results/f2_case_manifest_v1.json")


CASES = [
    # UNKNOWN: Relevant evidence is not documented.
    {
        "case_id": "f2_u01",
        "condition": "septic shock",
        "note": (
            "A patient presents with fever and fatigue. "
            "The note documents symptom onset and recent travel. "
            "No blood pressure, perfusion assessment, lactate, "
            "or vasopressor information is provided."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u02",
        "condition": "deep vein thrombosis",
        "note": (
            "A patient attends a follow-up visit for knee pain. "
            "The note discusses analgesic use and physiotherapy. "
            "There is no venous examination or thrombosis assessment."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u03",
        "condition": "drug allergy to penicillin",
        "note": (
            "A patient presents for evaluation of a sore throat. "
            "The note records temperature and duration of symptoms. "
            "Medication allergy history is not recorded."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u04",
        "condition": "chronic kidney disease",
        "note": (
            "A patient attends a dermatology appointment for a skin lesion. "
            "The note describes lesion size and appearance. "
            "No renal history, creatinine, or kidney-function assessment "
            "is provided."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u05",
        "condition": "pregnancy",
        "note": (
            "A 28-year-old patient attends an appointment for an ankle "
            "injury. The note describes tenderness and swelling. "
            "Pregnancy status, testing, and reproductive history "
            "are not documented."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u06",
        "condition": "active tuberculosis",
        "note": (
            "A patient attends a routine ophthalmology appointment. "
            "The note describes visual acuity and a retinal examination. "
            "No tuberculosis screening, symptoms review, or prior "
            "tuberculosis history is recorded."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u07",
        "condition": "heart failure",
        "note": (
            "A patient is seen for a hand laceration. "
            "The note describes wound depth and treatment. "
            "Cardiac history, volume-status examination, and "
            "cardiac investigations are not documented."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u08",
        "condition": "gastrointestinal bleeding",
        "note": (
            "A patient attends a visit for chronic shoulder discomfort. "
            "The note describes range of motion and pain severity. "
            "Gastrointestinal symptoms, bleeding history, and "
            "relevant laboratory results are not addressed."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u09",
        "condition": "immunodeficiency",
        "note": (
            "A patient is evaluated for an uncomplicated finger injury. "
            "The note records the mechanism and local examination. "
            "Immune-system history, recurrent infections, and "
            "immunosuppressive treatments are not discussed."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u10",
        "condition": "atrial fibrillation",
        "note": (
            "A patient attends a physiotherapy review for lower-back pain. "
            "The note describes mobility and exercise tolerance. "
            "No rhythm assessment, ECG, or arrhythmia history "
            "is documented."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u11",
        "condition": "history of anaphylaxis",
        "note": (
            "A patient attends a consultation for a minor burn. "
            "The note describes the burn and dressing instructions. "
            "Previous severe allergic reactions are not assessed."
        ),
        "expected_status": "unknown",
    },
    {
        "case_id": "f2_u12",
        "condition": "chronic liver disease",
        "note": (
            "A patient presents for evaluation of a wrist fracture. "
            "The note describes imaging and immobilization. "
            "Liver history, liver-function tests, and relevant "
            "systemic findings are not documented."
        ),
        "expected_status": "unknown",
    },
    # PRESENT: The note explicitly establishes the condition.
    {
        "case_id": "f2_p01",
        "condition": "atrial fibrillation",
        "note": (
            "A 12-lead ECG obtained today demonstrates atrial "
            "fibrillation, documented and confirmed by the treating clinician."
        ),
        "expected_status": "present",
    },
    {
        "case_id": "f2_p02",
        "condition": "deep vein thrombosis",
        "note": (
            "Compression ultrasound demonstrates an acute left "
            "popliteal vein thrombosis. The treating clinician "
            "documents confirmed deep vein thrombosis."
        ),
        "expected_status": "present",
    },
    {
        "case_id": "f2_p03",
        "condition": "pregnancy",
        "note": (
            "The patient has a documented intrauterine pregnancy "
            "confirmed by today's obstetric ultrasound."
        ),
        "expected_status": "present",
    },
    {
        "case_id": "f2_p04",
        "condition": "chronic kidney disease",
        "note": (
            "The treating clinician documents established chronic "
            "kidney disease with persistently reduced renal "
            "function for more than six months."
        ),
        "expected_status": "present",
    },
    # ABSENT: The note explicitly supports a negative status.
    {
        "case_id": "f2_a01",
        "condition": "atrial fibrillation",
        "note": (
            "The clinician reviews the current ECG and documents "
            "normal sinus rhythm with no atrial fibrillation. "
            "This is an assessment of the current rhythm only."
        ),
        "expected_status": "absent",
    },
    {
        "case_id": "f2_a02",
        "condition": "deep vein thrombosis",
        "note": (
            "A complete compression ultrasound of the symptomatic "
            "leg is negative for deep vein thrombosis. "
            "The clinician documents no current DVT."
        ),
        "expected_status": "absent",
    },
    {
        "case_id": "f2_a03",
        "condition": "pregnancy",
        "note": (
            "The treating clinician documents that the patient "
            "is not currently pregnant, supported by a negative "
            "pregnancy test obtained today."
        ),
        "expected_status": "absent",
    },
    {
        "case_id": "f2_a04",
        "condition": "history of anaphylaxis",
        "note": (
            "The clinician specifically asks about previous "
            "anaphylactic reactions. The patient denies any "
            "history of anaphylaxis, and this is documented."
        ),
        "expected_status": "absent",
    },
]


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Frozen manifest already exists: {OUTPUT_PATH}")

    case_ids = [case["case_id"] for case in CASES]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate F2 case IDs.")

    counts = Counter(case["expected_status"] for case in CASES)

    if counts != {"unknown": 12, "present": 4, "absent": 4}:
        raise ValueError(f"Unexpected evidence-status counts: {counts}")

    for case in CASES:
        if not case["condition"].strip() or not case["note"].strip():
            raise ValueError(f"Incomplete case: {case['case_id']}")

    manifest = {
        "version": "f2_v1",
        "case_source": "researcher_authored_synthetic_cases",
        "frozen_before_model_evaluation": True,
        "model": "gpt-5-mini-2025-08-07",
        "conditions": [
            "clinical_neutral_v1",
            "clinical_inference_v1",
        ],
        "repetitions_per_condition": 1,
        "primary_population": "unknown",
        "primary_error": ("Model outputs absent when the evidence status is unknown"),
        "case_count": len(CASES),
        "expected_status_counts": dict(sorted(counts.items())),
        "scoring_rubric": {
            "present": "The note explicitly supports the condition.",
            "absent": (
                "The note explicitly supports a negative status "
                "for the specified condition and assessment scope."
            ),
            "unknown": (
                "The available note does not establish either presence or absence."
            ),
        },
        "limitations": [
            "Synthetic, researcher-designed examples.",
            "Not a representative sample of clinical records.",
            (
                "Expected statuses are evidence judgments, "
                "not independently verified clinical diagnoses."
            ),
            "Clinical conditions and documentation styles vary.",
            "Case selection does not estimate real-world prevalence.",
        ],
        "cases": CASES,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("F2 CASE MANIFEST FROZEN")
    print("Unique cases:", len(CASES))
    print("Evidence statuses:", dict(counts))
    print("Prompt conditions:", manifest["conditions"])
    print("Planned API calls:", len(CASES) * 2)
    print("Saved:", OUTPUT_PATH)
    print("No OpenAI API calls made.")


if __name__ == "__main__":
    main()
