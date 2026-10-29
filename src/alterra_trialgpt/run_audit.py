from datasets import load_dataset

from alterra_trialgpt.dataset_audit import (
    audit_annotations,
    find_duplicate_annotation_ids,
)
from alterra_trialgpt.disagreement_analysis import (
    count_disagreements,
    count_label_transitions,
)

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"


def main() -> None:
    dataset = load_dataset(DATASET_ID)

    annotations = list(dataset["train"])

    audit_results = audit_annotations(annotations)
    if audit_results:
        raise ValueError(f"Cannot evaluate {len(audit_results)} invalid annotations.")

    disagreement_count = count_disagreements(annotations)
    disagreement_rate = disagreement_count / len(annotations) if annotations else 0

    transition_counts = count_label_transitions(annotations)
    assert sum(transition_counts.values()) == len(annotations)

    disagreements_from_transitions = 0

    for transition, count in transition_counts.items():
        criterion_type, expert_label, gpt4_label = transition
        if expert_label != gpt4_label:
            disagreements_from_transitions += count

    assert disagreements_from_transitions == disagreement_count

    duplicate_ids = find_duplicate_annotation_ids(annotations)

    print(f"Total annotations: {len(annotations)}")
    print(f"Invalid annotations: {len(audit_results)}")
    print(f"GPT-4 Disagreements: {disagreement_count}")
    print(f"Disagreement Rate: {disagreement_rate:.2%}")
    print(f"Disagreements from label transitions: {disagreements_from_transitions}")
    print(f"Unique duplicated ID values: {len(duplicate_ids)}")
    print(f"Duplicated IDs: {duplicate_ids}")

    print("\n Historical GPT-4 Disagreement transitions: ")

    for transition, count in transition_counts.most_common():
        criterion_type, expert_label, gpt4_label = transition

        if expert_label != gpt4_label:
            print(
                f"Criterion Type: {criterion_type}, Expert Label: {expert_label}, GPT-4 Label: {gpt4_label}, Count: {count}"
            )

    # Conditional probability of GPT-4 predicting "not applicable" given that the expert label is "not excluded"
    # for exclusion criteria
    not_excluded_total = 0

    for (criterion_type, expert_label, gpt_4_label), count in transition_counts.items():
        if criterion_type == "exclusion" and expert_label == "not excluded":
            not_excluded_total += count

    applicability_confusions = transition_counts[
        ("exclusion", "not excluded", "not applicable")
    ]

    applicability_confusion_rate = (applicability_confusions / not_excluded_total) * 100

    assert applicability_confusions <= not_excluded_total
    print(f"Applicability Confusion Rate: {applicability_confusion_rate:.2f}%")

    applicability_examples = []

    for annotation in annotations:
        if (
            annotation["criterion_type"] == "exclusion"
            and annotation["expert_eligibility"] == "not excluded"
            and annotation["gpt4_eligibility"] == "not applicable"
        ):
            applicability_examples.append(annotation)

    assert len(applicability_examples) == applicability_confusions

    for annotation in applicability_examples[:5]:
        print(f"\nAnnotation ID: {annotation['annotation_id']}")
        print(f"Patient ID: {annotation['patient_id']}")
        print(f"Criterion: {annotation['criterion_text']}")
        print(f"GPT-4 explanation: {annotation['gpt4_explanation']}")

    review_ids = {44, 108, 109}

    for annotation in annotations:
        if annotation["annotation_id"] in review_ids:
            print(f"\nAnnotation ID: {annotation['annotation_id']}")
            print(f"Criterion: {annotation['criterion_text']}")
            print(f"Patient note: {annotation['note']}")
            print(f"Expert eligibility: {annotation['expert_eligibility']}")
            print(f"GPT-4 eligibility: {annotation['gpt4_eligibility']}")
            print(f"GPT-4 explanation: {annotation['gpt4_explanation']}")


if __name__ == "__main__":
    main()
