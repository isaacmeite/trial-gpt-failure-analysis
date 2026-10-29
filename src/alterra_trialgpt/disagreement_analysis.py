from collections import Counter
from typing import Any


def count_disagreements(
    annotations: list[dict[str, Any]],
) -> int:
    """Count GPT-4 predictions that differ from expert labels."""

    disagreement_count = 0

    for annotation in annotations:
        expert_label = annotation.get("expert_eligibility")
        gpt4_label = annotation.get("gpt4_eligibility")

        if expert_label is None or gpt4_label is None:
            raise ValueError(
                "Cannot count disagreements for annotations with missing labels."
            )

        if expert_label != gpt4_label:
            disagreement_count += 1

    return disagreement_count


def count_label_transitions(
    annotations: list[dict[str, Any]],
) -> Counter[tuple[str, str, str]]:
    """Count criterion-type, expert-label, and GPT-4-label combinations"""

    transitions: Counter[tuple[str, str, str]] = Counter()

    for annotation in annotations:
        if annotation.get("criterion_type") is None:
            raise ValueError("Missing criterion_type in annotation.")
        if annotation.get("expert_eligibility") is None:
            raise ValueError("Missing expert_eligibility in annotation.")
        if annotation.get("gpt4_eligibility") is None:
            raise ValueError("Missing gpt4_eligibility in annotation.")

        transition = (
            annotation["criterion_type"],
            annotation["expert_eligibility"],
            annotation["gpt4_eligibility"],
        )
        transitions[transition] += 1

    return transitions
