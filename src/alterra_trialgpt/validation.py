VALID_LABELS = {
    "inclusion": {
        "included",
        "not included",
        "not enough information",
        "not applicable",
    },
    "exclusion": {
        "excluded",
        "not excluded",
        "not enough information",
        "not applicable",
    },
}


def is_valid_eligibility_label(criterion_type: str, label: str) -> bool:
    """
    Check whether a label is valid for a criterion type.

    Args:
        criterion_type (str): The type of criterion ('inclusion' or 'exclusion').
        label (str): The eligibility label to validate.

    Returns:
        bool: True if the label is valid for the criterion type, False otherwise.
    """

    if criterion_type not in VALID_LABELS:
        raise ValueError(
            f"Invalid criterion type: {criterion_type}. Must be 'inclusion' or 'exclusion'."
        )

    return label in VALID_LABELS.get(criterion_type, set())
