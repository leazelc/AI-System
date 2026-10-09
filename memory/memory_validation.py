import math


VALID_TYPES = {
    "core",
    "preferences",
    "projects",
    "experiences",
    "decisions",
}

VALID_ACTIONS = {
    "NEW",
    "UPDATE",
    "MERGE",
    "IGNORE",
}


def validate_decision(decision, valid_ids=None):
    """
    Validate a memory updater decision.

    Returns:
        (True, None) when valid.
        (False, reason) when invalid.
    """
    if not isinstance(decision, dict):
        return False, "Decision must be a dictionary."

    action = decision.get("action")

    if not isinstance(action, str) or action not in VALID_ACTIONS:
        return False, "Invalid action."

    if action == "IGNORE":
        if decision.get("memory") is not None:
            return False, "IGNORE must not contain memory content."
        return True, None

    memory = decision.get("memory")

    if not isinstance(memory, str) or not memory.strip():
        return False, "Memory content must be a non-empty string."

    if decision.get("type") not in VALID_TYPES:
        return False, "Invalid memory type."

    for field in ("importance", "confidence"):
        value = decision.get(field)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"{field} must be a number."

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False, f"{field} must be between 0.0 and 1.0."

    target_id = decision.get("target_id")

    if action == "NEW":
        if target_id is not None:
            return False, "NEW must not have a target_id."
        return True, None

    if not isinstance(target_id, str) or not target_id.strip():
        return False, f"{action} requires a target_id."

    if valid_ids is not None and target_id not in valid_ids:
        return False, "Target memory ID does not exist in valid candidates."

    return True, None
