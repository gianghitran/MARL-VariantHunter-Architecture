"""Feedback contract from Detection Agent back to Generation Agent."""

import json


def load_detection_feedback(path: str) -> dict:
    """Load Detection Agent feedback JSON."""
    with open(path, encoding="utf-8") as f:
        feedback = json.load(f)
    return validate_detection_feedback(feedback)


def validate_detection_feedback(feedback: dict) -> dict:
    """Validate and normalize feedback produced by a Detection Agent."""
    required = {"graph_id", "predicted_label", "malicious_score"}
    missing = required - set(feedback)
    if missing:
        raise ValueError(f"Detection feedback missing fields: {sorted(missing)}")

    normalized = dict(feedback)
    normalized["malicious_score"] = float(normalized["malicious_score"])
    normalized["confidence"] = float(normalized.get("confidence", normalized["malicious_score"]))
    normalized["is_false_negative"] = bool(normalized.get("is_false_negative", False))
    normalized["predicted_label"] = str(normalized["predicted_label"])
    return normalized


def compute_closed_loop_reward(feedback: dict, generation_reward: dict,
                               fn_bonus: float = 0.35,
                               evade_bonus: float = 0.2) -> dict:
    """
    Convert detector feedback into a Generation Agent reward component.

    A generated malicious sample is useful when it is valid and hard for the
    detector. False negatives receive a stronger bonus because they are the
    samples most valuable for adversarial augmentation.
    """
    feedback = validate_detection_feedback(feedback)
    gen_total = float(generation_reward.get("total", 0.0))
    malicious_score = feedback["malicious_score"]
    detector_evasion = 1.0 - malicious_score
    hard_sample_bonus = fn_bonus if feedback["is_false_negative"] else 0.0
    evasion_component = evade_bonus * detector_evasion
    closed_loop_total = min(1.0, gen_total + evasion_component + hard_sample_bonus)

    return {
        "closed_loop_total": closed_loop_total,
        "generation_total": gen_total,
        "detector_malicious_score": malicious_score,
        "detector_evasion": detector_evasion,
        "hard_sample_bonus": hard_sample_bonus,
        "predicted_label": feedback["predicted_label"],
        "is_false_negative": feedback["is_false_negative"],
    }
