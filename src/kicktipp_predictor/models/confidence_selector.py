from __future__ import annotations


def extract_display_confidence(pred: dict) -> float:
    """Return a user-facing confidence value from a prediction dict.

    Falls back gracefully if fields are missing.
    """
    # Prefer explicit 'confidence' if present
    if "confidence" in pred and pred["confidence"] is not None:
        try:
            return float(pred["confidence"])
        except Exception:
            pass

    # Otherwise, compute as margin between top two outcome probabilities
    try:
        pH = float(pred.get("home_win_probability", 1 / 3))
        pD = float(pred.get("draw_probability", 1 / 3))
        pA = float(pred.get("away_win_probability", 1 / 3))
        probs = sorted([pH, pD, pA], reverse=True)
        return float(probs[0] - probs[1])
    except Exception:
        return 0.0
