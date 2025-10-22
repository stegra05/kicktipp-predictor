from __future__ import annotations


def extract_display_confidence(pred: dict) -> float:
    """Extracts a user-facing confidence value from a prediction dictionary.

    This function first attempts to use a pre-calculated 'confidence' value from the
    prediction. If unavailable, it computes the confidence as the difference
    between the two highest outcome probabilities (home win, draw, away win),
    ensuring graceful fallback in case of missing data.
    """
    # Use pre-calculated confidence if available and valid.
    if "confidence" in pred and pred["confidence"] is not None:
        try:
            return float(pred["confidence"])
        except (ValueError, TypeError):
            pass

    # Fallback: Compute confidence from outcome probabilities.
    # The confidence is defined as the margin between the most likely and
    # second-most likely outcomes.
    try:
        home_win_prob = float(pred.get("home_win_probability", 1 / 3))
        draw_prob = float(pred.get("draw_probability", 1 / 3))
        away_win_prob = float(pred.get("away_win_probability", 1 / 3))

        probabilities = sorted([home_win_prob, draw_prob, away_win_prob], reverse=True)
        return float(probabilities[0] - probabilities[1])
    except (ValueError, TypeError):
        # Return a neutral confidence if data is corrupt or missing.
        return 0.0
