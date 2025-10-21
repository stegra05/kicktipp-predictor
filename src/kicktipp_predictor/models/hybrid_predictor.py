"""Legacy compatibility shim for HybridPredictor.

Maps to v2 MatchPredictor to satisfy tests expecting
`kicktipp_predictor.models.hybrid_predictor.HybridPredictor`.
"""

from ..predictor import MatchPredictor as _MatchPredictor


class HybridPredictor(_MatchPredictor):
    """Backwards-compatible alias for the v2 predictor implementation."""

    pass


