"""Legacy compatibility shim for DataFetcher.

Maps to the v2 DataLoader API to satisfy tests expecting
`kicktipp_predictor.core.scraper.data_fetcher.DataFetcher`.
"""

from ...data import DataLoader as _DataLoader


class DataFetcher(_DataLoader):
    """Backwards-compatible alias for v1 DataFetcher.

    Inherits v2 DataLoader implementation.
    """

    pass


