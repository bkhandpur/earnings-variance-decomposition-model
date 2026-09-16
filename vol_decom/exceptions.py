"""Typed exception hierarchy for :mod:`vol_decom`.

Every failure mode in the package raises one of these rather than returning
partially-valid data.  Callers can therefore distinguish "the network was
down" from "the data came back malformed" from "there is not enough history
to estimate anything", which matters when the package is driven from a batch
job over hundreds of tickers.
"""
from __future__ import annotations

from typing import Optional, Sequence

__all__ = [
    "VolDecomError",
    "DataError",
    "DataFetchError",
    "SchemaValidationError",
    "InsufficientDataError",
    "MissingEarningsDatesError",
    "CacheError",
    "AlignmentError",
    "BacktestError",
]


class VolDecomError(Exception):
    """Base class for every error raised by :mod:`vol_decom`."""


class DataError(VolDecomError):
    """Base class for data acquisition and integrity failures."""


class DataFetchError(DataError):
    """Raised when the upstream provider could not be reached or returned nothing.

    This is an *availability* failure, not a correctness failure. Retrying
    later may succeed.
    """


class SchemaValidationError(DataError):
    """Raised when fetched data violates the expected OHLCV contract.

    Triggers include. Missing columns, non-numeric prices, negative or zero
    prices, ``High < Low``, a close outside the day's range, a non-monotonic
    or duplicated date index, or an all-NaN column.
    """

    def __init__(self, message: str, violations: Optional[Sequence[str]] = None) -> None:
        self.violations: list[str] = list(violations or [])
        if self.violations:
            message = f"{message} Violations, " + ", ".join(self.violations)
        super().__init__(message)


class InsufficientDataError(DataError):
    """Raised when there is too little history to compute the requested estimate.

    Carries the shortfall so callers can widen the request programmatically.
    """

    def __init__(self, message: str, required: Optional[int] = None,
                 available: Optional[int] = None) -> None:
        self.required = required
        self.available = available
        if required is not None and available is not None:
            message = f"{message} (required={required}, available={available})"
        super().__init__(message)


class MissingEarningsDatesError(DataError):
    """Raised when no usable historical earnings announcement dates were found.

    Distinct from :class:`InsufficientDataError`: the price history may be
    perfectly fine while the event calendar is empty, which is common for
    recently-listed tickers, ADRs and index ETFs.
    """


class CacheError(DataError):
    """Raised when the on-disk cache is unreadable, unwritable or corrupt."""


class AlignmentError(VolDecomError):
    """Raised when an event cannot be aligned onto the trading-day grid.

    All calendar arithmetic in this package is explicit, if an announcement
    date falls outside the span of available sessions we refuse rather than
    silently snapping to an endpoint.
    """


class BacktestError(VolDecomError):
    """Raised on an ill-specified or unrunnable backtest configuration."""
