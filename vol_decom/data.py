"""Price and earnings-calendar loading, with on-disk caching and schema validation.

Design notes
------------
*Caching.*  Every fetch is keyed by ``(ticker, start, end, interval)`` and
written to ``./cache`` as Parquet plus a small JSON sidecar recording when it
was fetched and what range it covers.  A cached payload is reused when it
still covers the requested range and is younger than ``ttl_hours``.  Because
historical daily bars are immutable once the session has settled, the default
TTL is generous (24h) and a request for a range that ends in the past is
served from cache indefinitely unless ``force_refresh`` is set.

*Validation.*  Nothing leaves this module without passing
:func:`validate_ohlcv`.  The package's contract is that a malformed frame
raises :class:`~vol_decom.exceptions.SchemaValidationError` rather than
propagating quietly into a variance estimate.

*Market holidays.*  This module does **not** depend on an exchange-calendar
package.  The set of valid sessions is taken to be exactly the set of dates
present in the provider's response, which is the definition that matters for
trading-day arithmetic downstream.  Holidays therefore need no special
handling, they are simply absent from the index, and every window in
:mod:`vol_decom.engine` is counted in index positions, never in calendar
days.  What *does* need handling is an anomalous gap. A run of missing
weekdays too long to be a holiday closure.  :func:`detect_gaps` surfaces
those, and :func:`load_prices` can be configured to raise on them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import (
    CacheError,
    DataFetchError,
    InsufficientDataError,
    MissingEarningsDatesError,
    SchemaValidationError,
)

__all__ = [
    "OHLCV_COLUMNS",
    "DEFAULT_CACHE_DIR",
    "GapReport",
    "validate_ohlcv",
    "detect_gaps",
    "load_prices",
    "load_earnings_dates",
    "load_earnings_dates_from_csv",
    "clear_cache",
]

logger = logging.getLogger(__name__)

#: Columns every price frame must carry after loading.
OHLCV_COLUMNS: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

#: Default on-disk cache location, relative to the process working directory.
DEFAULT_CACHE_DIR = Path("./cache")

#: A run of missing weekdays longer than this is treated as a suspicious gap
#: rather than an exchange holiday. US markets never close for more than 4
#: consecutive weekdays outside of extraordinary events (9/11, Sandy).
MAX_HOLIDAY_RUN_DAYS: int = 4


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GapReport:
    """Description of the session-continuity of a price index.

    Attributes:
        gaps: ``(start, end, n_missing_weekdays)`` for each run of absent
            weekdays longer than :data:`MAX_HOLIDAY_RUN_DAYS`.
        n_sessions: Number of sessions actually present.
        first_session: First date in the index.
        last_session: Last date in the index.
    """
    gaps: Tuple[Tuple[pd.Timestamp, pd.Timestamp, int], ...]
    n_sessions: int
    first_session: Optional[pd.Timestamp]
    last_session: Optional[pd.Timestamp]

    @property
    def has_gaps(self) -> bool:
        """True if any anomalous gap was found."""
        return bool(self.gaps)

    def describe(self) -> str:
        """Human-readable one-line summary."""
        if not self.gaps:
            return f"{self.n_sessions} sessions, no anomalous gaps."
        parts = [f"{a.date()}->{b.date()} ({n} weekdays)" for a, b, n in self.gaps]
        return f"{self.n_sessions} sessions, {len(self.gaps)} gap(s), " + ", ".join(parts)


def validate_ohlcv(df: pd.DataFrame,
                   ticker: str = "<unknown>",
                   required_columns: Sequence[str] = OHLCV_COLUMNS,
                   allow_zero_volume: bool = True,
                   min_rows: int = 2) -> pd.DataFrame:
    """Validate and normalize an OHLCV frame.

    Checks performed, in order.

    1. Non-empty and at least ``min_rows`` rows.
    2. All ``required_columns`` present.
    3. Index is a ``DatetimeIndex``, timezone-naive after normalization,
       strictly increasing and free of duplicates.
    4. Price columns are numeric and strictly positive (a zero or negative
       price is unrecoverable. It breaks every log return).
    5. ``High >= Low``, and ``Open``/``Close`` lie within ``[Low, High]``.
    6. ``Volume`` is non-negative, and non-zero unless ``allow_zero_volume``.
    7. No all-NaN price column.

    Rows with a NaN in any price column are dropped (providers occasionally
    emit a placeholder row for a half-session), and the drop is logged.

    Args:
        df: Raw frame from the provider or cache.
        ticker: Used only in error messages.
        required_columns: Columns that must be present.
        allow_zero_volume: Permit zero-volume sessions.
        min_rows: Minimum surviving row count.

    Returns:
        A validated copy. Sorted, tz-naive index normalized to midnight,
        columns restricted to ``required_columns``, NaN price rows dropped.

    Raises:
        SchemaValidationError: On any structural or value violation.
        InsufficientDataError: If fewer than ``min_rows`` rows survive.
    """
    violations: List[str] = []

    if df is None or len(df) == 0:
        raise SchemaValidationError(f"{ticker}, provider returned an empty frame.")

    out = df.copy()

    # --- flatten a MultiIndex column header (yfinance emits one for some calls)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            c[0] if isinstance(c, tuple) and c[0] in required_columns else c[-1]
            for c in out.columns
        ]

    missing = [c for c in required_columns if c not in out.columns]
    if missing:
        raise SchemaValidationError(
            f"{ticker}, required OHLCV column(s) absent.",
            violations=[f"missing '{c}'" for c in missing],
        )
    out = out.loc[:, list(required_columns)]

    # --- index
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except (ValueError, TypeError) as exc:
            raise SchemaValidationError(
                f"{ticker}, index is not coercible to a DatetimeIndex ({exc})."
            ) from exc
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out.index.name = "Date"

    if out.index.has_duplicates:
        dupes = out.index[out.index.duplicated()].unique()
        violations.append(
            f"{len(dupes)} duplicated date(s), e.g. "
            + ", ".join(str(d.date()) for d in dupes[:3])
        )
    if not out.index.is_monotonic_increasing:
        # Sortable problems are repaired, duplicates are not.
        out = out.sort_index()
        if not out.index.is_monotonic_increasing:
            violations.append("index is not sortable into monotonic order")

    # --- numeric dtypes
    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in out.columns]
    for col in list(required_columns):
        if not pd.api.types.is_numeric_dtype(out[col]):
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.isna().all():
                violations.append(f"column '{col}' is non-numeric and uncoercible")
            out[col] = coerced

    if violations:
        raise SchemaValidationError(f"{ticker}, OHLCV schema validation failed.", violations)

    # --- drop rows with NaN prices
    nan_mask = out[price_cols].isna().any(axis=1)
    if nan_mask.any():
        logger.warning("%s. Dropping %d row(s) with NaN prices.", ticker, int(nan_mask.sum()))
        out = out.loc[~nan_mask]

    for col in price_cols:
        if out[col].isna().all():
            violations.append(f"column '{col}' is entirely NaN")

    if len(out) == 0:
        raise SchemaValidationError(f"{ticker}, no rows survived NaN filtering.")

    # --- value constraints (vectorized)
    for col in price_cols:
        bad = out[col] <= 0
        if bad.any():
            violations.append(
                f"{int(bad.sum())} non-positive '{col}' price(s), first at "
                f"{out.index[bad][0].date()}"
            )

    if "High" in out.columns and "Low" in out.columns:
        inverted = out["High"] < out["Low"]
        if inverted.any():
            violations.append(
                f"{int(inverted.sum())} session(s) with High < Low, first at "
                f"{out.index[inverted][0].date()}"
            )
        for col in ("Open", "Close"):
            if col in out.columns:
                outside = (out[col] > out["High"]) | (out[col] < out["Low"])
                if outside.any():
                    violations.append(
                        f"{int(outside.sum())} session(s) with '{col}' outside "
                        f"[Low, High], first at {out.index[outside][0].date()}"
                    )

    if "Volume" in out.columns:
        neg_vol = out["Volume"] < 0
        if neg_vol.any():
            violations.append(f"{int(neg_vol.sum())} negative Volume value(s)")
        if not allow_zero_volume:
            zero_vol = out["Volume"] == 0
            if zero_vol.any():
                violations.append(f"{int(zero_vol.sum())} zero-Volume session(s)")

    if violations:
        raise SchemaValidationError(f"{ticker}, OHLCV schema validation failed.", violations)

    if len(out) < min_rows:
        raise InsufficientDataError(
            f"{ticker}, too few valid sessions after validation.",
            required=min_rows, available=len(out),
        )

    return out


def detect_gaps(index: pd.DatetimeIndex,
                max_holiday_run: int = MAX_HOLIDAY_RUN_DAYS) -> GapReport:
    """Find runs of missing weekdays too long to be exchange holidays.

    Weekends are excluded by construction. The expected session grid is
    ``pandas`` business days (Mon-Fri) between the first and last observed
    session.  Any maximal run of expected-but-absent business days longer
    than ``max_holiday_run`` is reported.  Shorter runs are assumed to be
    holidays and are silent.

    Fully vectorized. Builds the business-day grid once and diffs it against
    the observed index.

    Args:
        index: The observed session index (tz-naive, sorted).
        max_holiday_run: Longest absent run treated as a normal closure.

    Returns:
        A :class:`GapReport`.

    Raises:
        SchemaValidationError: If ``index`` is empty or not a DatetimeIndex.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise SchemaValidationError("detect_gaps requires a DatetimeIndex.")
    if len(index) == 0:
        raise SchemaValidationError("detect_gaps requires a non-empty index.")

    index = index.sort_values()
    expected = pd.bdate_range(index[0], index[-1])
    missing = expected.difference(index)

    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if len(missing) > 0:
        # Group consecutive business days. Consecutive entries of `missing`
        # that are also adjacent in the business-day grid form one run.
        pos = expected.get_indexer(missing)
        breaks = np.where(np.diff(pos) != 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(pos) - 1]))
        for s, e in zip(starts, ends):
            run_len = int(e - s + 1)
            if run_len > max_holiday_run:
                gaps.append((missing[s], missing[e], run_len))

    return GapReport(
        gaps=tuple(gaps),
        n_sessions=len(index),
        first_session=index[0],
        last_session=index[-1],
    )


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

def _cache_paths(cache_dir: Path, ticker: str, kind: str,
                 suffix: str = "parquet") -> Tuple[Path, Path]:
    """Return ``(data_path, meta_path)`` for a cache entry."""
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in ticker.upper())
    base = cache_dir / f"{safe}__{kind}"
    return base.with_suffix(f".{suffix}"), base.with_suffix(".meta.json")


def _read_meta(meta_path: Path) -> Optional[dict]:
    """Load a cache sidecar, returning None if absent or unparseable."""
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable cache sidecar %s (%s).", meta_path, exc)
        return None


def _write_meta(meta_path: Path, payload: dict) -> None:
    """Write a cache sidecar, raising :class:`CacheError` on failure."""
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except OSError as exc:
        raise CacheError(f"Could not write cache sidecar {meta_path}, {exc}") from exc


def _cache_is_fresh(meta: Optional[dict],
                    start: pd.Timestamp,
                    end: pd.Timestamp,
                    ttl_hours: float,
                    auto_adjust: bool) -> bool:
    """Decide whether a cache entry can serve a ``[start, end]`` request.

    The entry is usable when it covers the requested range and either the
    requested range ends in the settled past, or the entry is younger than
    ``ttl_hours``.
    """
    if not meta:
        return False
    if meta.get("auto_adjust") is not auto_adjust:
        return False
    try:
        cached_start = pd.Timestamp(meta["start"])
        cached_end = pd.Timestamp(meta["end"])
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Malformed cache sidecar (%s), refetching.", exc)
        return False

    if cached_start > start or cached_end < end:
        return False

    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - fetched_at

    # Daily bars for a window that has already closed never change.
    today = pd.Timestamp.now().normalize()
    if cached_end < today - pd.Timedelta(days=1):
        return True
    return age < timedelta(hours=ttl_hours)


def clear_cache(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
                ticker: Optional[str] = None) -> int:
    """Delete cache entries.

    Args:
        cache_dir: Cache directory.
        ticker: If given, delete only this ticker's entries, otherwise delete
            every entry in the directory.

    Returns:
        Number of files removed.

    Raises:
        CacheError: If a file could not be removed.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    pattern = f"{ticker.upper()}__*" if ticker else "*"
    removed = 0
    for path in cache_dir.glob(pattern):
        if path.is_file():
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                raise CacheError(f"Could not remove {path}, {exc}") from exc
    return removed


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _default_range(years: float) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end)`` spanning the trailing ``years``."""
    end = pd.Timestamp.now().normalize()
    start = (end - pd.Timedelta(days=int(round(years * 365.25)))).normalize()
    return start, end


def load_prices(ticker: str,
                start: Optional[Union[str, pd.Timestamp]] = None,
                end: Optional[Union[str, pd.Timestamp]] = None,
                years: float = 7.0,
                cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
                ttl_hours: float = 24.0,
                force_refresh: bool = False,
                raise_on_gaps: bool = False,
                auto_adjust: bool = True,
                min_rows: int = 60) -> pd.DataFrame:
    """Load validated daily OHLCV bars, from cache when possible.

    Args:
        ticker: Equity symbol, e.g. ``"INTC"``.
        start: Inclusive start date. Defaults to ``years`` before ``end``.
        end: Inclusive end date. Defaults to today.
        years: Lookback used when ``start`` is omitted. The default of 7 years
            matches the event history used in the source research.
        cache_dir: Where to read/write Parquet cache entries.
        ttl_hours: Reuse a cache entry younger than this. Ignored for ranges
            that ended more than a day ago, which are immutable.
        force_refresh: Bypass the cache and refetch.
        raise_on_gaps: Raise instead of warning when :func:`detect_gaps`
            finds an anomalous gap.
        auto_adjust: Ask the provider for split/dividend-adjusted prices.
            Adjusted prices are correct for return-based volatility work,
            unadjusted prices inject spurious jumps on ex-dividend dates.
        min_rows: Minimum sessions required after validation.

    Returns:
        Validated OHLCV frame indexed by tz-naive session dates.

    Raises:
        DataFetchError: If the provider is unavailable or returns nothing, and
            no usable cache entry exists.
        SchemaValidationError: If the returned data violates the OHLCV contract.
        InsufficientDataError: If fewer than ``min_rows`` sessions survive.
        CacheError: If the cache directory is unusable.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise DataFetchError("Ticker must be a non-empty string.")

    default_start, default_end = _default_range(years)
    start_ts = pd.Timestamp(start).normalize() if start is not None else default_start
    end_ts = pd.Timestamp(end).normalize() if end is not None else default_end
    if start_ts >= end_ts:
        raise DataFetchError(
            f"{ticker}, start ({start_ts.date()}) must precede end ({end_ts.date()})."
        )

    cache_dir = Path(cache_dir)
    data_path, meta_path = _cache_paths(cache_dir, ticker, "prices")

    raw: Optional[pd.DataFrame] = None
    if not force_refresh and data_path.exists():
        meta = _read_meta(meta_path)
        if _cache_is_fresh(meta, start_ts, end_ts, ttl_hours, auto_adjust):
            try:
                raw = pd.read_parquet(data_path)
                logger.info("%s. Served %d rows from cache %s.", ticker, len(raw), data_path)
            except (OSError, ValueError) as exc:
                logger.warning("%s. Cache read failed (%s), refetching.", ticker, exc)
                raw = None

    if raw is None:
        raw = _fetch_prices_yf(ticker, start_ts, end_ts, auto_adjust=auto_adjust)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            raw.to_parquet(data_path)
            _write_meta(meta_path, {
                "ticker": ticker,
                "kind": "prices",
                "start": str(start_ts.date()),
                "end": str(end_ts.date()),
                "auto_adjust": auto_adjust,
                "rows": int(len(raw)),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        except (OSError, ValueError) as exc:
            # A failed cache write must not fail the request.
            logger.warning("%s. Could not cache to %s (%s).", ticker, data_path, exc)

    df = validate_ohlcv(raw, ticker=ticker, min_rows=min_rows)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"{ticker}, too few sessions in the requested window "
            f"[{start_ts.date()}, {end_ts.date()}].",
            required=min_rows, available=len(df),
        )

    report = detect_gaps(df.index)
    if report.has_gaps:
        msg = f"{ticker}, price history has anomalous gaps, {report.describe()}"
        if raise_on_gaps:
            raise SchemaValidationError(msg, violations=[
                f"{a.date()}..{b.date()} ({n} weekdays)" for a, b, n in report.gaps
            ])
        logger.warning(msg)

    df.attrs["ticker"] = ticker
    df.attrs["gap_report"] = report
    return df


def _fetch_prices_yf(ticker: str,
                     start: pd.Timestamp,
                     end: pd.Timestamp,
                     auto_adjust: bool = True) -> pd.DataFrame:
    """Fetch daily bars from yfinance.

    ``yfinance`` is imported lazily so that the rest of the package, and the
    entire test suite, can be used without it installed and without any
    network access.

    Args:
        ticker: Equity symbol.
        start: Inclusive start date.
        end: Inclusive end date.
        auto_adjust: Request adjusted prices.

    Returns:
        Raw provider frame.

    Raises:
        DataFetchError: If yfinance is missing, errors, or returns no rows.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma. No cover - environment dependent
        raise DataFetchError(
            "yfinance is required for live data. Install it with "
            "`pip install yfinance`, or supply a cached/CSV price frame."
        ) from exc

    logger.info("%s. Fetching %s..%s from yfinance.", ticker, start.date(), end.date())
    try:
        # `end` is exclusive in the yfinance API, add a day to make it inclusive.
        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=auto_adjust,
        )
    except (ValueError, KeyError, TypeError, ConnectionError, TimeoutError, OSError) as exc:
        raise DataFetchError(f"{ticker}, yfinance price request failed, {exc}") from exc

    if df is None or len(df) == 0:
        raise DataFetchError(
            f"{ticker}, yfinance returned no price rows for "
            f"{start.date()}..{end.date()}. Check the symbol is valid and listed."
        )
    return df


def load_earnings_dates(ticker: str,
                        limit: int = 40,
                        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
                        ttl_hours: float = 24.0,
                        force_refresh: bool = False,
                        include_future: bool = False,
                        asof: Optional[Union[str, pd.Timestamp]] = None
                        ) -> pd.DatetimeIndex:
    """Load historical earnings announcement dates.

    The provider returns timestamped announcements, this function normalizes
    them to session dates, de-duplicates, sorts ascending, and by default
    drops anything at or after ``asof`` so that a backtest cannot see a
    scheduled future event.

    A note on announcement timing, which the caller must handle. Most US
    large caps report *after* the close, so the price reaction lands on the
    **following** session. This function returns the announcement date as
    reported, the forward-shift onto the reacting session is applied
    explicitly in :func:`vol_decom.engine.align_events`, where it is a
    documented, configurable parameter rather than a hidden convention.

    Args:
        ticker: Equity symbol.
        limit: Maximum number of announcements to request from the provider.
            yfinance defaults to 12, which is far fewer than a 7-year study
            needs, so this is raised deliberately.
        cache_dir: Where to read/write cache entries.
        ttl_hours: Cache freshness window.
        force_refresh: Bypass the cache.
        include_future: Keep announcements dated at or after ``asof``.
        asof: Cut-off for "future". Defaults to today.

    Returns:
        Ascending, de-duplicated, tz-naive ``DatetimeIndex`` of announcement
        dates.

    Raises:
        MissingEarningsDatesError: If the provider exposes no earnings
            calendar for the symbol, or none survive filtering.
        DataFetchError: If the provider call itself fails.
    """
    ticker = ticker.upper().strip()
    cache_dir = Path(cache_dir)
    data_path, meta_path = _cache_paths(cache_dir, ticker, "earnings")

    raw: Optional[pd.DataFrame] = None
    if not force_refresh and data_path.exists():
        meta = _read_meta(meta_path)
        # Earnings calendars do change (dates get confirmed/moved), so unlike
        # settled price bars they always honour the TTL.
        if meta:
            try:
                fetched_at = datetime.fromisoformat(meta["fetched_at"])
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                fresh = (datetime.now(timezone.utc) - fetched_at) < timedelta(hours=ttl_hours)
                enough = int(meta.get("limit", 0)) >= limit
                if fresh and enough:
                    raw = pd.read_parquet(data_path)
            except (KeyError, ValueError, TypeError, OSError) as exc:
                logger.warning("%s. Earnings cache unusable (%s), refetching.", ticker, exc)
                raw = None

    if raw is None:
        raw = _fetch_earnings_yf(ticker, limit=limit)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            raw.to_parquet(data_path)
            _write_meta(meta_path, {
                "ticker": ticker,
                "kind": "earnings",
                "limit": limit,
                "rows": int(len(raw)),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        except (OSError, ValueError) as exc:
            logger.warning("%s. Could not cache earnings dates (%s).", ticker, exc)

    idx = pd.DatetimeIndex(pd.to_datetime(raw.index, utc=True, errors="coerce"))
    idx = idx[~idx.isna()]
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = pd.DatetimeIndex(idx).normalize().unique().sort_values()

    if not include_future:
        cutoff = pd.Timestamp(asof).normalize() if asof is not None \
            else pd.Timestamp.now().normalize()
        idx = idx[idx < cutoff]

    if len(idx) == 0:
        raise MissingEarningsDatesError(
            f"{ticker}, no usable historical earnings dates. The provider may not "
            f"cover this symbol (common for ETFs, ADRs and recent listings). "
            f"Supply them explicitly with --earnings-csv."
        )
    return pd.DatetimeIndex(idx)


def _fetch_earnings_yf(ticker: str, limit: int = 40) -> pd.DataFrame:
    """Fetch the earnings calendar from yfinance.

    Args:
        ticker: Equity symbol.
        limit: Number of announcements to request.

    Returns:
        Provider frame indexed by announcement timestamp.

    Raises:
        DataFetchError: If yfinance is missing or the call fails.
        MissingEarningsDatesError: If the symbol has no earnings calendar.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma. No cover - environment dependent
        raise DataFetchError(
            "yfinance is required to fetch earnings dates. Install it, or pass "
            "--earnings-csv with your own event list."
        ) from exc

    tkr = yf.Ticker(ticker)
    try:
        df = tkr.get_earnings_dates(limit=limit)
    except ImportError as exc:
        # yfinance's earnings path parses an HTML table via pandas.read_html,
        # which needs lxml. yfinance does not declare that dependency, so this
        # surfaces as a bare ImportError deep in pandas, translate it into
        # something the caller can act on.
        raise DataFetchError(
            f"{ticker}, fetching earnings dates needs an HTML parser that is not "
            f"installed ({exc}). Run `pip install lxml` (it is listed in "
            f"requirements.txt), or supply the dates yourself with --earnings-csv."
        ) from exc
    except (AttributeError, TypeError):
        # Older yfinance exposes only the `earnings_dates` property.
        try:
            df = tkr.earnings_dates
        except (AttributeError, KeyError, ValueError, TypeError) as exc:
            raise MissingEarningsDatesError(
                f"{ticker}, yfinance exposes no earnings calendar ({exc})."
            ) from exc
    except (ValueError, KeyError, ConnectionError, TimeoutError, OSError) as exc:
        raise DataFetchError(f"{ticker}, earnings-date request failed, {exc}") from exc

    if df is None or len(df) == 0:
        raise MissingEarningsDatesError(
            f"{ticker}, yfinance returned an empty earnings calendar."
        )
    return df


def load_earnings_dates_from_csv(path: Union[str, Path],
                                 column: Optional[str] = None
                                 ) -> pd.DatetimeIndex:
    """Load announcement dates from a local CSV.

    Useful for two cases the provider handles badly. Symbols with no earnings
    calendar, and studies needing more history than the provider retains
    (yfinance keeps roughly 3 years, while the source research used 7).

    The file may be a single column of dates with or without a header, or a
    wider table from which ``column`` is selected.

    Args:
        path: CSV path.
        column: Column holding the dates. If omitted, a column named
            ``date``/``earnings_date`` (case-insensitive) is used when present,
            otherwise the first column.

    Returns:
        Ascending, de-duplicated, tz-naive ``DatetimeIndex``.

    Raises:
        MissingEarningsDatesError: If the file is empty, the column is absent,
            or no value parses as a date.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise MissingEarningsDatesError(f"Could not read earnings CSV {path}, {exc}") from exc

    if df.empty:
        raise MissingEarningsDatesError(f"Earnings CSV {path} contains no rows.")

    if column is None:
        lowered = {str(c).strip().lower(): c for c in df.columns}
        for candidate in ("earnings_date", "date", "announcement_date"):
            if candidate in lowered:
                column = lowered[candidate]
                break
        else:
            column = df.columns[0]
    elif column not in df.columns:
        raise MissingEarningsDatesError(
            f"Column '{column}' not in {path}. Available columns are "
            f"{list(df.columns)}"
        )

    parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
    parsed = parsed.dropna()
    if parsed.empty:
        raise MissingEarningsDatesError(
            f"No parseable dates in column '{column}' of {path}."
        )
    idx = pd.DatetimeIndex(parsed).tz_localize(None).normalize().unique().sort_values()
    return pd.DatetimeIndex(idx)
