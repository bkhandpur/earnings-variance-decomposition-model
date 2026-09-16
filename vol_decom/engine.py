"""Earnings variance decomposition. Isolating jump variance from diffusive vol.

The model
--------
A stock's return process around an earnings announcement is treated as the
superposition of two components.

* a **continuous diffusive** component that is always running, whose scale is
  estimated from a quiet pre-event window, and
* a **discrete jump** that fires once, on the session in which the market
  reprices the announcement.

Because variance is additive across independent components, the observed
variance over a window containing the event is the sum of the two, and the
jump can be recovered by subtraction::

    sigma_jump^2 = sigma_total^2 - sigma_baseline^2
    sigma_jump   = sqrt( max(0, sigma_total^2 - sigma_baseline^2) )

The ``max(0, .)`` floor is not cosmetic. Both terms are sample estimates, so
sampling error alone will drive the difference negative for genuinely quiet
events. A negative variance is meaningless, so it is clamped, but the
clamping is *counted* and reported, because a high clamp rate is a signal
that the estimator is too noisy to trust rather than a signal that the stock
does not jump.

Trading-day alignment
---------------------
Every window in this module is expressed in **index positions on the observed
session grid**, never in calendar days. Given the return series ``r`` (where
``r_t`` is dated on the later of the two sessions it spans, per
:func:`vol_decom.estimators.log_returns`), an announcement is mapped to

    p = first position with date >= announcement_date, plus `announcement_offset`

and the three windows are, in positions.

    baseline  ->  [p - baseline_gap - baseline_window,  p - baseline_gap)
    event     ->  [p,                                   p + event_window)
    post-k    ->  [p + 1,                               p + 1 + k)

Why the default ``event_window`` is 2
-------------------------------------
This is inherited from the source prototype and is a deliberate,
timing-agnostic hedge, worth spelling out because it looks arbitrary.
Announcements land either before the open (BMO) or after the close (AMC).
For a BMO announcement on session ``D`` the reaction is the return dated
``D``, for an AMC announcement on ``D`` the reaction is the return dated
``D+1``. A 2-session window starting at ``D`` contains the reacting session
under either convention, so the decomposition does not need to know the
announcement time of day, which providers report unreliably. The cost is
that one non-event session is always included, diluting the estimate, set
``announcement_offset=1`` with ``event_window=1`` if you know the names in
your universe all report AMC.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .estimators import (
    TRADING_DAYS_PER_YEAR,
    annualize,
    log_returns,
    rolling_parkinson,
    rolling_yang_zhang,
)
from .exceptions import (
    AlignmentError,
    InsufficientDataError,
    MissingEarningsDatesError,
    SchemaValidationError,
)

__all__ = [
    "DecompositionConfig",
    "EventAlignment",
    "align_events",
    "decompose_events",
    "summarize_decomposition",
    "implied_jump_move",
    "volatility_risk_premium",
    "VRPResult",
    "vol_cone",
]

logger = logging.getLogger(__name__)

_BASELINE_ESTIMATORS = ("close_to_close", "parkinson", "yang_zhang")


@dataclass(frozen=True)
class DecompositionConfig:
    """Parameters governing the variance decomposition.

    Attributes:
        baseline_windows: Trailing window lengths, in trading days, over which
            the diffusive baseline is estimated. The first entry is the
            *primary* baseline used for the jump subtraction, the rest are
            computed alongside it for robustness comparison. Default
            ``(20, 30, 60)``, the 20-day primary matches the source prototype.
        event_window: Sessions in the event window, starting at the aligned
            session. See the module docstring for why 2 is the default.
        post_windows: Post-event realized-vol horizons, in sessions, measured
            from the session *after* the aligned session.
        baseline_gap: Sessions to leave between the end of the baseline window
            and the event window. 0 means the baseline runs right up to the
            event. Raise it to avoid contaminating the baseline with
            pre-announcement drift or vol ramp.
        announcement_offset: Sessions to shift the aligned position forward
            from the announcement date. 0 (default, prototype behaviour)
            combined with ``event_window=2`` is timing-agnostic.
        estimator: Baseline estimator: ``close_to_close``, ``parkinson`` or
            ``yang_zhang``. The event window always uses close-to-close
            realized variance, because range estimators cannot see the
            overnight gap that *is* the earnings jump.
        zero_mean: Use the zero-drift realized-variance form
            ``mean(r^2)``. Strongly recommended, see
            :func:`vol_decom.estimators.realized_variance`.
        ddof: Degrees of freedom when ``zero_mean`` is False.
        annualized: Report volatilities annualized by ``sqrt(252)``.
        periods_per_year: Annualization factor.
        legacy_prototype_mode: Reproduce the original prototype exactly:
            demeaned population variance (``zero_mean=False, ddof=0``) on a
            2-session event window, daily (un-annualized) output. Provided for
            regression checking only, it is known to zero out roughly half of
            all events. See the README.
    """
    baseline_windows: Tuple[int, ...] = (20, 30, 60)
    event_window: int = 2
    post_windows: Tuple[int, ...] = (5, 10)
    baseline_gap: int = 0
    announcement_offset: int = 0
    estimator: str = "close_to_close"
    zero_mean: bool = True
    ddof: int = 0
    annualized: bool = True
    periods_per_year: int = TRADING_DAYS_PER_YEAR
    legacy_prototype_mode: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On any out-of-range or inconsistent parameter.
        """
        if not self.baseline_windows:
            raise ValueError("baseline_windows must contain at least one window.")
        if any(w < 2 for w in self.baseline_windows):
            raise ValueError(f"all baseline_windows must be >= 2, got {self.baseline_windows}")
        if self.event_window < 1:
            raise ValueError(f"event_window must be >= 1, got {self.event_window}")
        if any(w < 1 for w in self.post_windows):
            raise ValueError(f"all post_windows must be >= 1, got {self.post_windows}")
        if self.baseline_gap < 0:
            raise ValueError(f"baseline_gap must be >= 0, got {self.baseline_gap}")
        if self.announcement_offset < 0:
            raise ValueError(
                f"announcement_offset must be >= 0, got {self.announcement_offset}"
            )
        if self.estimator not in _BASELINE_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {_BASELINE_ESTIMATORS}, got '{self.estimator}'"
            )
        if self.periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be > 0, got {self.periods_per_year}")

    @property
    def primary_baseline(self) -> int:
        """The baseline window used for the jump subtraction."""
        return int(self.baseline_windows[0])

    @property
    def effective_zero_mean(self) -> bool:
        """``zero_mean`` after applying :attr:`legacy_prototype_mode`."""
        return False if self.legacy_prototype_mode else self.zero_mean

    @property
    def effective_ddof(self) -> int:
        """``ddof`` after applying :attr:`legacy_prototype_mode`."""
        return 0 if self.legacy_prototype_mode else self.ddof

    @property
    def effective_annualized(self) -> bool:
        """``annualized`` after applying :attr:`legacy_prototype_mode`."""
        return False if self.legacy_prototype_mode else self.annualized

    @classmethod
    def prototype(cls) -> "DecompositionConfig":
        """Return the configuration that reproduces the original prototype."""
        return cls(
            baseline_windows=(20,),
            event_window=2,
            post_windows=(5, 10),
            estimator="close_to_close",
            legacy_prototype_mode=True,
        )


@dataclass(frozen=True, eq=False)
class EventAlignment:
    """Result of mapping announcement dates onto the trading-day grid.

    ``eq=False`` is deliberate. Instances are stored in ``DataFrame.attrs``,
    and pandas compares ``attrs`` dictionaries for equality during operations
    like ``concat`` (which ``repr`` triggers internally when it truncates wide
    frames). A dataclass-generated ``__eq__`` would compare the ndarray fields
    elementwise and raise "truth value of an array is ambiguous", identity
    comparison is both safe and the right semantics here.

    Attributes:
        positions: Integer positions into the *return* series, one per usable
            event, ascending and unique.
        announcement_dates: The announcement date behind each position.
        session_dates: The session date each position corresponds to.
        dropped_future: Announcements after the last available session.
        dropped_insufficient: Announcements dropped for lack of surrounding
            history, as ``(date, reason)`` pairs.
        dropped_duplicate: Announcements that collapsed onto a session
            already claimed by an earlier announcement.
    """
    positions: np.ndarray
    announcement_dates: pd.DatetimeIndex
    session_dates: pd.DatetimeIndex
    dropped_future: pd.DatetimeIndex
    dropped_insufficient: Tuple[Tuple[pd.Timestamp, str], ...]
    dropped_duplicate: pd.DatetimeIndex

    def __len__(self) -> int:
        return int(self.positions.size)

    def describe(self) -> str:
        """Human-readable summary of what was kept and what was dropped."""
        return (
            f"{len(self)} usable event(s), dropped "
            f"{len(self.dropped_future)} future, "
            f"{len(self.dropped_insufficient)} for insufficient history, "
            f"{len(self.dropped_duplicate)} duplicate."
        )


def align_events(return_index: pd.DatetimeIndex,
                 announcement_dates: Union[pd.DatetimeIndex, Sequence],
                 config: Optional[DecompositionConfig] = None,
                 strict: bool = False) -> EventAlignment:
    """Map announcement dates onto positions in the return series.

    Alignment rule, matching the source prototype::

        p = searchsorted(return_index, announcement_date, side='left')
              + announcement_offset

    i.e. the first session whose return date is on or after the announcement,
    optionally shifted forward. If the announcement falls on a non-session
    (weekend or holiday) this naturally rolls forward to the next session --
    which is the correct behaviour and needs no calendar package, since the
    session grid *is* the observed index.

    An event is usable only when both its baseline and its forward windows fit
    entirely inside the available history::

        p - baseline_gap - max(baseline_windows) >= 0
        p + max(event_window, 1 + max(post_windows)) <= len(return_index)

    This is stricter than the prototype's hardcoded ``idx >= 20 and
    idx + 10 < len(r)`` only in that it derives the bounds from the config
    instead of assuming 20/10.

    Fully vectorized. One ``searchsorted`` plus boolean masks, no per-event loop.

    Args:
        return_index: Dates of the return series, sorted ascending.
        announcement_dates: Announcement dates. Normalized and de-duplicated
            internally.
        config: Window configuration, defaults to :class:`DecompositionConfig`.
        strict: If True, raise instead of dropping when any event is unusable.

    Returns:
        An :class:`EventAlignment`.

    Raises:
        AlignmentError: If ``return_index`` is empty or unsorted, or if
            ``strict`` is set and any event had to be dropped.
        MissingEarningsDatesError: If no announcement dates were supplied, or
            none survive alignment.
    """
    cfg = config or DecompositionConfig()

    if not isinstance(return_index, pd.DatetimeIndex):
        raise AlignmentError("return_index must be a DatetimeIndex.")
    n = len(return_index)
    if n == 0:
        raise AlignmentError("return_index is empty, nothing to align against.")
    if not return_index.is_monotonic_increasing:
        raise AlignmentError("return_index must be sorted ascending before alignment.")

    ann = pd.DatetimeIndex(pd.to_datetime(list(announcement_dates), errors="coerce"))
    ann = ann[~ann.isna()]
    if ann.tz is not None:
        ann = ann.tz_localize(None)
    ann = pd.DatetimeIndex(ann).normalize().unique().sort_values()
    if len(ann) == 0:
        raise MissingEarningsDatesError("No valid announcement dates supplied.")

    # --- vectorized forward alignment
    raw_pos = np.searchsorted(return_index.to_numpy(), ann.to_numpy(), side="left")
    pos = raw_pos + cfg.announcement_offset

    # Announcements beyond the last session (includes scheduled future events).
    future_mask = pos >= n
    dropped_future = ann[future_mask]

    pos_ok = pos[~future_mask]
    ann_ok = ann[~future_mask]

    # --- window-fit bounds
    back_need = cfg.baseline_gap + max(cfg.baseline_windows)
    fwd_need = max(cfg.event_window, 1 + max(cfg.post_windows))

    too_early = pos_ok - back_need < 0
    too_late = pos_ok + fwd_need > n

    insufficient: List[Tuple[pd.Timestamp, str]] = []
    for date, early, late in zip(ann_ok[too_early | too_late],
                                 too_early[too_early | too_late],
                                 too_late[too_early | too_late]):
        # This loop runs only over *rejected* events (typically 0-4 of them)
        # purely to build human-readable diagnostics, it is not on the
        # estimation path.
        reason = "insufficient pre-event history" if early else "insufficient post-event history"
        insufficient.append((date, reason))

    keep = ~(too_early | too_late)
    pos_keep = pos_ok[keep]
    ann_keep = ann_ok[keep]

    # --- collapse duplicates (two announcements rolling onto one session)
    _, first_idx = np.unique(pos_keep, return_index=True)
    first_idx = np.sort(first_idx)
    dup_mask = np.ones(pos_keep.size, dtype=bool)
    dup_mask[first_idx] = False
    dropped_duplicate = ann_keep[dup_mask]

    pos_final = pos_keep[first_idx]
    ann_final = ann_keep[first_idx]

    if strict and (len(dropped_future) or insufficient or len(dropped_duplicate)):
        raise AlignmentError(
            "strict alignment failed, "
            f"{len(dropped_future)} future, {len(insufficient)} insufficient-history, "
            f"{len(dropped_duplicate)} duplicate event(s)."
        )

    if pos_final.size == 0:
        raise MissingEarningsDatesError(
            "No earnings events survived trading-day alignment. Every candidate was "
            "either in the future or lacked the surrounding history required by the "
            f"configuration (needs {back_need} sessions before and {fwd_need} after "
            f"each event, history has {n} sessions)."
        )

    return EventAlignment(
        positions=pos_final.astype(int),
        announcement_dates=pd.DatetimeIndex(ann_final),
        session_dates=pd.DatetimeIndex(return_index[pos_final]),
        dropped_future=pd.DatetimeIndex(dropped_future),
        dropped_insufficient=tuple(insufficient),
        dropped_duplicate=pd.DatetimeIndex(dropped_duplicate),
    )


def _window_matrix(values: np.ndarray,
                   positions: np.ndarray,
                   start_offset: int,
                   length: int) -> np.ndarray:
    """Gather a ``(n_events, length)`` matrix of windowed values.

    Window ``i`` spans positions ``[p_i + start_offset, p_i + start_offset + length)``.
    Uses one broadcast fancy-index rather than a loop, so extracting all
    windows for all events is a single vectorized gather.

    Args:
        values: 1-D source array.
        positions: Event positions.
        start_offset: Offset of the window start relative to each position.
        length: Window length.

    Returns:
        Matrix of shape ``(positions.size, length)``.

    Raises:
        IndexError: If any window falls outside ``values``. Callers are
            expected to have filtered via :func:`align_events` first.
    """
    if length < 1:
        raise ValueError(f"window length must be >= 1, got {length}")
    idx = positions[:, None] + start_offset + np.arange(length)[None, :]
    if idx.min() < 0 or idx.max() >= values.size:
        raise IndexError(
            f"window [{start_offset}, {start_offset + length}) escapes the return "
            f"series (positions {positions.min()}..{positions.max()}, "
            f"series length {values.size}). Align events first."
        )
    return values[idx]


def _row_variance(mat: np.ndarray, zero_mean: bool, ddof: int) -> np.ndarray:
    """Per-row realized variance of a window matrix.

    ``zero_mean=True``  -> ``mean(r^2)`` along each row (realized-variance form)
    ``zero_mean=False`` -> ``var(r, ddof=ddof)`` along each row (sample form)

    Args:
        mat: ``(n_events, window)`` matrix of returns.
        zero_mean: Convention selector.
        ddof: Degrees of freedom for the sample form.

    Returns:
        Length-``n_events`` array of per-session variances.
    """
    if zero_mean:
        return np.nanmean(mat ** 2, axis=1)
    if mat.shape[1] <= ddof:
        return np.full(mat.shape[0], np.nan)
    return np.nanvar(mat, axis=1, ddof=ddof)


def decompose_events(prices: pd.DataFrame,
                     earnings_dates: Union[pd.DatetimeIndex, Sequence],
                     config: Optional[DecompositionConfig] = None
                     ) -> pd.DataFrame:
    """Decompose realized variance into diffusive and jump components per event.

    For each aligned event this computes.

    * ``sigma_baseline_{w}`` for every ``w`` in ``config.baseline_windows`` --
      the diffusive scale from the trailing quiet window.
    * ``sigma_total``, realized vol over the event window.
    * ``sigma_jump``, the isolated jump component::

          sigma_jump = sqrt( max(0, sigma_total^2 - sigma_baseline^2) )

    * ``jump_clamped``, True where the subtraction went negative and was
      floored at zero.
    * ``variance_ratio``, ``sigma_total^2 / sigma_baseline^2``, the
      scale-free measure of how much the event inflated variance. A ratio of
      1 means the event was indistinguishable from a normal session.
    * ``event_return`` / ``abs_event_return``, the signed and absolute
      single-session move on the largest-magnitude session in the event
      window. This is the *realized move* that an options position is
      actually exposed to, and is the quantity compared against the
      market-implied move in :func:`volatility_risk_premium`.
    * ``sigma_post_{k}`` for every ``k`` in ``config.post_windows``, realized
      vol in the ``k`` sessions after the event, i.e. the post-event
      vol-crush regime.

    Every column is computed with array operations over all events at once,
    there is no loop over the event history.

    Args:
        prices: Validated OHLCV frame (see :func:`vol_decom.data.load_prices`).
            Must contain ``Close``. ``Open``/``High``/``Low`` are additionally
            required when ``config.estimator`` is a range estimator.
        earnings_dates: Announcement dates.
        config: Window and estimator configuration.

    Returns:
        One row per usable event, indexed by the aligned session date, with
        ``attrs['alignment']`` carrying the :class:`EventAlignment` and
        ``attrs['config']`` the configuration used. Sorted ascending by date.

    Raises:
        SchemaValidationError: If ``prices`` lacks the columns the chosen
            estimator needs.
        InsufficientDataError: If the price history is shorter than the
            configured windows require.
        MissingEarningsDatesError: If no event survives alignment.
        AlignmentError: If the price index is unusable for alignment.
    """
    cfg = config or DecompositionConfig()

    if "Close" not in prices.columns:
        raise SchemaValidationError("decompose_events requires a 'Close' column.")
    if cfg.estimator in ("parkinson", "yang_zhang"):
        missing = [c for c in ("Open", "High", "Low", "Close") if c not in prices.columns]
        if missing:
            raise SchemaValidationError(
                f"estimator '{cfg.estimator}' requires full OHLC.",
                violations=[f"missing '{c}'" for c in missing],
            )

    rets = log_returns(prices["Close"])
    r = rets.to_numpy(dtype=float)
    ret_index = pd.DatetimeIndex(rets.index)

    need = cfg.baseline_gap + max(cfg.baseline_windows) + max(
        cfg.event_window, 1 + max(cfg.post_windows)
    )
    if r.size < need:
        raise InsufficientDataError(
            "Price history is shorter than the configured windows require.",
            required=need + 1, available=r.size + 1,
        )

    align = align_events(ret_index, earnings_dates, config=cfg)
    pos = align.positions
    zero_mean = cfg.effective_zero_mean
    ddof = cfg.effective_ddof

    out: Dict[str, np.ndarray] = {}

    # --- event-window total variance (always close-to-close. The jump is a gap)
    event_mat = _window_matrix(r, pos, 0, cfg.event_window)
    var_total = _row_variance(event_mat, zero_mean, ddof)

    # The single session that actually carries the repricing.
    jump_col = np.nanargmax(np.abs(event_mat), axis=1)
    event_return = event_mat[np.arange(event_mat.shape[0]), jump_col]

    # --- baseline variance per configured window
    if cfg.estimator == "close_to_close":
        baseline_vars: Dict[int, np.ndarray] = {}
        for w in cfg.baseline_windows:
            mat = _window_matrix(r, pos, -(cfg.baseline_gap + w), w)
            baseline_vars[w] = _row_variance(mat, zero_mean, ddof)
    else:
        # Range estimators are evaluated on the OHLC grid. The rolling series
        # is right-aligned on the price index, the return at position p is
        # dated price-index position p+1, so the last baseline session before
        # the event window is price-index position p - baseline_gap.
        rolling_fn = rolling_parkinson if cfg.estimator == "parkinson" else rolling_yang_zhang
        baseline_vars = {}
        for w in cfg.baseline_windows:
            series = rolling_fn(prices, window=w, annualized=False)
            vals = series.to_numpy(dtype=float)
            take = pos - cfg.baseline_gap
            sigma_w = vals[take]
            baseline_vars[w] = sigma_w ** 2

    primary = cfg.primary_baseline
    var_baseline = baseline_vars[primary]

    # --- the decomposition
    var_jump_raw = var_total - var_baseline
    clamped = var_jump_raw < 0
    var_jump = np.where(clamped, 0.0, var_jump_raw)

    sigma_total = np.sqrt(var_total)
    sigma_jump = np.sqrt(var_jump)

    with np.errstate(divide="ignore", invalid="ignore"):
        variance_ratio = np.where(var_baseline > 0, var_total / var_baseline, np.nan)

    scale = float(np.sqrt(cfg.periods_per_year)) if cfg.effective_annualized else 1.0

    out["sigma_total"] = sigma_total * scale
    for w in cfg.baseline_windows:
        out[f"sigma_baseline_{w}"] = np.sqrt(baseline_vars[w]) * scale
    out["sigma_baseline"] = np.sqrt(var_baseline) * scale
    out["sigma_jump"] = sigma_jump * scale
    out["var_jump_raw"] = var_jump_raw
    out["jump_clamped"] = clamped
    out["variance_ratio"] = variance_ratio

    # --- post-event realized vol
    for k in cfg.post_windows:
        mat = _window_matrix(r, pos, 1, k)
        out[f"sigma_post_{k}"] = np.sqrt(_row_variance(mat, zero_mean, ddof)) * scale

    # --- per-event descriptors
    out["event_return"] = event_return
    out["abs_event_return"] = np.abs(event_return)

    df = pd.DataFrame(out, index=align.session_dates)
    df.index.name = "session_date"
    df.insert(0, "announcement_date", align.announcement_dates)
    df.insert(1, "return_position", pos)

    # Session close on the event-window start, useful for the backtest.
    price_pos = pos + 1  # returns index is offset one from the price index
    df.insert(2, "close", prices["Close"].to_numpy(dtype=float)[price_pos])

    df.attrs["alignment"] = align
    df.attrs["config"] = cfg
    df.attrs["ticker"] = prices.attrs.get("ticker", "<unknown>")
    df.attrs["annualized"] = cfg.effective_annualized
    return df.sort_index()


def summarize_decomposition(events: pd.DataFrame) -> Dict[str, float]:
    """Aggregate a decomposition into headline statistics.

    Args:
        events: Output of :func:`decompose_events`.

    Returns:
        Dict with the event count, mean/median jump vol, mean baseline and
        total vol, mean absolute event move, mean variance ratio, and the
        clamp rate (fraction of events where the jump variance floored at
        zero, an estimator-health diagnostic).

    Raises:
        ValueError: If ``events`` is empty.
    """
    if events is None or events.empty:
        raise ValueError("Cannot summarize an empty decomposition.")

    n = len(events)
    summary = {
        "n_events": float(n),
        "mean_sigma_jump": float(events["sigma_jump"].mean()),
        "median_sigma_jump": float(events["sigma_jump"].median()),
        "mean_sigma_baseline": float(events["sigma_baseline"].mean()),
        "mean_sigma_total": float(events["sigma_total"].mean()),
        "mean_abs_event_return": float(events["abs_event_return"].mean()),
        "median_abs_event_return": float(events["abs_event_return"].median()),
        "mean_variance_ratio": float(events["variance_ratio"].mean()),
        "clamp_rate": float(events["jump_clamped"].mean()),
        "downside_share": float((events["event_return"] < 0).mean()),
    }
    for col in events.columns:
        if col.startswith("sigma_post_"):
            summary[f"mean_{col}"] = float(events[col].mean())
    return summary


def implied_jump_move(implied_vol: Union[float, np.ndarray, pd.Series],
                      baseline_vol: Union[float, np.ndarray, pd.Series],
                      days_to_expiry: int,
                      periods_per_year: int = TRADING_DAYS_PER_YEAR
                      ) -> Union[float, np.ndarray, pd.Series]:
    """Back out the market-implied one-off earnings move from a term IV.

    This is the calculation in the source research, stated explicitly. An
    at-the-money implied vol quoted over an expiry that *contains* an earnings
    event prices two things at once. The ordinary diffusive vol that runs for
    the whole term, and a single-session jump. Over ``T = days_to_expiry /
    periods_per_year`` years::

        V_total    = implied_vol^2  * T          total variance priced
        V_diffuse  = baseline_vol^2 * T          variance from ordinary drift
        V_jump     = V_total - V_diffuse         the one-off event variance

        implied_move = sqrt( max(0, V_jump) )

    ``implied_move`` is dimensionless, a decimal one-day move, e.g. ``0.0787``
    for the 7.87% figure in the source deck. It is directly comparable to
    ``abs_event_return`` from :func:`decompose_events`, which is the realized
    counterpart.

    Args:
        implied_vol: Annualized ATM implied vol over the expiry, as a decimal
            (0.666 for 66.6%). Scalar, array or Series.
        baseline_vol: Annualized diffusive vol expected to run over the same
            term, as a decimal. Typically the realized baseline from
            :func:`decompose_events`, or a non-earnings-cycle IV.
        days_to_expiry: Term length in **trading** days. Convert calendar
            days before calling, 30 calendar days is ~21 trading days.
        periods_per_year: Annualization factor.

    Returns:
        The implied one-session move as a decimal, floored at zero.

    Raises:
        ValueError: If ``days_to_expiry`` or ``periods_per_year`` is not
            strictly positive.
    """
    if days_to_expiry <= 0:
        raise ValueError(f"days_to_expiry must be > 0, got {days_to_expiry}")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")

    tau = float(days_to_expiry) / float(periods_per_year)
    iv = np.asarray(implied_vol, dtype=float)
    bv = np.asarray(baseline_vol, dtype=float)
    var_jump = (iv ** 2 - bv ** 2) * tau
    move = np.sqrt(np.clip(var_jump, 0.0, None))

    if isinstance(implied_vol, pd.Series):
        return pd.Series(move, index=implied_vol.index, name="implied_move")
    if np.isscalar(implied_vol) and move.ndim == 0:
        return float(move)
    return move


@dataclass(frozen=True)
class VRPResult:
    """Volatility risk premium / structural edge across an event history.

    Attributes:
        per_event: One row per event with ``implied_move``, ``realized_move``,
            ``vrp`` (implied minus realized, in decimal move points),
            ``vrp_ratio`` (implied / realized) and ``implied_source``.
        mean_vrp: Mean of ``vrp``. Positive means options were, on average,
            priced above the move that actually happened, the seller's edge.
        median_vrp: Median of ``vrp``, robust to single-event outliers.
        mean_vrp_ex_outliers: Mean ``vrp`` after trimming, matching the source
            research's practice of quoting a figure with COVID-era events
            excluded.
        hit_rate: Fraction of events where ``implied_move > realized_move``,
            i.e. the fraction a short-vol position would have won.
        t_stat: One-sample t-statistic of ``vrp`` against zero. Above roughly
            2 in absolute value is conventionally read as the premium being
            statistically distinguishable from noise, though the usual caveats
            about overlapping samples and non-normality apply.
        n_events: Number of events in the comparison.
        n_trimmed: Number of events removed by outlier trimming.
        implied_source: How the implied leg was obtained, ``"market"`` if
            real IV was supplied, ``"historical_proxy"`` otherwise.
        proxy_note: Explanation when a proxy was used.
    """
    per_event: pd.DataFrame
    mean_vrp: float
    median_vrp: float
    mean_vrp_ex_outliers: float
    hit_rate: float
    t_stat: float
    n_events: int
    n_trimmed: int
    implied_source: str
    proxy_note: str = ""

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """Flat dict of the scalar fields, for printing or serialization."""
        return {
            "n_events": self.n_events,
            "implied_source": self.implied_source,
            "mean_vrp": self.mean_vrp,
            "median_vrp": self.median_vrp,
            "mean_vrp_ex_outliers": self.mean_vrp_ex_outliers,
            "hit_rate": self.hit_rate,
            "t_stat": self.t_stat,
            "n_trimmed": self.n_trimmed,
        }


def volatility_risk_premium(events: pd.DataFrame,
                            implied_moves: Optional[Union[pd.Series, float]] = None,
                            implied_vols: Optional[Union[pd.Series, float]] = None,
                            days_to_expiry: int = 21,
                            trim_quantile: float = 0.05,
                            min_events: int = 4) -> VRPResult:
    """Quantify the structural IV edge across a set of historical events.

    The metric is the **volatility risk premium** in move points::

        vrp_i = implied_move_i - realized_move_i

    where ``realized_move_i`` is ``abs_event_return``, the actual
    single-session repricing, and ``implied_move_i`` is what the options
    market charged for it beforehand. A positive mean is the edge that a
    delta-hedged short-vol structure harvests.

    **The implied leg, and what to do when it is unavailable.** Free data
    sources do not retain historical option surfaces, so there are three
    modes, in descending order of fidelity.

    1. ``implied_moves`` supplied, a per-event Series of implied one-session
       moves (decimals). Highest fidelity, use this if you have a surface.
    2. ``implied_vols`` supplied, a per-event Series (or one scalar) of
       annualized ATM IVs over an expiry containing the event. These are
       converted with :func:`implied_jump_move`, using each event's own
       realized baseline as the diffusive term.
    3. Neither supplied, **a historical proxy is used**, and this assumption
       must be stated wherever the result is quoted. The implied move for
       event ``i`` is estimated as the *leave-one-out* mean absolute event
       move across all other events::

           implied_move_i = mean_{j != i} ( abs_event_return_j )

       Leave-one-out matters. Including event ``i`` in its own benchmark would
       shrink every deviation toward zero and bias the premium. What this mode
       actually measures is **event-move dispersion around the historical
       norm**, not a true risk premium. It answers "was this event bigger or
       smaller than this name typically delivers?" and its mean is zero by
       construction. Treat it as a distributional diagnostic and a
       calibration check on the backtest, never as evidence of an edge. Only
       modes 1 and 2 can establish that.

    Args:
        events: Output of :func:`decompose_events`.
        implied_moves: Per-event implied one-session moves, as decimals.
            Aligned on the index of ``events``, a scalar is broadcast.
        implied_vols: Per-event annualized ATM IVs, as decimals. Ignored if
            ``implied_moves`` is given.
        days_to_expiry: Trading days to expiry, used only with
            ``implied_vols``.
        trim_quantile: Two-sided quantile trimmed for the
            ``mean_vrp_ex_outliers`` figure. 0.05 trims the top and bottom 5%.
            Set to 0 to disable.
        min_events: Minimum events required to report a premium.

    Returns:
        A :class:`VRPResult`.

    Raises:
        ValueError: If ``events`` is empty, has fewer than ``min_events`` rows,
            lacks the required columns, or ``trim_quantile`` is outside
            ``[0, 0.5)``.
    """
    if events is None or events.empty:
        raise ValueError("volatility_risk_premium requires a non-empty event frame.")
    for col in ("abs_event_return", "sigma_baseline"):
        if col not in events.columns:
            raise ValueError(f"events frame is missing required column '{col}'.")
    if not 0.0 <= trim_quantile < 0.5:
        raise ValueError(f"trim_quantile must be in [0, 0.5), got {trim_quantile}")
    if len(events) < min_events:
        raise ValueError(
            f"need at least {min_events} events to estimate a premium, got {len(events)}."
        )

    realized = events["abs_event_return"].astype(float)
    n = len(realized)
    proxy_note = ""

    if implied_moves is not None:
        if np.isscalar(implied_moves):
            implied = pd.Series(float(implied_moves), index=events.index)
        else:
            implied = pd.Series(implied_moves).reindex(events.index).astype(float)
        source = "market"
    elif implied_vols is not None:
        if np.isscalar(implied_vols):
            iv = pd.Series(float(implied_vols), index=events.index)
        else:
            iv = pd.Series(implied_vols).reindex(events.index).astype(float)
        # Convert the annualized baseline back to the same footing as the IV.
        baseline = events["sigma_baseline"].astype(float)
        if not events.attrs.get("annualized", True):
            baseline = baseline * np.sqrt(TRADING_DAYS_PER_YEAR)
        implied = pd.Series(
            implied_jump_move(iv.to_numpy(), baseline.to_numpy(), days_to_expiry),
            index=events.index,
        )
        source = "market"
    else:
        # Leave-one-out historical mean, (sum - self) / (n - 1), vectorized.
        total = realized.sum()
        implied = (total - realized) / (n - 1)
        source = "historical_proxy"
        proxy_note = (
            "No option data supplied. The implied leg is a leave-one-out mean of "
            "realized absolute event moves, so the reported premium measures "
            "dispersion of event moves around this name's historical norm and has "
            "a mean of approximately zero by construction. It is NOT evidence of a "
            "volatility risk premium. Supply implied_moves or implied_vols for that."
        )

    valid = implied.notna() & realized.notna()
    if int(valid.sum()) < min_events:
        raise ValueError(
            f"only {int(valid.sum())} event(s) have both an implied and a realized "
            f"move, need {min_events}."
        )

    implied = implied[valid]
    realized_v = realized[valid]
    vrp = implied - realized_v

    per_event = pd.DataFrame({
        "implied_move": implied,
        "realized_move": realized_v,
        "vrp": vrp,
        "vrp_ratio": np.where(realized_v > 0, implied / realized_v, np.nan),
    })
    per_event["implied_source"] = source

    # Trimmed mean, for the "excluding outliers" figure.
    if trim_quantile > 0 and len(vrp) >= 5:
        lo, hi = vrp.quantile(trim_quantile), vrp.quantile(1.0 - trim_quantile)
        kept = vrp[(vrp >= lo) & (vrp <= hi)]
        n_trimmed = len(vrp) - len(kept)
        mean_ex = float(kept.mean()) if len(kept) else float("nan")
    else:
        n_trimmed = 0
        mean_ex = float(vrp.mean())

    sd = float(vrp.std(ddof=1))
    t_stat = float(vrp.mean() / (sd / np.sqrt(len(vrp)))) if sd > 0 else float("nan")

    return VRPResult(
        per_event=per_event,
        mean_vrp=float(vrp.mean()),
        median_vrp=float(vrp.median()),
        mean_vrp_ex_outliers=mean_ex,
        hit_rate=float((implied > realized_v).mean()),
        t_stat=t_stat,
        n_events=int(len(vrp)),
        n_trimmed=int(n_trimmed),
        implied_source=source,
        proxy_note=proxy_note,
    )


def vol_cone(prices: pd.DataFrame,
             windows: Sequence[int] = (5, 10, 20, 30, 60, 90, 120),
             quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
             estimator: str = "close_to_close",
             annualized: bool = True) -> pd.DataFrame:
    """Build a volatility cone. The distribution of realized vol by horizon.

    For each window length, the full history of trailing realized vol is
    computed and reduced to the requested quantiles. Plotting quantile against
    window length gives the classic cone that narrows with horizon, and shows
    at a glance whether current vol is rich or cheap for its horizon.

    Vectorized: one rolling reduction per window, no nested loops.

    Args:
        prices: Validated OHLCV frame.
        windows: Window lengths in trading days.
        quantiles: Quantiles to report, in ``[0, 1]``.
        estimator: ``close_to_close``, ``parkinson`` or ``yang_zhang``.
        annualized: Report annualized vol.

    Returns:
        Frame indexed by window length, with one column per quantile
        (named ``q05``, ``q50``, ...) plus a ``last`` column giving the most
        recent observation at that horizon.

    Raises:
        ValueError: If ``estimator`` is unknown, ``windows`` is empty, or a
            quantile is outside ``[0, 1]``.
        SchemaValidationError: If required columns are absent.
        InsufficientDataError: If no window is short enough for the history.
    """
    if not windows:
        raise ValueError("windows must be non-empty.")
    if any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError(f"quantiles must lie in [0, 1], got {quantiles}")
    if estimator not in _BASELINE_ESTIMATORS:
        raise ValueError(f"estimator must be one of {_BASELINE_ESTIMATORS}, got '{estimator}'")

    from .estimators import rolling_close_to_close  # local import. Avoid cycle noise

    rows: Dict[int, Dict[str, float]] = {}
    usable = [w for w in sorted(windows) if w + 2 <= len(prices)]
    if not usable:
        raise InsufficientDataError(
            "Price history is too short for any requested cone window.",
            required=min(windows) + 2, available=len(prices),
        )

    for w in usable:
        if estimator == "close_to_close":
            series = rolling_close_to_close(prices["Close"], w, annualized=annualized)
        elif estimator == "parkinson":
            series = rolling_parkinson(prices, w, annualized=annualized)
        else:
            series = rolling_yang_zhang(prices, w, annualized=annualized)
        series = series.dropna()
        if series.empty:
            continue
        row = {f"q{int(round(q * 100)):02d}": float(series.quantile(q)) for q in quantiles}
        row["last"] = float(series.iloc[-1])
        row["n_obs"] = float(len(series))
        rows[w] = row

    if not rows:
        raise InsufficientDataError(
            "No cone window produced a usable observation.",
            required=min(usable) + 2, available=len(prices),
        )

    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index.name = "window"
    out.attrs["estimator"] = estimator
    out.attrs["annualized"] = annualized
    out.attrs["ticker"] = prices.attrs.get("ticker", "<unknown>")
    return out
