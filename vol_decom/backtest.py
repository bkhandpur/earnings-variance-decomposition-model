"""Delta-neutral earnings volatility-capture backtest.

What is being simulated
-----------------------
The trade from the source research. Around each earnings event, sell the
straddle that expires *after* the announcement and buy the straddle that
expires *before* it, delta-hedging daily so the position carries no
directional exposure. The short leg holds the event, the long leg neutralizes
the pre-event gamma. What remains is a bet that the market overprices the
one-off earnings jump.

The P&L model
-------------
Each leg is valued by full Black-Scholes revaluation and hedged at every
session close. Over one session the delta-hedged P&L of a long straddle is
exactly::

    PnL_t = [ V(S_{t+1}, tau_{t+1}, sigma_{t+1}) - V(S_t, tau_t, sigma_t) ]
            - Delta_t * ( S_{t+1} - S_t )

summed over the holding period. The first bracket is the change in the option's
mark, the second is the P&L of the hedge that was on. What remains after the
hedge is stripped out is pure volatility exposure, the position makes money
if and only if the underlying delivers more variance than its mark implied.

Revaluation rather than the gamma approximation. The textbook shortcut for a
delta-hedged position is the dollar-gamma form
``0.5 * Gamma * S^2 * (sigma^2 dt - r^2)``. That is a *second-order* expansion
of the payoff, and earnings are precisely where it fails. For a 25% jump the
quadratic term diverges badly from the straddle's true payoff, which is
piecewise linear far from the strike and therefore bounded. Using it produces
losses several times larger than the position can actually sustain, an
artifact of the approximation, not a property of the trade. Full revaluation
costs one extra array of normal CDFs and is exact for the hedging strategy
being simulated, so that is what this module does.

Implied volatility as a term structure, and the IV crush
--------------------------------------------------------
A single flat vol cannot represent an earnings event. Each leg is instead
marked off its *remaining* variance to expiry::

    remaining_var(t) = sigma_baseline^2 * tau_t  +  V_jump * f_t
    sigma(t)         = sqrt( remaining_var(t) / tau_t )

where ``V_jump`` is the one-off event variance the leg was sold at and ``f_t``
is the fraction of the event still ahead at time ``t`` (1 before the
announcement, 0 after). This makes the IV crush an *emergent* property rather
than an assumption. On the session the event resolves, ``f_t`` drops to zero,
the jump loading falls out of the remaining variance, and the mark collapses
toward the diffusive level. That collapse is the short leg's profit, and it is
the mechanism the source research is trading. The long leg of the calendar
expires before the announcement and so never carries a jump loading at all.

The implied-volatility input, and why the default is deliberately edge-free
--------------------------------------------------------------------------
Free data does not retain historical option surfaces, so the level the trade is
sold at has to come from somewhere. Rather than invent a favourable one, the
default construction prices each event at exactly the move this name
historically delivers::

    implied_move_i = k * mean_{j != i} |realized_move_j|

with ``k = implied_move_multiplier``, default **1.0**, and the benchmark taken
leave-one-out so no event is priced using its own outcome. At ``k = 1.0`` the
expected P&L is zero by construction. The position collects the average move
and pays the actual one. What the backtest measures there is the *dispersion*
of outcomes, the risk profile of the structure, not its profitability.

There is a subtlety worth stating, because it is a genuine trap. Options are
quoted in variance, but a delta-hedged short straddle pays out the *absolute*
move. An at-the-money straddle marked at variance ``V`` is worth
``sqrt(2/pi) * sqrt(V) * S``, about 0.798 times the root-variance, because
Black-Scholes assumes a Gaussian move whose mean absolute size is
``sqrt(2/pi)`` of its standard deviation. An earnings jump is not Gaussian --
it is closer to a two-point ``+/-m`` distribution, whose mean absolute size
equals its standard deviation exactly. So a mark that is *variance*-fair for a
jump is roughly 20% short of being *P&L*-fair, and a short position calibrated
on variance alone bleeds at a rate that has nothing to do with mispricing. This
module therefore calibrates on the move and converts::

    V_jump = ( implied_move / sqrt(2/pi) )^2

which is the variance that makes the straddle's event component worth
``implied_move``. That is what makes ``k = 1.0`` genuinely edge-free.

To test a thesis about mispricing, set ``k`` from real data. Either supply
``implied_vols`` from a surface, or raise ``k`` to reflect a measured premium
(the source research measured 2.37 move points, ~2.78 excluding COVID-era
events). Any positive mean return at ``k > 1`` is a direct consequence of that
input assumption and must be reported as such. This is the single most
important caveat in the package. The backtest cannot discover a volatility
risk premium, it can only propagate one you supply.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .estimators import TRADING_DAYS_PER_YEAR
from .exceptions import BacktestError, InsufficientDataError, SchemaValidationError

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "bs_straddle",
    "infer_events_per_year",
    "run_backtest",
]

logger = logging.getLogger(__name__)

#: Floor on time-to-expiry, in years, to keep Black-Scholes gamma finite.
_MIN_TAU = 0.5 / TRADING_DAYS_PER_YEAR

#: E|Z| for a standard normal. An at-the-money straddle is worth
#: ``SQRT_2_OVER_PI * S * sigma * sqrt(tau)``, i.e. this constant times the spot
#: times the square root of the variance it is marked at. It is the conversion
#: factor between a variance quote and the absolute move that variance pays for,
#: and it is why a variance-fair mark is not a P&L-fair mark for a jump. See
#: the module docstring.
SQRT_2_OVER_PI: float = float(np.sqrt(2.0 / np.pi))

_STRUCTURES = ("calendar", "short_straddle")


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF, vectorized."""
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorized.

    Uses :func:`scipy.special.ndtr` when available and falls back to an
    ``math.erf``-based implementation otherwise, so the package remains
    usable without SciPy.
    """
    x = np.asarray(x, dtype=float)
    try:
        from scipy.special import ndtr
        return ndtr(x)
    except ImportError:  # pragma. No cover - environment dependent
        from math import erf
        vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / np.sqrt(2.0))), otypes=[float])
        return vec(x)


def bs_straddle(spot: Union[float, np.ndarray],
                strike: Union[float, np.ndarray],
                vol: Union[float, np.ndarray],
                tau: Union[float, np.ndarray],
                rate: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Black-Scholes price, delta and gamma of a straddle (one call + one put).

    Formulas, for spot ``S``, strike ``K``, vol ``sigma``, time ``tau`` in
    years and rate ``r``::

        d1 = [ ln(S/K) + (r + sigma^2 / 2) * tau ] / (sigma * sqrt(tau))
        d2 = d1 - sigma * sqrt(tau)

        call  = S * Phi(d1) - K * exp(-r*tau) * Phi(d2)
        put   = K * exp(-r*tau) * Phi(-d2) - S * Phi(-d1)
        price = call + put

        delta = Phi(d1) + (Phi(d1) - 1)        = 2*Phi(d1) - 1
        gamma = 2 * phi(d1) / (S * sigma * sqrt(tau))

    The straddle's gamma is twice a single option's because call and put share
    the same gamma. ``tau`` is floored at half a session to keep gamma finite
    at expiry.

    Args:
        spot: Underlying price. Scalar or array (broadcast with the others).
        strike: Strike price.
        vol: Annualized implied volatility as a decimal.
        tau: Time to expiry in years.
        rate: Continuously-compounded risk-free rate. Defaults to 0, which is
            the right simplification here. Over a 1-3 week hold the carry term
            is negligible next to the variance term being measured.

    Returns:
        ``(price, delta, gamma)``, each broadcast to the common shape.

    Raises:
        ValueError: If ``vol`` or ``spot`` or ``strike`` is non-positive
            anywhere.
    """
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    sig = np.asarray(vol, dtype=float)
    t = np.clip(np.asarray(tau, dtype=float), _MIN_TAU, None)

    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("bs_straddle requires strictly positive spot and strike.")
    if np.any(sig <= 0):
        raise ValueError("bs_straddle requires strictly positive volatility.")

    sqrt_t = np.sqrt(t)
    d1 = (np.log(S / K) + (rate + 0.5 * sig ** 2) * t) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    disc = np.exp(-rate * t)

    call = S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    put = K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    price = call + put
    delta = 2.0 * _norm_cdf(d1) - 1.0
    gamma = 2.0 * _norm_pdf(d1) / (S * sig * sqrt_t)
    return price, delta, gamma


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters of the earnings vol-capture backtest.

    All offsets are in **trading sessions** relative to the event session (the
    aligned session from :func:`vol_decom.engine.decompose_events`), never in
    calendar days.

    Attributes:
        structure: ``"calendar"`` for the short-post/long-pre calendar spread
            of the source research, or ``"short_straddle"`` for a plain
            delta-hedged short straddle over the event.
        entry_offset: Sessions before the event session at which the position
            is opened. Default 5 (~one week), matching the research.
        exit_offset: Sessions after the event session at which it is closed.
            Default 1. Close the session after the announcement, once the IV
            crush has happened.
        front_expiry_offset: Sessions after the event session on which the
            short leg expires. Must exceed ``exit_offset`` so the leg is still
            alive at exit, expiring exactly at exit would put the position in
            the gamma singularity.
        back_expiry_offset: Sessions *before* the event session on which the
            long leg expires. Only used for ``"calendar"``. The long leg is
            deliberately dead before the announcement, that is what makes
            the structure an event-isolating calendar rather than a hedge.
        implied_move_multiplier: ``k`` in the front-leg pricing. The event is
            marked at ``k`` times the move this name historically delivers, so
            1.0 (default) is edge-free by construction and ``k = 1.10`` asserts
            the market charges 10% more than the move that arrives. See the
            module docstring for the variance-to-move conversion this uses.
        back_iv_multiplier: Multiple of the baseline vol at which the long leg
            is priced. Default 1.0.
        hedge_band: Reserved for a hedging-friction extension, a value of 0
            means hedge every session at the close (the model assumption).
        transaction_cost_bps: Round-trip cost per leg, in basis points of
            premium, charged at entry. Applied to both legs.
        slippage_vol_points: Volatility points of adverse slippage applied to
            the level each leg trades at, in decimals (0.01 = 1 vol point).
            The short leg is sold 1 point lower and the long leg bought 1
            point higher, so this is always a cost.
        rate: Risk-free rate used in Black-Scholes.
        capital_basis: Denominator for per-event returns.
            ``"front_premium"`` (default) divides by the short leg's premium,
            the standard risk proxy for a short-vol position and always
            strictly positive. ``"net_credit"`` divides by the absolute net
            premium taken in, which is what the trade actually collects but is
            numerically unstable for a calendar whose two legs nearly offset --
            a near-zero credit produces unbounded returns. ``"notional"``
            divides by the entry spot, giving returns per unit of underlying.
        compound: Whether the equity curve compounds returns. Default False.
            A short options position can lose more than its premium, so a
            per-event return below -100% is legitimate, compounding it would
            drive the equity curve negative and make the drawdown meaningless.
            Additive accumulation is the correct model for a strategy that
            allocates a fixed size to each event independently.
        events_per_year: Used to annualize the Sharpe ratio. ``None`` (the
            default) infers it from the median spacing of the actual events,
            which is the right behaviour for a package that has to work on any
            listed name. Hardcoding 4 assumes a quarterly reporter and is
            simply wrong for semi-annual reporters, many foreign issuers, and
            any history with gaps. Set a float to override.
        periods_per_year: Sessions per year.
    """
    structure: str = "calendar"
    entry_offset: int = 5
    exit_offset: int = 1
    front_expiry_offset: int = 3
    back_expiry_offset: int = 3
    implied_move_multiplier: float = 1.0
    back_iv_multiplier: float = 1.0
    hedge_band: float = 0.0
    transaction_cost_bps: float = 25.0
    slippage_vol_points: float = 0.0
    rate: float = 0.0
    capital_basis: str = "front_premium"
    compound: bool = False
    events_per_year: Optional[float] = None
    periods_per_year: int = TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            BacktestError: On any inconsistent or out-of-range parameter.
        """
        if self.structure not in _STRUCTURES:
            raise BacktestError(
                f"structure must be one of {_STRUCTURES}, got '{self.structure}'"
            )
        if self.entry_offset < 1:
            raise BacktestError(f"entry_offset must be >= 1, got {self.entry_offset}")
        if self.exit_offset < 0:
            raise BacktestError(f"exit_offset must be >= 0, got {self.exit_offset}")
        if self.front_expiry_offset <= self.exit_offset:
            raise BacktestError(
                f"front_expiry_offset ({self.front_expiry_offset}) must exceed "
                f"exit_offset ({self.exit_offset}), the short leg has to be alive "
                f"when the position is closed."
            )
        if self.structure == "calendar":
            if self.back_expiry_offset < 1:
                raise BacktestError(
                    f"back_expiry_offset must be >= 1, got {self.back_expiry_offset}"
                )
            if self.back_expiry_offset >= self.entry_offset:
                raise BacktestError(
                    f"back_expiry_offset ({self.back_expiry_offset}) must be less than "
                    f"entry_offset ({self.entry_offset}), otherwise the long leg expires "
                    f"before the position is even opened."
                )
        if self.implied_move_multiplier <= 0:
            raise BacktestError(
                f"implied_move_multiplier must be > 0, got {self.implied_move_multiplier}"
            )
        if self.back_iv_multiplier <= 0:
            raise BacktestError(
                f"back_iv_multiplier must be > 0, got {self.back_iv_multiplier}"
            )
        valid_bases = ("front_premium", "net_credit", "notional")
        if self.capital_basis not in valid_bases:
            raise BacktestError(
                f"capital_basis must be one of {valid_bases}, got '{self.capital_basis}'"
            )
        if self.events_per_year is not None and self.events_per_year <= 0:
            raise BacktestError(
                f"events_per_year must be > 0 or None, got {self.events_per_year}"
            )
        if self.transaction_cost_bps < 0:
            raise BacktestError("transaction_cost_bps must be >= 0.")
        if self.slippage_vol_points < 0:
            raise BacktestError("slippage_vol_points must be >= 0.")

    @property
    def holding_sessions(self) -> int:
        """Number of hedge steps in the holding period."""
        return int(self.entry_offset + self.exit_offset)


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a backtest run.

    Attributes:
        trades: One row per simulated event, with entry/exit dates, the vol
            levels traded, premium, gamma P&L by leg, net P&L, and return.
        equity_curve: Cumulative return path across events, in event order.
        stats: Headline statistics, see :meth:`summary_table`.
        config: The configuration used.
        n_skipped: Events dropped for insufficient surrounding history.
    """
    trades: pd.DataFrame
    equity_curve: pd.Series
    stats: Dict[str, float]
    config: BacktestConfig
    n_skipped: int = 0

    def summary_table(self) -> pd.DataFrame:
        """Format the headline statistics as a two-column frame for printing."""
        labels = {
            "n_trades": "Trades",
            "win_rate": "Win Rate",
            "mean_return": "Mean Return / Event",
            "median_return": "Median Return / Event",
            "std_return": "Std Dev / Event",
            "sharpe_ratio": "Sharpe Ratio (annualized)",
            "profit_factor": "Profit Factor",
            "max_drawdown": "Max Drawdown",
            "total_return": "Cumulative Return",
            "best_trade": "Best Trade",
            "worst_trade": "Worst Trade",
        }
        rows = [(labels.get(k, k), self.stats[k]) for k in labels if k in self.stats]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


def infer_events_per_year(event_dates: pd.DatetimeIndex,
                          default: float = 4.0) -> float:
    """Infer the reporting frequency from the observed spacing of events.

    Uses the **median** calendar gap between consecutive announcements, which
    is robust to the one or two long gaps that appear whenever a provider's
    history is incomplete::

        events_per_year = 365.25 / median_gap_days

    This matters for the Sharpe ratio, which scales by
    ``sqrt(events_per_year)``. Assuming 4 for a semi-annual reporter overstates
    Sharpe by about 40%, so a package intended to run on any listed name should
    not hardcode it.

    Args:
        event_dates: Announcement or event session dates, any order.
        default: Returned when fewer than 3 events make a median meaningless.

    Returns:
        Estimated events per year, clamped to ``[1, 12]`` since no equity
        reports more often than monthly or less often than annually.
    """
    dates = pd.DatetimeIndex(event_dates).sort_values()
    if len(dates) < 3:
        return float(default)
    gaps = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(float)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return float(default)
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return float(default)
    return float(np.clip(365.25 / median_gap, 1.0, 12.0))


def _max_drawdown(equity: np.ndarray, compound: bool) -> float:
    """Maximum peak-to-trough decline of an equity curve.

    For a compounded curve the drawdown is the usual ratio::

        dd_t = equity_t / cummax(equity)_t - 1

    For an additive curve the ratio is wrong, the denominator can approach or
    cross zero, so the decline is measured in units of the starting capital::

        dd_t = equity_t - cummax(equity)_t

    Vectorized via :func:`numpy.maximum.accumulate`.

    Args:
        equity: Equity path, 1.0-based, excluding the starting point.
        compound: Whether the path was built by compounding.

    Returns:
        Max drawdown as a non-positive fraction, 0.0 for an empty path.
    """
    if equity.size == 0:
        return 0.0
    path = np.concatenate(([1.0], equity))
    peak = np.maximum.accumulate(path)
    if compound:
        return float(np.min(path / peak - 1.0))
    return float(np.min(path - peak))


def _hedged_leg_pnl(spot_grid: np.ndarray,
                    spot_pp_grid: np.ndarray,
                    strike: np.ndarray,
                    expiry_pp: np.ndarray,
                    baseline: np.ndarray,
                    jump_var: np.ndarray,
                    jump_remaining: np.ndarray,
                    rate: float,
                    dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delta-hedged P&L of a **long** straddle leg, by full revaluation.

    For each event, over each session step ``t``::

        tau_t            = (expiry_position - position_t) * dt
        remaining_var_t  = sigma_baseline^2 * tau_t + V_jump * f_t
        sigma_t          = sqrt( remaining_var_t / tau_t )
        V_t              = BS_straddle(S_t, K, sigma_t, tau_t)     (intrinsic once expired)
        PnL_t            = (V_{t+1} - V_t) - Delta_t * (S_{t+1} - S_t)

    and the leg's P&L is the sum of ``PnL_t`` over steps where it was still
    alive. This is the **long** side, callers negate it for short legs.

    Marking off remaining variance rather than a flat vol is what produces the
    IV crush endogenously, ``f_t`` falls to zero once the event has passed, the
    jump loading leaves the remaining variance, and ``sigma_t`` collapses.

    Every quantity is an ``(n_events, n_steps + 1)`` matrix, so the entire
    event history is revalued in one set of array operations, there is no
    loop over events or over hedge dates.

    Args:
        spot_grid: ``(m, H+1)`` spot at each grid point, entry through exit.
        spot_pp_grid: ``(m, H+1)`` price-index position of each grid point.
        strike: ``(m, 1)`` fixed strike per event.
        baseline: ``(m, 1)`` annualized diffusive vol the leg is marked against.
        expiry_pp: ``(m,)`` price-index position of the leg's expiry.
        jump_var: ``(m, 1)`` one-off event variance loaded into the mark.
        jump_remaining: ``(m, H+1)`` fraction of that loading still ahead.
        rate: Risk-free rate.
        dt: One session in years.

    Returns:
        ``(pnl, entry_value, entry_iv)``, length-``m`` arrays giving the
        accumulated long-side P&L, the leg's value at entry (its premium), and
        the effective IV it was marked at on entry.
    """
    tau = (expiry_pp[:, None] - spot_pp_grid) * dt
    alive = tau > 0.0
    # Guard the division for expired points, their values are overwritten below.
    safe_tau = np.where(alive, tau, 1.0)

    remaining_var = (baseline ** 2) * safe_tau + jump_var * jump_remaining
    sigma = np.sqrt(np.clip(remaining_var / safe_tau, 1e-12, None))

    value, delta, _ = bs_straddle(spot_grid, strike, sigma, safe_tau, rate=rate)
    intrinsic = np.abs(spot_grid - strike)
    value = np.where(alive, value, intrinsic)
    delta = np.where(alive, delta, 0.0)

    d_value = value[:, 1:] - value[:, :-1]
    d_spot = spot_grid[:, 1:] - spot_grid[:, :-1]
    step_pnl = d_value - delta[:, :-1] * d_spot

    pnl = np.sum(np.where(alive[:, :-1], step_pnl, 0.0), axis=1)
    return pnl, value[:, 0], sigma[:, 0]


def run_backtest(prices: pd.DataFrame,
                 events: pd.DataFrame,
                 config: Optional[BacktestConfig] = None,
                 implied_vols: Optional[Union[pd.Series, float]] = None,
                 implied_vol_days: Optional[int] = None,
                 ) -> BacktestResult:
    """Simulate the delta-neutral earnings vol trade across an event history.

    Mechanics per event, all offsets in trading sessions from the event session.

    1. Open at ``entry_offset`` sessions before the event. Strikes are struck
       at the entry close for both legs (at-the-money) and held fixed.
    2. Sell the front straddle expiring ``front_expiry_offset`` sessions after
       the event, for ``"calendar"``, buy the back straddle expiring
       ``back_expiry_offset`` sessions before it.
    3. Delta-hedge at every session close. The resulting P&L is the dollar-gamma
       weighted variance difference given in the module docstring.
    4. Close at ``exit_offset`` sessions after the event.

    Returns are expressed as a fraction of the entry capital basis, net of
    ``transaction_cost_bps`` and ``slippage_vol_points``.

    The implementation is fully vectorized over events. Spot, time-to-expiry
    and expiry masks are built as ``(n_events, n_steps)`` matrices and reduced
    in one pass. The only Python-level loop is over the (at most two) legs of
    the structure, which is a loop over strategy definition, not over data.

    Args:
        prices: Validated OHLCV frame, the same one passed to
            :func:`vol_decom.engine.decompose_events`.
        events: Output of :func:`vol_decom.engine.decompose_events`.
        config: Backtest configuration.
        implied_vols: Optional per-event annualized ATM IV for the front leg,
            as decimals. Supplying real surface data here is what turns this
            from a risk simulation into an edge measurement. A scalar is
            broadcast to all events.
        implied_vol_days: Trading days to expiry for ``implied_vols``. When
            omitted, the quote is assumed to match the front option's term.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        BacktestError: If the configuration is unusable, or no event has
            enough surrounding history to simulate.
        SchemaValidationError: If ``prices`` lacks a ``Close`` column or
            ``events`` lacks the expected columns.
        InsufficientDataError: If the price history is shorter than one
            holding period.
    """
    cfg = config or BacktestConfig()

    if implied_vol_days is not None and implied_vol_days <= 0:
        raise BacktestError("implied_vol_days must be strictly positive.")

    if "Close" not in prices.columns:
        raise SchemaValidationError("run_backtest requires a 'Close' column in prices.")
    if events is None or events.empty:
        raise BacktestError("run_backtest requires a non-empty event frame.")
    for col in ("return_position", "sigma_baseline", "abs_event_return"):
        if col not in events.columns:
            raise SchemaValidationError(f"events frame is missing required column '{col}'.")

    close = prices["Close"].to_numpy(dtype=float)
    log_rets = np.diff(np.log(close))  # log_rets[j] moves price position j -> j+1
    n_prices = close.size
    if n_prices < cfg.holding_sessions + 2:
        raise InsufficientDataError(
            "Price history is shorter than one holding period.",
            required=cfg.holding_sessions + 2, available=n_prices,
        )

    # Event session in *price* index positions. Returns index j <-> price index j+1.
    event_pp = events["return_position"].to_numpy(dtype=int) + 1

    # --- window-fit filter (vectorized)
    back_need = max(cfg.entry_offset, cfg.back_expiry_offset
                    if cfg.structure == "calendar" else 0)
    fwd_need = max(cfg.exit_offset, cfg.front_expiry_offset)
    usable = (event_pp - back_need >= 0) & (event_pp + fwd_need < n_prices)
    n_skipped = int((~usable).sum())
    if not usable.any():
        raise BacktestError(
            f"No event has enough surrounding history for this configuration "
            f"(needs {back_need} sessions before and {fwd_need} after each event, "
            f"history has {n_prices} sessions)."
        )
    ev = events.loc[usable].copy()
    event_pp = event_pp[usable]
    m = event_pp.size
    if n_skipped:
        logger.warning("Backtest skipped %d event(s) for insufficient history.", n_skipped)

    H = cfg.holding_sessions
    dt = 1.0 / float(cfg.periods_per_year)

    # --- grids, H+1 points from entry through exit, so H hedge steps
    entry_pp = event_pp - cfg.entry_offset
    spot_pp_grid = entry_pp[:, None] + np.arange(H + 1)[None, :]
    spot_grid = close[spot_pp_grid]
    strike = close[entry_pp][:, None]
    exit_spot = close[event_pp + cfg.exit_offset]

    # --- volatility inputs
    baseline = ev["sigma_baseline"].to_numpy(dtype=float)
    if not events.attrs.get("annualized", True):
        baseline = baseline * np.sqrt(cfg.periods_per_year)
    if np.any(~np.isfinite(baseline)) or np.any(baseline <= 0):
        raise BacktestError(
            "Non-positive or non-finite baseline volatility in the event frame, "
            "cannot price options against it."
        )

    # Number of sessions the decomposition treated as the event window.
    ev_cfg = events.attrs.get("config", None)
    event_window = int(getattr(ev_cfg, "event_window", 2))
    front_tau0 = (cfg.entry_offset + cfg.front_expiry_offset) * dt

    if implied_vols is not None:
        if np.isscalar(implied_vols):
            front_iv_quoted = np.full(m, float(implied_vols))
        else:
            front_iv_quoted = pd.Series(implied_vols).reindex(ev.index).to_numpy(dtype=float)
        if np.any(~np.isfinite(front_iv_quoted)) or np.any(front_iv_quoted <= 0):
            raise BacktestError("implied_vols must be finite and strictly positive.")
        # Decompose the quoted flat IV into its diffusive and event parts, so the
        # event loading can be released on the event session (the IV crush)
        # instead of being smeared across the whole term.
        quote_tau = (implied_vol_days * dt
                     if implied_vol_days is not None else front_tau0)
        jump_var_total = np.clip(
            (front_iv_quoted ** 2 - baseline ** 2) * quote_tau, 0.0, None
        )
        iv_source = "supplied"
    else:
        if m < 2:
            raise BacktestError(
                "The default implied-vol construction needs at least 2 events for its "
                "leave-one-out benchmark, supply implied_vols instead."
            )
        # Leave-one-out mean absolute event move. What this name typically
        # delivers, computed without using the event's own outcome.
        abs_move = ev["abs_event_return"].to_numpy(dtype=float)
        loo_move = (np.nansum(abs_move) - abs_move) / (m - 1)
        implied_move = cfg.implied_move_multiplier * loo_move
        # Convert the move into the variance that makes the straddle's event
        # component worth exactly that move. Calibrating on variance directly
        # would leave the position ~20% underpriced for a jump-like move, see
        # the module docstring.
        jump_var_total = (implied_move / SQRT_2_OVER_PI) ** 2
        iv_source = f"historical_move(k={cfg.implied_move_multiplier:g})"

    # Adverse slippage in vol points. Sell the short leg lower, buy the long
    # leg higher. Applied to the level each leg trades at.
    slip = cfg.slippage_vol_points
    front_baseline = np.clip(baseline - slip, 1e-6, None)
    back_baseline = baseline + slip

    # --- fraction of the event loading still ahead, at each grid point.
    # The event return sits at step `entry_offset - 1` (the return dated on the
    # event session), the window extends forward from there.
    event_steps = [cfg.entry_offset - 1 + j for j in range(event_window)]
    in_path = [t for t in event_steps if 0 <= t < H]
    if not in_path:
        raise BacktestError(
            f"The {event_window}-session event window does not intersect the holding "
            f"path (entry_offset={cfg.entry_offset}, exit_offset={cfg.exit_offset}). "
            f"Increase exit_offset so the position is still open on the event session."
        )
    # The loading stays fully priced until the whole event window has passed,
    # then drops to zero. Releasing it gradually across the window would require
    # knowing which of the window's sessions carries the jump, information the
    # position does not have at the time (providers report announcement times
    # unreliably, which is why the window is 2 sessions wide in the first
    # place). Holding the mark up until the window closes is the no-look-ahead
    # convention, and it makes the premium collected match the variance the
    # window can actually deliver.
    last_event_step = max(in_path)
    grid_t = np.arange(H + 1)
    jump_remaining = np.tile(
        (grid_t <= last_event_step).astype(float)[None, :], (m, 1)
    )

    # --- front (short) leg
    front_expiry_pp = event_pp + cfg.front_expiry_offset
    front_pnl_long, front_premium, front_iv = _hedged_leg_pnl(
        spot_grid, spot_pp_grid, strike, front_expiry_pp,
        front_baseline[:, None], jump_var_total[:, None], jump_remaining,
        cfg.rate, dt,
    )
    front_pnl = -front_pnl_long  # we are short the front leg

    # --- back (long) leg, calendar only. It expires before the announcement,
    # so it carries no event loading, only ordinary diffusive variance.
    if cfg.structure == "calendar":
        back_expiry_pp = event_pp - cfg.back_expiry_offset
        back_pnl, back_premium, back_iv = _hedged_leg_pnl(
            spot_grid, spot_pp_grid, strike, back_expiry_pp,
            back_baseline[:, None], np.zeros((m, 1)), np.zeros((m, H + 1)),
            cfg.rate, dt,
        )
    else:
        back_pnl = np.zeros(m)
        back_premium = np.zeros(m)
        back_iv = np.full(m, np.nan)

    net_credit = front_premium - back_premium
    gross_pnl = front_pnl + back_pnl

    # Transaction cost. Bps of the gross premium traded on both legs.
    cost = (cfg.transaction_cost_bps / 10_000.0) * (front_premium + back_premium)
    net_pnl = gross_pnl - cost

    # --- returns
    if cfg.capital_basis == "front_premium":
        basis = front_premium.copy()
    elif cfg.capital_basis == "net_credit":
        basis = np.abs(net_credit)
        # A calendar whose legs nearly offset has a near-zero credit, which
        # makes the return unbounded. Fall back to the gross premium at risk.
        degenerate = basis < 0.05 * front_premium
        if degenerate.any():
            logger.warning(
                "%d event(s) had a net credit below 5%% of the front premium, "
                "their returns use the gross premium as the capital basis instead.",
                int(degenerate.sum()),
            )
        basis = np.where(degenerate, front_premium + back_premium, basis)
    else:
        basis = close[entry_pp].astype(float)
    if np.any(basis <= 0):
        raise BacktestError("Non-positive capital basis, cannot compute returns.")
    ret = net_pnl / basis

    trades = pd.DataFrame({
        "announcement_date": ev["announcement_date"].to_numpy(),
        "entry_date": prices.index[entry_pp],
        "event_date": prices.index[event_pp],
        "exit_date": prices.index[event_pp + cfg.exit_offset],
        "entry_spot": close[entry_pp],
        "exit_spot": exit_spot,
        "strike": close[entry_pp],
        "sigma_baseline": baseline,
        "front_iv": front_iv,
        "implied_event_move": np.sqrt(jump_var_total),
        "back_iv": back_iv,
        "front_premium": front_premium,
        "back_premium": back_premium,
        "net_credit": net_credit,
        "realized_move": ev["abs_event_return"].to_numpy(dtype=float),
        "front_pnl": front_pnl,
        "back_pnl": back_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
        "capital_basis": basis,
        "return": ret,
    }, index=pd.DatetimeIndex(ev.index, name="session_date"))
    trades.attrs["iv_source"] = iv_source

    # --- statistics
    # Additive unless `compound` is set. A short options position can lose more
    # than its premium, so compounding a sub -100% return would send the equity
    # curve negative and make every downstream statistic meaningless.
    equity = np.cumprod(1.0 + ret) if cfg.compound else 1.0 + np.cumsum(ret)
    epy = (cfg.events_per_year if cfg.events_per_year is not None
           else infer_events_per_year(pd.DatetimeIndex(ev.index)))
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    sd = float(np.std(ret, ddof=1)) if ret.size > 1 else 0.0

    stats: Dict[str, float] = {
        "n_trades": float(ret.size),
        "win_rate": float((ret > 0).mean()),
        "mean_return": float(ret.mean()),
        "median_return": float(np.median(ret)),
        "std_return": sd,
        "sharpe_ratio": (
            float(ret.mean() / sd * np.sqrt(epy)) if sd > 0 else float("nan")
        ),
        "profit_factor": (
            float(gross_win / gross_loss) if gross_loss > 0
            else (float("inf") if gross_win > 0 else float("nan"))
        ),
        "max_drawdown": _max_drawdown(equity, cfg.compound),
        "total_return": float(equity[-1] - 1.0) if equity.size else 0.0,
        "best_trade": float(ret.max()),
        "worst_trade": float(ret.min()),
        "mean_net_credit": float(net_credit.mean()),
        "mean_front_pnl": float(front_pnl.mean()),
        "mean_back_pnl": float(back_pnl.mean()),
        "mean_cost": float(cost.mean()),
        "mean_implied_move": float(np.sqrt(jump_var_total).mean() * SQRT_2_OVER_PI),
        "mean_realized_move": float(ev["abs_event_return"].mean()),
        "events_per_year": epy,
    }

    equity_curve = pd.Series(equity, index=trades.index, name="equity")

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        stats=stats,
        config=cfg,
        n_skipped=n_skipped,
    )
