"""vol_decom, earnings volatility variance decomposition.

Separates the discrete jump variance released by an earnings announcement from
the continuous diffusive volatility that runs the rest of the time, using the
additivity of variance across independent components::

    sigma_jump = sqrt( max(0, sigma_total^2 - sigma_baseline^2) )

and turns the resulting event history into a measure of the structural
volatility risk premium and a delta-neutral backtest of harvesting it.

Modules
-------
:mod:`vol_decom.data`
    yfinance loading with Parquet caching, OHLCV schema validation and
    explicit gap/holiday handling.
:mod:`vol_decom.estimators`
    Close-to-close, Parkinson and Yang-Zhang realized-vol estimators,
    annualized by ``sqrt(252)``.
:mod:`vol_decom.engine`
    Trading-day event alignment, the variance decomposition itself, volatility
    cones and the volatility risk premium metric.
:mod:`vol_decom.backtest`
    Vectorized delta-neutral straddle/calendar simulator with the standard
    performance statistics.
:mod:`vol_decom.visualizer`
    Matplotlib/Plotly figures, every function returns a figure object.
:mod:`vol_decom.exceptions`
    Typed exception hierarchy.

Quickstart
----------
::

    from vol_decom.data import load_prices, load_earnings_dates
    from vol_decom.engine import decompose_events, summarize_decomposition

    prices = load_prices("INTC", years=7)
    events = decompose_events(prices, load_earnings_dates("INTC"))
    print(summarize_decomposition(events))
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "__version__",
    # exceptions
    "VolDecomError", "DataError", "DataFetchError", "SchemaValidationError",
    "InsufficientDataError", "MissingEarningsDatesError", "CacheError",
    "AlignmentError", "BacktestError",
    # estimators
    "TRADING_DAYS_PER_YEAR", "annualize", "log_returns", "close_to_close",
    "parkinson", "yang_zhang", "rolling_close_to_close", "rolling_parkinson",
    "rolling_yang_zhang", "realized_variance",
    # data
    "load_prices", "load_earnings_dates", "load_earnings_dates_from_csv",
    "validate_ohlcv", "detect_gaps", "clear_cache",
    # engine
    "DecompositionConfig", "EventAlignment", "align_events", "decompose_events",
    "summarize_decomposition", "implied_jump_move", "volatility_risk_premium",
    "VRPResult", "vol_cone",
    # backtest
    "BacktestConfig", "BacktestResult", "run_backtest", "bs_straddle",
    # visualizer
    "plot_vol_cone", "plot_jump_distribution", "plot_decomposition_timeline",
    "plot_pnl_curve", "plot_vrp_scatter", "save_figure",
]

from .exceptions import (
    AlignmentError,
    BacktestError,
    CacheError,
    DataError,
    DataFetchError,
    InsufficientDataError,
    MissingEarningsDatesError,
    SchemaValidationError,
    VolDecomError,
)
from .estimators import (
    TRADING_DAYS_PER_YEAR,
    annualize,
    close_to_close,
    log_returns,
    parkinson,
    realized_variance,
    rolling_close_to_close,
    rolling_parkinson,
    rolling_yang_zhang,
    yang_zhang,
)
from .data import (
    clear_cache,
    detect_gaps,
    load_earnings_dates,
    load_earnings_dates_from_csv,
    load_prices,
    validate_ohlcv,
)
from .engine import (
    DecompositionConfig,
    EventAlignment,
    VRPResult,
    align_events,
    decompose_events,
    implied_jump_move,
    summarize_decomposition,
    vol_cone,
    volatility_risk_premium,
)
from .backtest import BacktestConfig, BacktestResult, bs_straddle, run_backtest
from .visualizer import (
    plot_decomposition_timeline,
    plot_jump_distribution,
    plot_pnl_curve,
    plot_vol_cone,
    plot_vrp_scatter,
    save_figure,
)
