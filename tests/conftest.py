"""Shared synthetic fixtures. No network access anywhere in the test suite."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

TRADING_DAYS = 252


def make_ohlcv(returns: np.ndarray,
               start_price: float = 100.0,
               start: str = "2019-01-02",
               intraday_range: float = 0.006,
               seed: int = 0,
               ticker: str = "TEST") -> pd.DataFrame:
    """Build a valid OHLCV frame from a series of log returns.

    Open/High/Low are synthesized around the close path so that the frame
    satisfies every constraint :func:`vol_decom.data.validate_ohlcv` enforces:
    strictly positive prices, ``High >= Low``, and Open/Close inside the range.

    Args:
        returns: Log returns, the frame has ``len(returns) + 1`` rows.
        start_price: Initial close.
        start: First session date (business days follow).
        intraday_range: Scale of the synthetic intraday range.
        seed: RNG seed for the range noise.
        ticker: Value stored in ``attrs['ticker']``.

    Returns:
        A validated-shape OHLCV frame indexed by business days.
    """
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.concatenate(([0.0], np.cumsum(returns))))
    n = close.size
    idx = pd.bdate_range(start, periods=n)

    up = np.abs(rng.normal(0.0, intraday_range, n))
    dn = np.abs(rng.normal(0.0, intraday_range, n))
    high = close * (1.0 + up)
    low = close * (1.0 - dn)
    prev = np.concatenate(([close[0]], close[:-1]))
    open_ = np.clip(prev * (1.0 + rng.normal(0.0, intraday_range / 2, n)), low, high)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": np.full(n, 1_000_000.0)},
        index=idx,
    )
    df.index.name = "Date"
    df.attrs["ticker"] = ticker
    return df


@pytest.fixture
def quiet_returns() -> np.ndarray:
    """600 sessions of homoskedastic 2%/day returns, no jumps."""
    rng = np.random.default_rng(11)
    return rng.normal(0.0, 0.02, 600)


@pytest.fixture
def jump_scenario() -> Tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """A quiet diffusion with a large, known jump on each earnings reaction.

    Baseline vol is a constant 1%/session. Every 63 sessions a +12% jump is
    injected. The announcement is dated the session *before* the jump, which
    is the after-the-close convention. With the default 2-session event window
    the jump is the second observation.

    Returns:
        ``(prices, announcement_dates, jump_positions)`` where
        ``jump_positions`` index into the return array.
    """
    rng = np.random.default_rng(5)
    n = 700
    rets = rng.normal(0.0, 0.01, n)
    jump_pos = np.arange(120, n - 60, 63)
    rets[jump_pos] = 0.12
    prices = make_ohlcv(rets, ticker="JUMPY")
    # returns index position p <-> price index position p+1
    announce = pd.DatetimeIndex([prices.index[p] for p in jump_pos])
    return prices, announce, jump_pos


@pytest.fixture
def zero_jump_scenario() -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """A price path where the 'event' sessions are *calmer* than the baseline.

    Baseline is 3%/session, the two designated event sessions are pinned to a
    near-zero move. The decomposition must therefore clamp the jump variance
    at zero rather than returning a NaN from a negative square root.
    """
    rng = np.random.default_rng(3)
    n = 400
    rets = rng.normal(0.0, 0.03, n)
    quiet_pos = np.array([200, 300])
    for p in quiet_pos:
        rets[p] = 1e-6
        rets[p + 1] = 1e-6
    prices = make_ohlcv(rets, ticker="CALM")
    announce = pd.DatetimeIndex([prices.index[p] for p in quiet_pos])
    return prices, announce
