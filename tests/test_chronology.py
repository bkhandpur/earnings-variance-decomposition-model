import numpy as np
import pandas as pd
import pytest
from vol_decom.engine import decompose_events, DecompositionConfig
from vol_decom.backtest import run_backtest
from vol_decom.exceptions import BacktestError, SchemaValidationError
from tests.conftest import make_ohlcv


def sample():
    r = 0.005 * np.sin(np.arange(400)) + 0.003 * np.cos(np.arange(400) * 0.31)
    pos = np.array([80, 140, 200, 260, 320])
    r[pos] = [0.04, -0.06, 0.08, -0.03, 0.1]
    prices = make_ohlcv(r)
    dates = prices.index[pos + 1]
    return prices, dates


def test_independent_entry_inputs_and_warmup():
    prices, dates = sample()
    events = decompose_events(prices, dates)
    result = run_backtest(prices, events)
    assert result.n_skipped == 2
    first = result.trades.iloc[0]
    assert first.history_events == 2
    assert first.implied_event_move == pytest.approx(0.05)
    entry = prices.index.get_loc(first.entry_date)
    r = np.diff(np.log(prices.Close.to_numpy()))[entry - 20 : entry]
    assert first.sigma_baseline == pytest.approx(np.sqrt(np.mean(r * r) * 252))


@pytest.mark.parametrize("estimator", ["close_to_close", "parkinson", "yang_zhang"])
def test_future_prices_do_not_change_entry_decisions(estimator):
    prices, dates = sample()
    cfg = DecompositionConfig(estimator=estimator)
    result = run_backtest(prices, decompose_events(prices, dates, cfg))
    changed = prices.copy()
    entry = result.trades.iloc[0].entry_date
    # Change all OHLC consistently after entry, including that trade's exit.
    changed.loc[changed.index > entry, ["Open", "High", "Low", "Close"]] *= 1.7
    future = run_backtest(changed, decompose_events(changed, dates, cfg))
    cols = [
        "entry_spot",
        "strike",
        "sigma_baseline",
        "implied_event_move",
        "front_iv",
        "front_premium",
        "back_premium",
        "capital_basis",
        "history_events",
    ]
    np.testing.assert_allclose(
        result.trades.iloc[0][cols].astype(float), future.trades.iloc[0][cols].astype(float)
    )
    assert result.trades.iloc[0].net_pnl != future.trades.iloc[0].net_pnl


def test_future_event_outcomes_and_row_order_do_not_change_earlier_inputs():
    prices, dates = sample()
    events = decompose_events(prices, dates)
    earlier = run_backtest(prices, events)
    changed = events.copy()
    changed.loc[dates[-1], "abs_event_return"] = 50
    future = run_backtest(prices, changed.iloc[::-1])
    pd.testing.assert_frame_equal(earlier.trades.iloc[:-1], future.trades.iloc[:-1])


def test_timestamped_iv_and_constant_scenario():
    prices, dates = sample()
    events = decompose_events(prices, dates)
    iv = pd.Series(0.5, index=events.index)
    with pytest.raises(BacktestError, match="timestamps"):
        run_backtest(prices, events, implied_vols=iv)
    observed = pd.Series(pd.Timestamp("2010-01-01", tz="UTC"), index=events.index)
    result = run_backtest(prices, events, implied_vols=iv, implied_observed_at=observed)
    assert result.trades.attrs["iv_source"] == "timestamped_iv"
    observed.iloc[0] = pd.Timestamp("2099-01-01", tz="UTC")
    with pytest.raises(BacktestError, match="entry close"):
        run_backtest(prices, events, implied_vols=iv, implied_observed_at=observed)


def test_invalid_close_and_event_positions():
    prices, dates = sample()
    events = decompose_events(prices, dates)
    prices.iloc[10, prices.columns.get_loc("Close")] = np.inf
    with pytest.raises(SchemaValidationError, match="finite"):
        run_backtest(prices, events)
