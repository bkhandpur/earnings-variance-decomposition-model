import math
from dataclasses import replace

import numpy as np
import pytest

from main import main
from tests.test_chronology import sample
from vol_decom.backtest import BacktestConfig, bs_straddle, run_backtest
from vol_decom.engine import decompose_events


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_pricing_finite_inputs(index, bad):
    args = [100, 100, 0.2, 0.1, 0.03]
    args[index] = bad
    with pytest.raises(ValueError, match="finite"):
        bs_straddle(*args[:4], rate=args[4])


def test_expiry_and_broadcast():
    price, delta, gamma = bs_straddle(np.array([90, 100, 110]), 100, 0.2, 0)
    np.testing.assert_array_equal(price, [10, 0, 10])
    np.testing.assert_array_equal(delta, [-1, 0, 1])
    np.testing.assert_array_equal(gamma, [0, 0, 0])
    with pytest.raises(ValueError, match="nonnegative"):
        bs_straddle(100, 100, 0.2, -0.1)


def reference_bs(s, k, vol, tau):
    # Independent scalar CDF and closed form, without the implementation.
    cdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    d1 = (math.log(s / k) + 0.5 * vol * vol * tau) / (vol * math.sqrt(tau))
    d2 = d1 - vol * math.sqrt(tau)
    return s * (2 * cdf(d1) - 1) + k * (1 - 2 * cdf(d2))


@pytest.mark.parametrize("structure", ["calendar", "short_straddle"])
def test_entry_slippage_fixed_model_path(structure):
    prices, dates = sample()
    events = decompose_events(prices, dates)
    cfg = BacktestConfig(structure=structure, transaction_cost_bps=0)
    zero = run_backtest(prices, events, cfg).trades
    slip = run_backtest(prices, events, replace(cfg, slippage_vol_points=0.02)).trades
    for col in [
        "front_iv",
        "back_iv",
        "front_premium",
        "back_premium",
        "front_pnl",
        "back_pnl",
        "model_gross_pnl",
    ]:
        np.testing.assert_allclose(zero[col], slip[col], equal_nan=True)
    for _, row in slip.iterrows():
        front = reference_bs(
            row.entry_spot,
            row.strike,
            max(row.front_iv - 0.02, 1e-6),
            (cfg.entry_offset + cfg.front_expiry_offset) / cfg.periods_per_year,
        )
        assert row.front_entry_execution == pytest.approx(front, abs=1e-10)
        assert row.execution_adjustment <= 0
        assert row.net_pnl == pytest.approx(
            row.model_gross_pnl + row.execution_adjustment - row.cost
        )
    assert (slip.net_pnl <= zero.net_pnl).all()


def test_cli_required_backtest_and_decomposition_only(tmp_path):
    common = [
        "--ticker",
        "SYNTHETIC",
        "--prices-csv",
        "examples/synthetic_prices.csv",
        "--earnings-csv",
        "examples/synthetic_events.csv",
    ]
    assert main(common + ["--entry-offset", "1"]) != 0
    assert main(common + ["--no-backtest"]) == 0


def test_failed_stage_export_and_unknown_file(tmp_path):
    import json

    common = [
        "--ticker",
        "SYNTHETIC",
        "--prices-csv",
        "examples/synthetic_prices.csv",
        "--earnings-csv",
        "examples/synthetic_events.csv",
    ]
    assert main(common + ["--entry-offset", "1", "--save-csv", "--outdir", str(tmp_path)]) == 1
    stages = json.loads((tmp_path / "SYNTHETIC_stages.json").read_text())["stages"]
    assert stages["decomposition"] == "complete"
    assert stages["backtest"].startswith("failed")
    assert (
        main(
            [
                "--ticker",
                "SYNTHETIC",
                "--prices-csv",
                str(tmp_path / "missing.csv"),
                "--earnings-csv",
                "examples/synthetic_events.csv",
            ]
        )
        == 1
    )
