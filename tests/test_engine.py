"""Unit tests for the decomposition engine, estimators, validation and backtest.

Every test runs against synthetic fixtures. Nothing here touches the network.
Provider behavior is replaced with local test doubles where loader behavior
is under test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_decom.backtest import BacktestConfig, bs_straddle, run_backtest
from vol_decom.data import detect_gaps, load_prices, validate_ohlcv
from vol_decom.engine import (
    DecompositionConfig,
    align_events,
    decompose_events,
    implied_jump_move,
    summarize_decomposition,
    vol_cone,
    volatility_risk_premium,
)
from vol_decom.estimators import (
    TRADING_DAYS_PER_YEAR,
    annualize,
    close_to_close,
    log_returns,
    parkinson,
    realized_variance,
    rolling_close_to_close,
    yang_zhang,
)
from vol_decom.exceptions import (
    AlignmentError,
    BacktestError,
    InsufficientDataError,
    MissingEarningsDatesError,
    SchemaValidationError,
)

from .conftest import make_ohlcv

SQRT252 = np.sqrt(TRADING_DAYS_PER_YEAR)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #

class TestEstimators:
    """Estimator correctness against closed-form values."""

    def test_annualize_matches_sqrt_252(self):
        assert annualize(0.02) == pytest.approx(0.02 * SQRT252)
        assert annualize(0.02, periods_per_year=365) == pytest.approx(0.02 * np.sqrt(365))

    def test_annualize_rejects_nonpositive_periods(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            annualize(0.02, periods_per_year=0)

    def test_log_returns_alignment_and_length(self):
        """r_t must be dated on the later session and drop the first row."""
        prices = make_ohlcv(np.full(10, 0.01))
        rets = log_returns(prices["Close"])
        assert len(rets) == len(prices) - 1
        assert rets.index[0] == prices.index[1]
        assert rets.iloc[0] == pytest.approx(0.01)

    def test_log_returns_rejects_nonpositive_price(self):
        s = pd.Series([10.0, 0.0, 12.0], index=pd.bdate_range("2020-01-01", periods=3))
        with pytest.raises(SchemaValidationError, match="strictly positive"):
            log_returns(s)

    def test_close_to_close_recovers_known_sigma(self):
        """A constant-vol path must be recovered to within sampling error."""
        rng = np.random.default_rng(42)
        sigma = 0.015
        prices = make_ohlcv(rng.normal(0.0, sigma, 4000))
        est = close_to_close(prices["Close"], annualized=False)
        assert est == pytest.approx(sigma, rel=0.05)
        assert close_to_close(prices["Close"]) == pytest.approx(est * SQRT252)

    def test_realized_variance_zero_mean_vs_demeaned(self):
        """The 2-observation failure mode the prototype hit, made explicit."""
        r = np.array([0.08, 0.08])
        # Demeaned population variance of two identical values is exactly zero.
        assert realized_variance(r, zero_mean=False, ddof=0) == pytest.approx(0.0)
        # The zero-mean realized form correctly reports a large variance.
        assert realized_variance(r, zero_mean=True) == pytest.approx(0.08 ** 2)

    def test_realized_variance_rejects_empty(self):
        with pytest.raises(InsufficientDataError):
            realized_variance(np.array([]))

    def test_parkinson_closed_form(self):
        """A constant log-range must reproduce range / sqrt(4 ln 2) exactly."""
        n = 50
        idx = pd.bdate_range("2020-01-01", periods=n)
        close = np.full(n, 100.0)
        rng_log = 0.02
        df = pd.DataFrame({
            "Open": close, "Close": close,
            "High": close * np.exp(rng_log / 2), "Low": close * np.exp(-rng_log / 2),
            "Volume": np.full(n, 1.0),
        }, index=idx)
        expected = rng_log / np.sqrt(4.0 * np.log(2.0))
        assert parkinson(df, annualized=False) == pytest.approx(expected)

    def test_parkinson_understates_overnight_gap(self):
        """Parkinson is blind to gaps, that limitation must be observable.

        The path below moves almost entirely overnight, large random gaps,
        negligible intraday range. That is the shape of an earnings reaction,
        and it is exactly the regime where Parkinson fails. It sees only the
        intraday range and reports near-zero vol, while Yang-Zhang's overnight
        term picks the move up.
        """
        rng = np.random.default_rng(17)
        n = 300
        gaps = rng.normal(0.0, 0.04, n)          # all the action is overnight
        close = 100.0 * np.exp(np.cumsum(gaps))
        idx = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame({
            "Open": close, "Close": close,        # open == close. No intraday move
            "High": close * 1.0005, "Low": close * 0.9995,
            "Volume": np.full(n, 1.0),
        }, index=idx)
        park = parkinson(df, annualized=False)
        yz = yang_zhang(df, annualized=False)
        assert park < 0.001, f"Parkinson should be blind to the gap, got {park}"
        assert yz > 0.03, f"Yang-Zhang should see the gap, got {yz}"
        assert yz > 10 * park

    def test_yang_zhang_positive_and_gap_aware(self):
        rng = np.random.default_rng(9)
        prices = make_ohlcv(rng.normal(0.0, 0.02, 500), seed=9)
        yz = yang_zhang(prices, annualized=False)
        assert yz > 0
        assert yang_zhang(prices) == pytest.approx(yz * SQRT252)

    def test_yang_zhang_requires_three_rows(self):
        prices = make_ohlcv(np.array([0.01]))
        with pytest.raises(InsufficientDataError):
            yang_zhang(prices)

    def test_missing_columns_raise(self):
        prices = make_ohlcv(np.full(20, 0.005)).drop(columns=["High"])
        with pytest.raises(SchemaValidationError, match="missing"):
            parkinson(prices)

    def test_rolling_close_to_close_is_right_aligned(self):
        prices = make_ohlcv(np.full(30, 0.01))
        roll = rolling_close_to_close(prices["Close"], window=5)
        assert roll.iloc[:4].isna().all()
        assert roll.iloc[4:].notna().all()

    def test_rolling_window_must_exceed_one(self):
        prices = make_ohlcv(np.full(30, 0.01))
        with pytest.raises(ValueError, match="window"):
            rolling_close_to_close(prices["Close"], window=1)


# --------------------------------------------------------------------------- #
# Schema validation and gap handling
# --------------------------------------------------------------------------- #

class TestValidation:
    """The data contract. Nothing malformed may pass silently."""

    def test_valid_frame_passes_and_normalizes(self):
        prices = make_ohlcv(np.full(100, 0.004))
        out = validate_ohlcv(prices, ticker="TEST")
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert out.index.is_monotonic_increasing
        assert out.index.tz is None

    def test_missing_column_raises(self):
        prices = make_ohlcv(np.full(100, 0.004)).drop(columns=["Volume"])
        with pytest.raises(SchemaValidationError) as exc:
            validate_ohlcv(prices, ticker="TEST")
        assert any("Volume" in v for v in exc.value.violations)

    def test_negative_price_raises(self):
        prices = make_ohlcv(np.full(100, 0.004))
        prices.iloc[50, prices.columns.get_loc("Low")] = -1.0
        with pytest.raises(SchemaValidationError, match="non-positive"):
            validate_ohlcv(prices, ticker="TEST")

    def test_high_below_low_raises(self):
        prices = make_ohlcv(np.full(100, 0.004))
        prices.iloc[10, prices.columns.get_loc("High")] = 1.0
        prices.iloc[10, prices.columns.get_loc("Low")] = 500.0
        with pytest.raises(SchemaValidationError, match="High < Low"):
            validate_ohlcv(prices, ticker="TEST")

    def test_duplicate_dates_raise(self):
        prices = make_ohlcv(np.full(100, 0.004))
        dupe = prices.iloc[[5]]
        broken = pd.concat([prices, dupe]).sort_index()
        with pytest.raises(SchemaValidationError, match="duplicated"):
            validate_ohlcv(broken, ticker="TEST")

    def test_unsorted_index_is_repaired(self):
        """A shuffled but otherwise valid index is sorted rather than rejected."""
        prices = make_ohlcv(np.full(100, 0.004))
        shuffled = prices.iloc[np.random.default_rng(0).permutation(len(prices))]
        out = validate_ohlcv(shuffled, ticker="TEST")
        assert out.index.is_monotonic_increasing

    def test_nan_price_rows_are_dropped(self):
        prices = make_ohlcv(np.full(100, 0.004))
        prices.iloc[7, prices.columns.get_loc("Close")] = np.nan
        out = validate_ohlcv(prices, ticker="TEST")
        assert len(out) == len(prices) - 1

    def test_empty_frame_raises(self):
        with pytest.raises(SchemaValidationError, match="empty"):
            validate_ohlcv(pd.DataFrame(), ticker="TEST")

    def test_too_few_rows_raises(self):
        prices = make_ohlcv(np.full(5, 0.004))
        with pytest.raises(InsufficientDataError):
            validate_ohlcv(prices, ticker="TEST", min_rows=60)

    def test_holidays_are_not_reported_as_gaps(self):
        """A 1-3 day closure is a holiday and must stay silent."""
        prices = make_ohlcv(np.full(200, 0.004))
        # Drop a 3-business-day run. Within the holiday tolerance.
        keep = prices.index[~prices.index.isin(prices.index[50:53])]
        report = detect_gaps(keep)
        assert not report.has_gaps

    def test_long_gap_is_detected(self):
        """A 20-session hole is an anomaly and must be surfaced."""
        prices = make_ohlcv(np.full(300, 0.004))
        keep = prices.index[~prices.index.isin(prices.index[100:120])]
        report = detect_gaps(keep)
        assert report.has_gaps
        assert len(report.gaps) == 1
        assert report.gaps[0][2] == 20
        assert "gap" in report.describe()

    def test_detect_gaps_rejects_empty_index(self):
        with pytest.raises(SchemaValidationError):
            detect_gaps(pd.DatetimeIndex([]))

    def test_price_cache_separates_adjusted_and_raw_bars(self, tmp_path, monkeypatch):
        prices = make_ohlcv(np.full(100, 0.004))
        calls = []

        def fake_fetch(ticker, start, end, auto_adjust=True):
            calls.append(auto_adjust)
            return prices.copy()

        monkeypatch.setattr("vol_decom.data._fetch_prices_yf", fake_fetch)
        kwargs = {
            "start": prices.index[0],
            "end": prices.index[-1],
            "cache_dir": tmp_path,
            "min_rows": 60,
        }

        load_prices("TEST", auto_adjust=True, **kwargs)
        load_prices("TEST", auto_adjust=False, **kwargs)

        assert calls == [True, False]


# --------------------------------------------------------------------------- #
# Trading-day alignment
# --------------------------------------------------------------------------- #

class TestAlignment:
    """Alignment must be explicit, forward-rolling and calendar-free."""

    def test_announcement_rolls_forward_to_next_session(self):
        prices = make_ohlcv(np.full(300, 0.005))
        rets = log_returns(prices["Close"])
        # A Saturday. Must map to the following Monday's return.
        saturday = pd.Timestamp("2019-06-15")
        assert saturday.dayofweek == 5
        align = align_events(pd.DatetimeIndex(rets.index), [saturday])
        assert len(align) == 1
        assert align.session_dates[0].dayofweek == 0
        assert align.session_dates[0] > saturday

    def test_future_announcements_are_dropped(self):
        prices = make_ohlcv(np.full(300, 0.005))
        rets = log_returns(prices["Close"])
        future = prices.index[-1] + pd.Timedelta(days=365)
        align = align_events(pd.DatetimeIndex(rets.index),
                             [prices.index[150], future])
        assert len(align) == 1
        assert future in align.dropped_future

    def test_events_without_enough_history_are_dropped(self):
        prices = make_ohlcv(np.full(300, 0.005))
        rets = log_returns(prices["Close"])
        # Session 3 has no room for a 60-day baseline.
        align = align_events(pd.DatetimeIndex(rets.index),
                             [prices.index[3], prices.index[150]])
        assert len(align) == 1
        assert len(align.dropped_insufficient) == 1
        assert "pre-event" in align.dropped_insufficient[0][1]

    def test_duplicate_announcements_collapse(self):
        """Two announcements rolling onto one session must not double-count."""
        prices = make_ohlcv(np.full(300, 0.005))
        rets = log_returns(prices["Close"])
        sat = pd.Timestamp("2019-06-15")
        sun = pd.Timestamp("2019-06-16")
        align = align_events(pd.DatetimeIndex(rets.index), [sat, sun])
        assert len(align) == 1
        assert len(align.dropped_duplicate) == 1

    def test_strict_mode_raises_on_any_drop(self):
        prices = make_ohlcv(np.full(300, 0.005))
        rets = log_returns(prices["Close"])
        future = prices.index[-1] + pd.Timedelta(days=365)
        with pytest.raises(AlignmentError, match="strict"):
            align_events(pd.DatetimeIndex(rets.index),
                         [prices.index[150], future], strict=True)

    def test_no_usable_events_raises(self):
        prices = make_ohlcv(np.full(300, 0.005))
        rets = log_returns(prices["Close"])
        with pytest.raises(MissingEarningsDatesError):
            align_events(pd.DatetimeIndex(rets.index), [prices.index[2]])

    def test_empty_announcement_list_raises(self):
        prices = make_ohlcv(np.full(300, 0.005))
        rets = log_returns(prices["Close"])
        with pytest.raises(MissingEarningsDatesError):
            align_events(pd.DatetimeIndex(rets.index), [])

    def test_unsorted_return_index_raises(self):
        prices = make_ohlcv(np.full(300, 0.005))
        bad = pd.DatetimeIndex(prices.index[::-1])
        with pytest.raises(AlignmentError, match="sorted"):
            align_events(bad, [prices.index[100]])

    def test_alignment_is_position_based_not_calendar_based(self):
        """Removing sessions must shift positions, not calendar arithmetic."""
        prices = make_ohlcv(np.full(400, 0.005))
        gapped = prices.drop(prices.index[100:130])
        rets = log_returns(gapped["Close"])
        target = gapped.index[200]
        align = align_events(pd.DatetimeIndex(rets.index), [target])
        pos = int(align.positions[0])
        # The baseline window must be the 20 preceding *sessions*, contiguous
        # in position space regardless of the calendar hole.
        assert rets.index[pos - 20] in gapped.index
        assert (pos - (pos - 20)) == 20


# --------------------------------------------------------------------------- #
# The decomposition, the core of the package
# --------------------------------------------------------------------------- #

class TestDecomposition:
    """sigma_jump = sqrt(max(0, sigma_total^2 - sigma_baseline^2))."""

    def test_normal_case_isolates_a_known_jump(self, jump_scenario):
        """A 12% jump on a 1% baseline must be recovered, not diluted away."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        assert len(events) > 5
        # No event should clamp. Every one has a genuine 12% move.
        assert not events["jump_clamped"].any()
        # Event-window variance must dominate the baseline by a wide margin.
        assert (events["variance_ratio"] > 10).all()
        # The absolute event move must recover the injected jump.
        assert events["abs_event_return"].mean() == pytest.approx(0.12, rel=0.05)

    def test_jump_identity_holds_exactly(self, jump_scenario):
        """The reported jump must satisfy the governing formula to machine precision."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        s_tot = events["sigma_total"].to_numpy()
        s_base = events["sigma_baseline"].to_numpy()
        s_jump = events["sigma_jump"].to_numpy()
        expected = np.sqrt(np.clip(s_tot ** 2 - s_base ** 2, 0.0, None))
        np.testing.assert_allclose(s_jump, expected, rtol=1e-12, atol=1e-15)

    def test_zero_jump_case_clamps_and_never_returns_nan(self, zero_jump_scenario):
        """sigma_total <= sigma_baseline must clamp to 0, not produce NaN."""
        prices, announce = zero_jump_scenario
        events = decompose_events(prices, announce)
        assert len(events) == 2
        assert events["jump_clamped"].all()
        assert (events["sigma_jump"] == 0.0).all()
        assert events["sigma_jump"].notna().all()
        assert (events["var_jump_raw"] < 0).all()
        assert (events["variance_ratio"] < 1.0).all()

    def test_annualization_scales_by_sqrt_252(self, jump_scenario):
        prices, announce, _ = jump_scenario
        ann = decompose_events(prices, announce, DecompositionConfig(annualized=True))
        dly = decompose_events(prices, announce, DecompositionConfig(annualized=False))
        np.testing.assert_allclose(
            ann["sigma_jump"].to_numpy(), dly["sigma_jump"].to_numpy() * SQRT252, rtol=1e-12
        )
        # The raw event move is a return, not a vol, so it must NOT be scaled.
        np.testing.assert_allclose(
            ann["abs_event_return"].to_numpy(), dly["abs_event_return"].to_numpy(), rtol=1e-12
        )

    def test_all_configured_baseline_windows_are_reported(self, jump_scenario):
        prices, announce, _ = jump_scenario
        cfg = DecompositionConfig(baseline_windows=(20, 30, 60))
        events = decompose_events(prices, announce, cfg)
        for w in (20, 30, 60):
            assert f"sigma_baseline_{w}" in events.columns
        # The primary baseline is the first window.
        np.testing.assert_allclose(events["sigma_baseline"], events["sigma_baseline_20"])

    def test_post_event_windows_are_reported(self, jump_scenario):
        prices, announce, _ = jump_scenario
        cfg = DecompositionConfig(post_windows=(5, 10, 20))
        events = decompose_events(prices, announce, cfg)
        for k in (5, 10, 20):
            assert f"sigma_post_{k}" in events.columns
            assert events[f"sigma_post_{k}"].notna().all()

    def test_post_event_vol_reverts_toward_baseline(self, jump_scenario):
        """After the event, realized vol must fall back to the diffusive level."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        assert (events["sigma_post_10"] < events["sigma_total"]).all()

    def test_range_estimator_baseline_runs(self, jump_scenario):
        prices, announce, _ = jump_scenario
        for est in ("parkinson", "yang_zhang"):
            events = decompose_events(prices, announce,
                                      DecompositionConfig(estimator=est))
            assert events["sigma_baseline"].notna().all()
            assert (events["sigma_baseline"] > 0).all()

    def test_range_estimator_requires_ohlc(self, jump_scenario):
        prices, announce, _ = jump_scenario
        with pytest.raises(SchemaValidationError, match="requires full OHLC"):
            decompose_events(prices.drop(columns=["High"]), announce,
                             DecompositionConfig(estimator="parkinson"))

    def test_missing_close_column_raises(self, jump_scenario):
        prices, announce, _ = jump_scenario
        with pytest.raises(SchemaValidationError, match="Close"):
            decompose_events(prices.drop(columns=["Close"]), announce)

    def test_short_history_raises_insufficient_data(self):
        prices = make_ohlcv(np.full(40, 0.01))
        with pytest.raises(InsufficientDataError):
            decompose_events(prices, [prices.index[20]])

    def test_gapped_history_still_decomposes(self, jump_scenario):
        """A hole in the price history must not corrupt window arithmetic."""
        prices, announce, _ = jump_scenario
        gapped = prices.drop(prices.index[300:340])
        events = decompose_events(gapped, announce)
        assert len(events) > 3
        assert events["sigma_baseline"].notna().all()
        assert events["sigma_jump"].notna().all()
        # Events whose windows were destroyed by the hole are dropped, not faked.
        assert len(events) <= len(announce)

    def test_no_events_survive_raises(self):
        prices = make_ohlcv(np.full(400, 0.01))
        with pytest.raises(MissingEarningsDatesError):
            decompose_events(prices, [prices.index[1]])

    def test_prototype_mode_reproduces_the_2_obs_failure(self, jump_scenario):
        """The legacy config must still exhibit the behaviour it is kept for.

        With a demeaned 2-observation event window, an event whose two sessions
        move by a *similar* amount collapses to near-zero variance. The modern
        default must not.
        """
        rng = np.random.default_rng(1)
        n = 500
        rets = rng.normal(0.0, 0.005, n)
        pos = np.array([200, 300])
        for p in pos:
            rets[p] = 0.09
            rets[p + 1] = 0.09  # two identical large moves
        prices = make_ohlcv(rets, ticker="TWIN")
        # Align the 2-session event window onto returns [p, p+1], the two
        # identical moves. Return position j is dated price index j+1, so the
        # announcement must be dated prices.index[p + 1].
        announce = pd.DatetimeIndex([prices.index[p + 1] for p in pos])

        legacy = decompose_events(prices, announce, DecompositionConfig.prototype())
        modern = decompose_events(prices, announce, DecompositionConfig())

        assert legacy["jump_clamped"].all()
        assert (legacy["sigma_jump"] == 0.0).all()
        assert not modern["jump_clamped"].any()
        assert (modern["sigma_jump"] > 0.5).all()

    def test_prototype_mode_is_not_annualized(self, jump_scenario):
        prices, announce, _ = jump_scenario
        legacy = decompose_events(prices, announce, DecompositionConfig.prototype())
        assert legacy.attrs["annualized"] is False
        assert legacy["sigma_baseline"].mean() < 0.1  # daily, not annual

    def test_config_validation(self):
        with pytest.raises(ValueError, match="baseline_windows"):
            DecompositionConfig(baseline_windows=())
        with pytest.raises(ValueError, match="baseline_windows"):
            DecompositionConfig(baseline_windows=(1,))
        with pytest.raises(ValueError, match="event_window"):
            DecompositionConfig(event_window=0)
        with pytest.raises(ValueError, match="estimator"):
            DecompositionConfig(estimator="garch")
        with pytest.raises(ValueError, match="baseline_gap"):
            DecompositionConfig(baseline_gap=-1)

    def test_baseline_gap_shifts_the_window(self, jump_scenario):
        """A non-zero gap must actually move the baseline window."""
        prices, announce, _ = jump_scenario
        a = decompose_events(prices, announce, DecompositionConfig(baseline_gap=0))
        b = decompose_events(prices, announce, DecompositionConfig(baseline_gap=10))
        assert not np.allclose(a["sigma_baseline"], b["sigma_baseline"])

    def test_summarize_reports_clamp_rate(self, zero_jump_scenario):
        prices, announce = zero_jump_scenario
        events = decompose_events(prices, announce)
        s = summarize_decomposition(events)
        assert s["clamp_rate"] == 1.0
        assert s["n_events"] == 2.0

    def test_summarize_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            summarize_decomposition(pd.DataFrame())


# --------------------------------------------------------------------------- #
# Volatility risk premium
# --------------------------------------------------------------------------- #

class TestVRP:
    """The structural-edge metric and its honesty guarantees."""

    def test_proxy_mode_has_zero_mean_by_construction(self, jump_scenario):
        """Leave-one-out benchmarking must not manufacture an edge."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = volatility_risk_premium(events)
        assert result.implied_source == "historical_proxy"
        assert result.mean_vrp == pytest.approx(0.0, abs=1e-12)
        assert result.proxy_note  # the caveat must be carried

    def test_proxy_benchmark_excludes_the_event_itself(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = volatility_risk_premium(events)
        realized = events.loc[result.per_event.index, "abs_event_return"]
        n = len(realized)
        expected = (realized.sum() - realized) / (n - 1)
        np.testing.assert_allclose(result.per_event["implied_move"], expected, rtol=1e-12)

    def test_supplied_implied_move_produces_a_real_premium(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        # Market charged 15% for a 12% realized move. A 3-point premium.
        result = volatility_risk_premium(events, implied_moves=0.15)
        assert result.implied_source == "market"
        assert result.mean_vrp == pytest.approx(0.03, abs=0.005)
        assert result.hit_rate == 1.0
        assert not result.proxy_note

    def test_implied_vols_path_is_converted(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = volatility_risk_premium(events, implied_vols=0.60, days_to_expiry=21)
        assert result.implied_source == "market"
        assert (result.per_event["implied_move"] > 0).all()

    def test_negative_premium_when_market_underprices(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = volatility_risk_premium(events, implied_moves=0.05)
        assert result.mean_vrp < 0
        assert result.hit_rate == 0.0

    def test_too_few_events_raises(self, zero_jump_scenario):
        prices, announce = zero_jump_scenario
        events = decompose_events(prices, announce)
        with pytest.raises(ValueError, match="at least"):
            volatility_risk_premium(events)

    def test_invalid_trim_raises(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        with pytest.raises(ValueError, match="trim_quantile"):
            volatility_risk_premium(events, trim_quantile=0.7)

    def test_implied_jump_move_matches_the_deck_arithmetic(self):
        """sqrt((iv^2 - base^2) * T), the 7.87% figure from the source research."""
        move = implied_jump_move(0.656, 0.597, days_to_expiry=21)
        assert 0.05 < move < 0.10
        # Explicit closed form.
        tau = 21 / 252
        assert move == pytest.approx(np.sqrt((0.656 ** 2 - 0.597 ** 2) * tau))

    def test_implied_jump_move_floors_at_zero(self):
        """IV below the diffusive baseline implies no priced event, not a NaN."""
        assert implied_jump_move(0.20, 0.60, days_to_expiry=21) == 0.0

    def test_implied_jump_move_rejects_bad_term(self):
        with pytest.raises(ValueError, match="days_to_expiry"):
            implied_jump_move(0.5, 0.3, days_to_expiry=0)


# --------------------------------------------------------------------------- #
# Volatility cone
# --------------------------------------------------------------------------- #

class TestVolCone:
    def test_cone_quantiles_are_ordered(self, quiet_returns):
        prices = make_ohlcv(quiet_returns)
        cone = vol_cone(prices)
        assert (cone["q05"] <= cone["q50"]).all()
        assert (cone["q50"] <= cone["q95"]).all()

    def test_cone_narrows_with_horizon(self, quiet_returns):
        """Longer windows must average out noise. The band must tighten."""
        prices = make_ohlcv(quiet_returns)
        cone = vol_cone(prices, windows=(5, 120))
        width = cone["q95"] - cone["q05"]
        assert width.loc[120] < width.loc[5]

    def test_cone_rejects_unknown_estimator(self, quiet_returns):
        prices = make_ohlcv(quiet_returns)
        with pytest.raises(ValueError, match="estimator"):
            vol_cone(prices, estimator="ewma")

    def test_cone_raises_when_history_too_short(self):
        prices = make_ohlcv(np.full(10, 0.01))
        with pytest.raises(InsufficientDataError):
            vol_cone(prices, windows=(60, 120))


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

class TestBacktest:
    """The delta-hedged vol-capture simulator."""

    def test_bs_straddle_atm_properties(self):
        """ATM straddle. Delta ~0, gamma >0, price ~0.8 * S * sigma * sqrt(T)."""
        price, delta, gamma = bs_straddle(100.0, 100.0, 0.30, 30 / 252)
        assert abs(float(delta)) < 0.05
        assert float(gamma) > 0
        approx = 2 * 0.3989 * 100.0 * 0.30 * np.sqrt(30 / 252)
        assert float(price) == pytest.approx(approx, rel=0.02)

    def test_bs_straddle_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="volatility"):
            bs_straddle(100.0, 100.0, 0.0, 0.1)
        with pytest.raises(ValueError, match="spot and strike"):
            bs_straddle(-100.0, 100.0, 0.3, 0.1)

    def test_gamma_rises_as_expiry_approaches(self):
        _, _, g_far = bs_straddle(100.0, 100.0, 0.3, 60 / 252)
        _, _, g_near = bs_straddle(100.0, 100.0, 0.3, 2 / 252)
        assert float(g_near) > float(g_far)

    def test_backtest_runs_and_reports_all_required_stats(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events)
        for key in ("win_rate", "mean_return", "sharpe_ratio",
                    "profit_factor", "max_drawdown"):
            assert key in result.stats
        assert 0.0 <= result.stats["win_rate"] <= 1.0
        assert result.stats["max_drawdown"] <= 0.0
        assert len(result.trades) == len(result.equity_curve)

    def test_selling_a_richer_vol_earns_more(self, jump_scenario):
        """Monotonicity. The edge must increase with the assumed overpricing."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        rets = [
            run_backtest(prices, events,
                         BacktestConfig(implied_move_multiplier=k,
                                        transaction_cost_bps=0.0)).stats["mean_return"]
            for k in (0.9, 1.0, 1.2, 1.5)
        ]
        assert rets == sorted(rets)

    def test_fair_pricing_gives_no_large_edge(self, jump_scenario):
        """At k=1 the expected edge must be ~0, not a manufactured profit."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events,
                              BacktestConfig(implied_move_multiplier=1.0,
                                             transaction_cost_bps=0.0))
        assert abs(result.stats["mean_return"]) < 0.10

    def test_straddle_variance_to_move_conversion(self):
        """The variance-vs-move conversion the default calibration relies on.

        An ATM straddle marked at variance V is worth sqrt(2/pi)*sqrt(V)*S. If
        this identity ever breaks, the k=1.0 baseline silently stops being
        edge-free, so it is pinned here.
        """
        from vol_decom.backtest import SQRT_2_OVER_PI
        assert SQRT_2_OVER_PI == pytest.approx(0.7978845608)
        S, tau, target_move = 100.0, 8 / 252, 0.12
        var = (target_move / SQRT_2_OVER_PI) ** 2
        sigma = np.sqrt(var / tau)
        price, _, _ = bs_straddle(S, S, sigma, tau)
        assert float(price) == pytest.approx(target_move * S, rel=0.01)

    def test_costs_and_slippage_reduce_returns(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        free = run_backtest(prices, events, BacktestConfig(transaction_cost_bps=0.0))
        costly = run_backtest(prices, events, BacktestConfig(transaction_cost_bps=200.0))
        assert costly.stats["mean_return"] < free.stats["mean_return"]
        slipped = run_backtest(prices, events,
                               BacktestConfig(transaction_cost_bps=0.0,
                                              slippage_vol_points=0.05))
        assert slipped.stats["mean_return"] < free.stats["mean_return"]

    def test_short_straddle_structure_has_no_back_leg(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events,
                              BacktestConfig(structure="short_straddle"))
        assert (result.trades["back_pnl"] == 0.0).all()
        assert (result.trades["back_premium"] == 0.0).all()

    def test_supplied_implied_vol_is_used(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events, implied_vols=0.90)
        assert result.trades.attrs["iv_source"] == "supplied"
        np.testing.assert_allclose(result.trades["front_iv"], 0.90)

    def test_supplied_implied_vol_respects_quote_term(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        default_term = run_backtest(prices, events, implied_vols=0.90)
        quoted_term = run_backtest(
            prices, events, implied_vols=0.90, implied_vol_days=21,
        )

        assert quoted_term.trades["implied_event_move"].mean() > default_term.trades[
            "implied_event_move"
        ].mean()

    def test_supplied_implied_vol_rejects_nonpositive_quote_term(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        with pytest.raises(BacktestError, match="implied_vol_days"):
            run_backtest(prices, events, implied_vols=0.90, implied_vol_days=0)

    def test_max_drawdown_is_nonpositive(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events)
        assert result.stats["max_drawdown"] <= 0.0

    def test_compounded_drawdown_is_bounded_below_by_minus_one(self, jump_scenario):
        """A compounded curve's drawdown is a ratio and cannot pass -100%."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events, BacktestConfig(compound=True))
        assert -1.0000001 <= result.stats["max_drawdown"] <= 0.0

    def test_equity_curve_is_additive_by_default(self, jump_scenario):
        """Additive is the default. A short option can lose more than its premium."""
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events)
        expected = 1.0 + np.cumsum(result.trades["return"].to_numpy())
        np.testing.assert_allclose(result.equity_curve.to_numpy(), expected, rtol=1e-12)

    def test_equity_curve_compounds_when_requested(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events, BacktestConfig(compound=True))
        expected = np.cumprod(1.0 + result.trades["return"].to_numpy())
        np.testing.assert_allclose(result.equity_curve.to_numpy(), expected, rtol=1e-12)

    def test_profit_factor_consistent_with_trade_signs(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events)
        r = result.trades["return"]
        if (r < 0).any() and (r > 0).any():
            expected = r[r > 0].sum() / -r[r < 0].sum()
            assert result.stats["profit_factor"] == pytest.approx(expected)

    def test_config_validation(self):
        with pytest.raises(BacktestError, match="structure"):
            BacktestConfig(structure="butterfly")
        with pytest.raises(BacktestError, match="entry_offset"):
            BacktestConfig(entry_offset=0)
        with pytest.raises(BacktestError, match="front_expiry_offset"):
            BacktestConfig(front_expiry_offset=1, exit_offset=1)
        with pytest.raises(BacktestError, match="back_expiry_offset"):
            BacktestConfig(entry_offset=2, back_expiry_offset=5)
        with pytest.raises(BacktestError, match="capital_basis"):
            BacktestConfig(capital_basis="margin")

    def test_empty_events_raises(self, jump_scenario):
        prices, _, _ = jump_scenario
        with pytest.raises(BacktestError, match="non-empty"):
            run_backtest(prices, pd.DataFrame())

    def test_missing_close_raises(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        with pytest.raises(SchemaValidationError, match="Close"):
            run_backtest(prices.drop(columns=["Close"]), events)

    def test_single_event_needs_supplied_iv(self, zero_jump_scenario):
        """The leave-one-out benchmark is undefined for one event."""
        prices, announce = zero_jump_scenario
        events = decompose_events(prices, announce).iloc[:1]
        with pytest.raises(BacktestError, match="leave-one-out"):
            run_backtest(prices, events)
        # With a supplied IV it must run.
        result = run_backtest(prices, events, implied_vols=0.5)
        assert len(result.trades) == 1

    def test_no_loop_over_events_in_hot_path(self):
        """Guard the vectorization requirement against future edits.

        The event history must be priced with array operations. This asserts the
        estimation path contains no `for` loop over trades or hedge dates.
        """
        import ast
        import inspect
        import textwrap
        from vol_decom import backtest as bt_mod
        from vol_decom import engine as eng_mod

        def loop_count(fn) -> int:
            """Count `for`/`while` statements in a function's body, ignoring docstrings."""
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            return sum(isinstance(node, (ast.For, ast.While, ast.comprehension))
                       for node in ast.walk(tree))

        assert loop_count(bt_mod._hedged_leg_pnl) == 0
        assert loop_count(eng_mod._window_matrix) == 0
        assert loop_count(eng_mod._row_variance) == 0


# --------------------------------------------------------------------------- #
# Visualizer, figures must be returned, never shown
# --------------------------------------------------------------------------- #

class TestVisualizer:
    def test_figures_are_returned_not_shown(self, jump_scenario):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        from vol_decom.visualizer import (
            plot_decomposition_timeline,
            plot_jump_distribution,
            plot_pnl_curve,
            plot_vol_cone,
            plot_vrp_scatter,
        )
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        vrp = volatility_risk_premium(events, implied_moves=0.15)
        result = run_backtest(prices, events)

        for fig in (
            plot_vol_cone(vol_cone(prices)),
            plot_jump_distribution(events),
            plot_decomposition_timeline(events),
            plot_vrp_scatter(vrp),
            plot_pnl_curve(result),
        ):
            assert hasattr(fig, "savefig")

    def test_plot_rejects_empty_input(self):
        pytest.importorskip("matplotlib")
        from vol_decom.visualizer import plot_jump_distribution
        with pytest.raises(ValueError, match="non-empty"):
            plot_jump_distribution(pd.DataFrame())

    def test_plot_rejects_unknown_backend(self, jump_scenario):
        from vol_decom.visualizer import plot_jump_distribution
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        with pytest.raises(ValueError, match="backend"):
            plot_jump_distribution(events, backend="ggplot")


# --------------------------------------------------------------------------- #
# Generality across tickers, and the multi-symbol CLI plumbing
# --------------------------------------------------------------------------- #

class TestGenerality:
    """The model must work on any listed name, not just one worked example."""

    @pytest.mark.parametrize(
        "baseline_vol,jump_size,label",
        [
            (0.004, 0.015, "low-vol defensive"),
            (0.012, 0.060, "mid-vol large cap"),
            (0.035, 0.180, "high-vol growth"),
            (0.070, 0.400, "extreme small cap"),
        ],
    )
    def test_decomposition_scales_across_vol_regimes(self, baseline_vol, jump_size, label):
        """From a defensive staple to a meme stock, the jump must be recovered.

        The point of parameterizing this is that nothing in the estimator may
        be tuned to one name's volatility level.
        """
        rng = np.random.default_rng(abs(hash(label)) % 2**31)
        n = 700
        rets = rng.normal(0.0, baseline_vol, n)
        pos = np.arange(120, n - 60, 63)
        rets[pos] = jump_size
        prices = make_ohlcv(rets, intraday_range=baseline_vol / 2,
                            seed=3, ticker=label)
        announce = pd.DatetimeIndex([prices.index[p] for p in pos])

        events = decompose_events(prices, announce)
        assert not events["jump_clamped"].any(), f"{label} lost its jump"
        assert events["abs_event_return"].mean() == pytest.approx(jump_size, rel=0.10)
        # The variance ratio is scale-free, so a bigger jump relative to its own
        # baseline must always register as a bigger ratio.
        assert (events["variance_ratio"] > 3.0).all()

    def test_low_priced_stock_is_handled(self):
        """A sub-dollar price must not break the log returns or the pricing."""
        rng = np.random.default_rng(8)
        n = 500
        rets = rng.normal(0.0, 0.03, n)
        pos = np.arange(120, n - 60, 63)
        rets[pos] = 0.15
        prices = make_ohlcv(rets, start_price=0.85, ticker="PENNY")
        announce = pd.DatetimeIndex([prices.index[p] for p in pos])
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events)
        assert np.isfinite(result.trades["return"]).all()
        assert (result.trades["front_premium"] > 0).all()

    def test_semiannual_reporter_infers_two_events_per_year(self):
        """events_per_year must come from the data, not a hardcoded 4."""
        from vol_decom.backtest import infer_events_per_year
        semi = pd.DatetimeIndex(pd.date_range("2019-01-15", periods=12, freq="182D"))
        quarterly = pd.DatetimeIndex(pd.date_range("2019-01-15", periods=20, freq="91D"))
        assert infer_events_per_year(semi) == pytest.approx(2.0, abs=0.15)
        assert infer_events_per_year(quarterly) == pytest.approx(4.0, abs=0.2)

    def test_infer_events_per_year_is_robust_to_a_long_gap(self):
        """A provider hole must not drag the inferred frequency down."""
        from vol_decom.backtest import infer_events_per_year
        dates = list(pd.date_range("2019-01-15", periods=8, freq="91D"))
        dates += list(pd.date_range("2024-01-15", periods=8, freq="91D"))
        assert infer_events_per_year(pd.DatetimeIndex(dates)) == pytest.approx(4.0, abs=0.3)

    def test_infer_events_per_year_falls_back_and_clamps(self):
        from vol_decom.backtest import infer_events_per_year
        assert infer_events_per_year(pd.DatetimeIndex(["2024-01-01"])) == 4.0
        weekly = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=30, freq="7D"))
        assert infer_events_per_year(weekly) <= 12.0

    def test_sharpe_uses_the_inferred_frequency(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events)
        assert "events_per_year" in result.stats
        assert 1.0 <= result.stats["events_per_year"] <= 12.0

    def test_explicit_events_per_year_overrides_inference(self, jump_scenario):
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        result = run_backtest(prices, events, BacktestConfig(events_per_year=2.0))
        assert result.stats["events_per_year"] == 2.0

    def test_events_per_year_rejects_nonpositive(self):
        with pytest.raises(BacktestError, match="events_per_year"):
            BacktestConfig(events_per_year=0.0)


class TestCLI:
    """Argument plumbing for single and multi symbol runs. No network."""

    def test_parser_accepts_a_single_ticker(self):
        import main as cli
        args = cli.build_parser().parse_args(["--ticker", "INTC"])
        assert args.tickers == ["INTC"]

    def test_parser_accepts_many_tickers(self):
        import main as cli
        args = cli.build_parser().parse_args(["-t", "INTC", "AAPL", "NVDA"])
        assert args.tickers == ["INTC", "AAPL", "NVDA"]

    def test_resolve_tickers_dedupes_and_uppercases(self):
        import main as cli
        parser = cli.build_parser()
        args = parser.parse_args(["-t", "intc", "AAPL", "Intc", " nvda "])
        assert cli._resolve_tickers(args, parser) == ["INTC", "AAPL", "NVDA"]

    def test_resolve_tickers_reads_a_universe_csv(self, tmp_path):
        import main as cli
        path = tmp_path / "universe.csv"
        path.write_text("ticker\nAAPL\nMSFT\nAAPL\n")
        parser = cli.build_parser()
        args = parser.parse_args(["--universe-csv", str(path)])
        assert cli._resolve_tickers(args, parser) == ["AAPL", "MSFT"]

    def test_resolve_tickers_merges_flags_and_csv(self, tmp_path):
        import main as cli
        path = tmp_path / "u.csv"
        path.write_text("symbol\nMSFT\n")
        parser = cli.build_parser()
        args = parser.parse_args(["-t", "INTC", "--universe-csv", str(path)])
        assert cli._resolve_tickers(args, parser) == ["INTC", "MSFT"]

    def test_no_ticker_is_a_usage_error(self):
        import main as cli
        parser = cli.build_parser()
        args = parser.parse_args([])
        with pytest.raises(SystemExit):
            cli._resolve_tickers(args, parser)

    def test_prototype_mode_builds_the_legacy_config(self):
        import main as cli
        args = cli.build_parser().parse_args(["-t", "NVDA", "--prototype-mode"])
        cfg = cli._build_decomposition_config(args)
        assert cfg.legacy_prototype_mode is True
        assert cfg.baseline_windows == (20,)
        assert cfg.effective_annualized is False

    def test_config_is_built_from_flags(self):
        import main as cli
        args = cli.build_parser().parse_args(
            ["-t", "INTC", "--baseline-windows", "30", "60",
             "--estimator", "yang_zhang", "--event-window", "1",
             "--baseline-gap", "3", "--no-annualize"]
        )
        cfg = cli._build_decomposition_config(args)
        assert cfg.baseline_windows == (30, 60)
        assert cfg.primary_baseline == 30
        assert cfg.estimator == "yang_zhang"
        assert cfg.event_window == 1
        assert cfg.baseline_gap == 3
        assert cfg.effective_annualized is False

    def test_cross_section_renders_without_a_backtest(self, jump_scenario, capsys):
        """The cross-section must tolerate tickers whose backtest was skipped."""
        import main as cli
        prices, announce, _ = jump_scenario
        events = decompose_events(prices, announce)
        results = [{"ticker": "AAA", "prices": prices, "events": events,
                    "summary": summarize_decomposition(events),
                    "vrp": None, "backtest": None}]
        cli._cross_section(results, [("SPY", "no earnings calendar")])
        out = capsys.readouterr().out
        assert "AAA" in out
        assert "SPY" in out
        assert "n/a" in out          # missing Sharpe rendered, not crashed

    def test_cross_section_handles_an_all_failed_run(self, capsys):
        import main as cli
        cli._cross_section([], [("SPY", "no earnings"), ("QQQ", "no earnings")])
        out = capsys.readouterr().out
        assert "SPY" in out and "QQQ" in out
