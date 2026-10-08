# Methodology and usage

## Retrospective decomposition

For each aligned event, the default estimates per-session variance as the mean
squared close-to-close log return. The baseline precedes the event window, with an
optional gap. Excess variance is `max(0, v_event - v_baseline)` and displayed
volatility is its square root, optionally multiplied by `sqrt(252)`. Integrated
window variance would be the daily variance rate times the window length.
The annualized excess volatility is not an expected absolute earnings move.

The variance ratio is `v_event/v_baseline`, undefined for a zero baseline.
Negative estimates are retained as `var_jump_raw` and flagged `jump_clamped`.
A high clamp rate signals estimator noise; changing windows changes the estimate.
The default zero-mean estimator differs from the demeaned population variance in
`DecompositionConfig.prototype()`. The preserved prototype is a historical
reference; parity comparisons require identical inputs, ordering and tolerances.

`event_return` selects the signed largest absolute session log return.
`event_window_log_return` is the sum across the event window. These are different
observations. The post-event diagnostic starts one session after the first event
session and may overlap a multi-session event window; it is descriptive.

Parkinson and Yang-Zhang baselines require consistent OHLC adjustments. Dates
align forward to the next observed session. Apply the announcement offset for
known after-close events. Missing sessions cannot recover an unobserved close;
inspect the loader's gap report and alignment exclusions. Do not treat forward
alignment as a complete exchange-calendar or announcement-time model.

## Entry chronology

`run_backtest(prices, events, config)` opens just after 16:00 New York at the entry
session close. The trailing primary baseline ends at that close. The expanding
absolute-move estimate uses only event windows completed by then, including prior
observations whose own trade was excluded. At least `min_history_events=2` completed
observations are required; insufficient history skips trades without future-data
fallback. Missing historical moves are excluded. Events are ordered by date.
The event schedule itself is assumed known; this is not a point-in-time vendor
calendar archive.

A scalar `implied_vols` is a constant-IV scenario. A Series requires a matching
`implied_observed_at` Series with observation timestamps no later than entry
(naive timestamps interpreted as UTC). Quote freshness, liquidity and expiry
compatibility still require external verification. `implied_vol_days` defines
its term. Remaining jump variance is `(IV² − baseline²)*term`, floored at zero.

Default move calibration converts the estimated absolute move m into jump
variance `(m/sqrt(2/pi))²`, an ATM Gaussian straddle approximation. It does not
promise zero expected P&L at multiplier one. The retrospective leave-one-out
move diagnostic includes future observations and is never used for trade inputs.
Constant inputs to the diagnostic are assumptions, not historical market evidence.

## Option simulation and risk

The calendar sells the straddle expiring after the event and buys the straddle
expiring before it. Both strikes are entry spot. The short-straddle alternative
omits the long leg. Each leg is revalued with Black-Scholes; expiry uses intrinsic
value. Hedge P&L uses the previous close's delta against the next spot change.
Jump loading remains until the modeled window resolves. `back_iv_multiplier`
applies to the long leg. Nonzero hedge bands are rejected because they are unimplemented.

P&L is per underlying unit (multiply by 100 for standard US equity option contracts).
Fees are a round-trip fraction of gross entry premiums, plus adverse volatility
slippage. Stock borrow, hedge transaction costs, dividends, cash-account financing,
American exercise/assignment, discrete strikes and actual option surfaces are not
modeled. A nonzero Black-Scholes rate does not add financing cashflows.

Returns divide by front premium, net credit (with a reported near-zero fallback)
or underlying notional. Those are scenario normalization choices, not brokerage
margin or feasible portfolio capital. Additive accumulation allocates a fixed
independent unit to each event and can cross zero. Overlapping trades are independent,
without aggregate margin constraints. Compounding is rejected when a trade loses
100% or more. Sharpe uses event frequency and is descriptive with limited samples;
win rate alone does not summarize tail risk. Short option losses remain unbounded.

## CLI and API

- Offline: `vol-decom --ticker SYNTHETIC --prices-csv examples/synthetic_prices.csv --earnings-csv examples/synthetic_events.csv`
- Decomposition only: add `--no-backtest`.
- Sensitivity: `--baseline-windows 30 60 --estimator yang_zhang --baseline-gap 3`.
- Explicit scenario: `--implied-vol 50 --iv-days 21 --iv-multiplier 1.1`.
- Universe: `--tickers INTC AAPL` or `--universe-csv symbols.csv`.
- Outputs: `--save-csv --outdir output`; charts require `pip install -e '.[plot]'`
  and `--plot`, optionally `--backend plotly`.

The numerical API exposes `decompose_events`, `summarize_decomposition`, `vol_cone`,
`volatility_risk_premium`, `BacktestConfig`, `run_backtest` and `bs_straddle`.
Provider access and disk caching are optional; CSV inputs do not require yfinance.
