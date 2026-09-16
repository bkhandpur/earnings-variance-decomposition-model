# Earnings Variance Decomposition

`vol_decom` separates earnings jump variance from the diffusive volatility observed
between announcements. It also measures an event volatility premium and simulates a
delta-neutral options trade around each release.

![tests](https://github.com/bkhandpur/earnings-variance-decomposition-model/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Research question

An earnings announcement introduces a discrete jump into an otherwise continuous return
process. Both risks are priced by the same option, but they behave differently: diffusive
volatility can be delta hedged, while the announcement jump is released in one session.

The model estimates a pre-event diffusive baseline, measures realized variance across
the event and recovers the jump by subtraction. This decomposition makes the assumption
behind an earnings-volatility trade explicit: the priced jump must exceed the move that
ultimately realizes.

Built from a research prototype used for a March 2026 derivatives pitch. The original single file script is preserved verbatim in [`prototype/original_prototype.py`](prototype/original_prototype.py) and reproduces bit for bit via `--prototype-mode`.

## Cross-sectional use

Nothing in the model is tuned to a single company. Volatility levels, reporting
frequency, price scale and calendar quirks are inferred from the supplied data. The CLI
accepts one symbol or a universe.

```bash
python main.py --ticker INTC AAPL NVDA TSLA KO JPM SPY --years 6 --summary-only
```

```
==============================================================================
CROSS-SECTION  (6 ticker(s) processed, 1 failed)
==============================================================================
  Ticker  Events Baseline  Jump  Total VarRatio MeanMove Down% Clamp% WinRate Sharpe
    INTC      23    42.0% 98.9% 113.1%    12.4x     9.7% 69.6%  13.0%   78.3%  +0.52
    TSLA      23    53.7% 70.4%  94.6%     5.1x     8.1% 60.9%  21.7%   60.9%  +0.35
    NVDA      23    47.8% 47.4%  71.2%     4.6x     6.0% 52.2%  47.8%   69.6%  +0.07
     JPM      23    22.3% 26.9%  37.5%     4.4x     3.1% 52.2%  26.1%   65.2%  +0.21
    AAPL      23    25.5% 29.2%  39.5%     4.0x     3.3% 43.5%  34.8%   56.5%  +0.29
      KO      23    15.5% 16.4%  22.8%     2.9x     1.9% 34.8%  26.1%   60.9%  +0.62

  Sorted by variance ratio, the share of event-window variance not
  explained by the diffusive baseline. Higher means more of this name's
  risk is released as a discrete jump rather than as ordinary vol.

  Not processed
    SPY        [MissingEarningsDatesError] SPY, yfinance returned an empty earnings calendar.
```

The ordering is the economically sensible one. `INTC` releases 12.4x its baseline variance at earnings, `KO` only 2.9x, and mean absolute moves run from 1.9 percent to 9.7 percent across the same set. `SPY` has no earnings calendar, so it raises a typed exception, gets reported, and the run continues.

Verified across a deliberately awkward basket covering mega caps, high volatility growth names, defensives, financials, foreign ADRs, hyphenated symbols such as `BRK-B`, recent listings such as `PLTR`, low nominal prices such as `F`, and index ETFs that correctly fail. Large universes load from a file with `--universe-csv`.

---

## Core formulas

**Close to close realized volatility.** The textbook estimator. Robust, but it discards the intraday range and so is the noisiest of the three for a given window length.

$$r_i = \ln\\!\\left(\frac{C_i}{C_{i-1}}\right), \qquad \sigma_{cc} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(r_i - \bar r\right)^2}$$

**Parkinson (1980).** Uses the high low range and is roughly 5x more efficient than close to close for a driftless diffusion, but structurally blind to overnight gaps, which is where an earnings move actually happens.

$$\sigma_{P} = \sqrt{\frac{1}{4n\ln 2}\sum_{i=1}^{n}\left[\ln\\!\\left(\frac{H_i}{L_i}\right)\right]^{2}}$$

**Yang Zhang (2000).** Drift independent and gap aware, combining overnight, open to close, and Rogers Satchell range terms. This is the preferred baseline estimator here because it is the only one of the three that sees the overnight gap.

$$\sigma_{YZ} = \sqrt{\sigma_o^{2} + k\\,\sigma_c^{2} + (1-k)\\,\sigma_{rs}^{2}}, \qquad k = \frac{0.34}{1.34 + \frac{n+1}{n-1}}$$

$$\sigma_o^2 = \frac{1}{n-1}\sum \left(o_i - \bar o\right)^2, \quad \sigma_c^2 = \frac{1}{n-1}\sum\left(c_i - \bar c\right)^2, \quad \sigma_{rs}^2 = \frac{1}{n}\sum \left[u_i(u_i - c_i) + d_i(d_i - c_i)\right]$$

where $o_i = \ln(O_i/C_{i-1})$, $c_i = \ln(C_i/O_i)$, $u_i = \ln(H_i/O_i)$, and $d_i = \ln(L_i/O_i)$.

**Annualization.** Every estimator returns an annualized figure.

$$\sigma_{\text{annualized}} = \sigma_{\text{daily}} \cdot \sqrt{252}$$

**Jump variance isolation.** The governing identity of the package. Variance is additive across independent components, so the diffusive baseline subtracts out.

$$\sigma_{\text{jump}} = \sqrt{\max\\!\\left(0,\; \sigma_{\text{total}}^{2} - \sigma_{\text{baseline}}^{2}\right)}$$

The floor at zero is necessary because both terms are sample estimates, and sampling error alone drives the difference negative for genuinely quiet events. Every clamp is **counted and reported** as `clamp_rate`. A high rate signals that the estimator is too noisy to trust, not that the stock does not jump.

**Implied event move.** Backing the market's priced jump out of a term implied volatility that spans the announcement, over $\tau = $ `days_to_expiry` divided by 252 years.

$$m_{\text{implied}} = \sqrt{\max\\!\\left(0,\;\left(\sigma_{IV}^{2} - \sigma_{\text{baseline}}^{2}\right)\tau\right)}$$

**Volatility risk premium, the edge metric.** Per event, measured in move points.

$$\text{VRP}_i = m_{\text{implied},i} - \left|r_{\text{event},i}\right|, \qquad \overline{\text{VRP}} = \frac{1}{N}\sum_{i=1}^{N}\text{VRP}_i$$

A positive mean is the seller's structural edge. Reported alongside its median, a trimmed mean, the hit rate $\Pr(m_{\text{implied}} > |r_{\text{event}}|)$, and a $t$ statistic against zero.

**Variance to move conversion.** Used by the backtest, and a genuine trap worth stating. An at the money straddle marked at variance $V$ is worth $\sqrt{2/\pi}\sqrt{V}S \approx 0.798\sqrt{V}S$, because Black Scholes assumes a Gaussian move whose mean absolute size is $\sqrt{2/\pi}$ of its standard deviation. An earnings jump is closer to a two point $\pm m$ distribution, where mean absolute size *equals* the standard deviation. A variance fair mark is therefore about 20 percent short of being P&L fair for a jump.

$$V_{\text{jump}} = \left(\frac{m_{\text{implied}}}{\sqrt{2/\pi}}\right)^{2}$$

**Delta hedged leg P&L.** Each leg is valued by full Black Scholes revaluation and hedged at every session close.

$$\text{PnL} = \sum_{t}\left[\underbrace{V(S_{t+1},\tau_{t+1},\sigma_{t+1}) - V(S_t,\tau_t,\sigma_t)}_{\text{change in mark}} - \underbrace{\Delta_t\left(S_{t+1}-S_t\right)}_{\text{hedge P\\&L}}\right]$$

Each leg is marked off its *remaining* variance to expiry, which makes the implied volatility crush emergent rather than assumed.

$$\sigma(t) = \sqrt{\frac{\sigma_{\text{baseline}}^{2}\tau_t + V_{\text{jump}}\\,f_t}{\tau_t}}, \qquad f_t = \begin{cases}1 & \text{event still ahead}\\\\ 0 & \text{event has passed}\end{cases}$$

When $f_t$ drops to zero the jump loading leaves the remaining variance and the mark collapses toward the diffusive level. That collapse is the short leg's profit.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or install as a package.

```bash
pip install -e ".[all]"
```

## Quickstart

```bash
python main.py --ticker INTC --years 6
```

```
==============================================================================
INTC  |  earnings variance decomposition  |  vol_decom 1.0.0
==============================================================================
  Prices     1507 sessions, 2020-08-21 -> 2026-08-21
  Continuity 1507 sessions, no anomalous gaps.
  Earnings   49 announcement(s) from yfinance, 2014-07-15 -> 2026-07-23

==============================================================================
VARIANCE DECOMPOSITION  (23 events, vol figures annualized)
==============================================================================
  Baseline estimator          close_to_close
  Baseline windows            [20, 30, 60] (primary 20d)
  Event window                2 session(s)
------------------------------------------------------------------------------
  Mean baseline vol           41.99%
  Mean total vol (event)      113.12%
  Mean JUMP vol               98.88%
  Median jump vol             89.15%
  Mean variance ratio         12.37x
------------------------------------------------------------------------------
  Mean |event move|           9.68%
  Median |event move|         8.22%
  Downside events             69.57%
  Mean post-event vol (10d)   66.01%
  Mean post-event vol ( 5d)   81.17%
------------------------------------------------------------------------------
  Jump clamped at zero        13.04% of events

  Alignment 23 usable event(s), dropped 0 future, 26 for insufficient history, 0 duplicate.

  Per-event detail
               announcement_date  sigma_baseline  sigma_total  sigma_jump  abs_event_return  variance_ratio  jump_clamped
  session_date
  2021-10-21          2021-10-21          0.1817       1.3998      1.3879            0.1242         59.3547         False
  2024-01-25          2024-01-25          0.3296       1.4269      1.3884            0.1268         18.7476         False
  2024-08-01          2024-08-01          0.4161       3.4477      3.4225            0.3019         68.6397         False
  2025-01-30          2025-01-30          0.4368       0.3613      0.0000            0.0294          0.6842          True
  2026-04-23          2026-04-23          0.7881       2.3922      2.2586            0.2119          9.2137         False
```

`INTC` has a mean baseline around 42 percent annualized while its event window averages about 113 percent, a **12.4x variance ratio**, and the decomposition attributes roughly 99 percent annualized to the jump alone. The August 2024 event, a 26 percent down session, shows a 68x ratio. Three events clamp to zero, meaning their event window was *quieter* than the preceding 20 sessions. Nearly 70 percent of events resolved to the downside, which is the negative skew the source research identified in this name.

### Backtest

```bash
python main.py --ticker INTC --years 6 --plot --save-csv
```

```
==============================================================================
BACKTEST  (calendar, delta-hedged daily)
==============================================================================
  Entry / exit                T-5 / T+1 sessions
  Front leg expiry            T+3
  Back leg expiry             T-3
  IV source                   historical_move(k=1)
  Costs                       25 bps/leg, 0.0 vol pts slippage
------------------------------------------------------------------------------
  Trades                      23
  Win Rate                    78.26%
  Mean Return per Event       11.50%
  Median Return per Event     14.05%
  Std Dev per Event           44.60%
  Sharpe Ratio (annualized)   0.52
  Profit Factor               2.00
  Max Drawdown                -127.30%  (of starting capital)
  Sum of Returns              264.58%
  Best / Worst Trade          70.19% / -121.09%
```

Read the win rate and the worst trade **together**. Roughly 78 percent of events won, and one lost more than the entire capital basis. That shape, many small wins and a rare large loss, is the whole risk of short volatility, and it is why the mean is a poor summary of this strategy.

### Supplying a real implied volatility

`--implied-vol` feeds both the volatility risk premium metric and the backtest, so the trade is priced off the level you give rather than the historical default.

```bash
python main.py --ticker INTC --years 6 --implied-vol 65.6 --iv-days 21
```

```
  Mean VRP                    +3.46 move points
  Median VRP                  +6.88 move points
  Mean VRP ex-outliers        +4.97 move points  (4 trimmed)
  Hit rate (implied > real)   78.26%
  t-statistic vs zero         +1.63
------------------------------------------------------------------------------
  Win Rate                    52.17%
  Mean Return per Event       -10.04%
  Profit Factor               0.60
  Mean implied / realized     6.47% / 9.68% move
```

The two runs disagree, and that disagreement is the point. A **flat** 65.6 percent implied volatility held across six years implies a 6.47 percent average event move against 9.68 percent realized, so it systematically *underprices* this name. The mean premium is positive only because many small wins drag it up, the $t$ statistic of 1.63 fails to clear significance, and the backtest loses money. The 65.6 percent figure was a single March 2026 quote for one expiry, while this name's realized baseline rose from roughly 30 percent in 2021 to about 79 percent by 2026. Applying one level uniformly is a modelling error that this tool makes visible rather than hides. A serious study needs a per event implied volatility series, not a constant.

More options are documented in `python main.py --help`.

### Library use

```python
from vol_decom import (load_prices, load_earnings_dates, decompose_events,
                       DecompositionConfig, volatility_risk_premium, run_backtest,
                       plot_decomposition_timeline)

prices = load_prices("INTC", years=7)                       # cached to ./cache
events = decompose_events(prices, load_earnings_dates("INTC"),
                          DecompositionConfig(baseline_windows=(30, 60),
                                              estimator="yang_zhang"))
vrp    = volatility_risk_premium(events, implied_vols=0.656, days_to_expiry=21)
bt     = run_backtest(prices, events)

fig = plot_decomposition_timeline(events)   # returns a figure, never calls show()
```

---

## Package layout

| Module | Responsibility |
|---|---|
| `vol_decom/data.py` | yfinance loading, Parquet caching in `./cache`, OHLCV schema validation, gap and holiday handling |
| `vol_decom/estimators.py` | Close to close, Parkinson, Yang Zhang, both point and rolling, annualized by root 252 |
| `vol_decom/engine.py` | Trading day event alignment, the variance decomposition, volatility cones, the premium metric |
| `vol_decom/backtest.py` | Vectorized delta neutral calendar and straddle simulator with Black Scholes revaluation |
| `vol_decom/visualizer.py` | Matplotlib and Plotly figures, every function *returns* a figure object |
| `vol_decom/exceptions.py` | Typed exception hierarchy |
| `main.py` | argparse CLI, single ticker or cross sectional universe |

**Why argparse rather than typer.** argparse is standard library. This package is meant to be cloned and run by someone evaluating it, and every dependency that is not load bearing is one more place that fails. Typer would buy shell completion and terser declarations, and neither is worth an extra install for a research CLI this size.

## Tests

```bash
pytest tests/
```

```
112 passed in 0.78s
```

Everything runs against synthetic OHLCV and synthetic earnings event fixtures, with **no network access anywhere in the suite**. Coverage includes the normal case with a known 12 percent jump recovered from a 1 percent baseline, the zero jump case where total volatility falls below the baseline and must clamp to zero rather than return a NaN, missing and gapped and malformed data, weekend and holiday alignment, duplicate and future announcements, four volatility regimes from defensive staple to extreme small cap, sub dollar prices, semi annual reporters, the estimators against closed form values, and a guard asserting the hot paths contain no Python level loops.

Continuous integration runs the suite on Python 3.9 through 3.12.

---

## Methodology and assumptions

Read this section before quoting any number out of this package.

**The backtest cannot discover a volatility risk premium.** It can only propagate one you supply. Free data does not retain historical option surfaces, so with no `--implied-vol` the default of `k=1.0` prices every event at the leave one out mean move that name historically delivered, which makes the expected edge **zero by construction**. A positive mean at `k=1.0` is the *skew* of the move distribution, not alpha. Any result at `k>1` follows directly from your assumption that the market overprices by that factor. Only a real surface, or an externally measured premium, can establish an edge.

**The premium proxy is a diagnostic, not a premium.** With no option data supplied, `volatility_risk_premium` benchmarks each event against the leave one out mean of the others, so its mean is approximately zero by construction. It measures dispersion around the historical norm and answers whether an event was bigger than typical. It is not evidence of a risk premium, and the code says so in its own output.

**yfinance data quality.** Prices are split and dividend adjusted by default, which is correct for return based volatility work but means the series is revised retroactively by corporate actions. The earnings calendar is the weaker link. It retains roughly three years of history, occasionally reports dates that disagree with company filings, and gives no reliable indication of before open versus after close timing. For a longer study, supply your own dates via `--earnings-csv`. In the `INTC` run above, 26 of 49 announcements were dropped for insufficient surrounding history, and that attrition is reported rather than hidden.

**No intraday data.** Everything is daily. The event window therefore cannot separate the opening gap from the rest of the session, intraday delta hedging cannot be simulated, and the true realized variance of the event session is understated because the within session path is invisible. Daily close to close on a jump session is a lower bound on what a real gamma position would have paid.

**The two session event window is a deliberate hedge, not an arbitrary choice.** Announcements land either before the open, where the reaction is dated day $D$, or after the close, where it is dated $D+1$. A two session window starting at $D$ contains the reacting session under either convention, so the decomposition does not need announcement timing, which providers report unreliably. The cost is that one non event session is always included, diluting the estimate. If you know your universe reports after the close, use `--announcement-offset 1 --event-window 1`.

**Delta hedging is modelled at the daily close only.** There is no intraday rebalancing, no bid ask on the hedge, no borrow cost, no margin financing, and no early assignment. Options are struck exactly at the money on the entry close and held, whereas real strikes are discrete. The risk free rate defaults to zero, which is immaterial over a one to three week hold relative to the variance term being measured.

**Returns can exceed negative 100 percent,** because a short options position can lose more than its premium. The equity curve is therefore **additive** by default. Compounding a return below negative 100 percent would send the curve negative and make every downstream statistic meaningless. Max drawdown is reported in units of starting capital.

**Sharpe is annualized by the square root of the inferred event frequency,** taken from the median calendar spacing of the actual announcements rather than assumed to be quarterly. With roughly 23 events the estimate is extremely noisy. The $t$ statistic on the premium is the more honest significance measure, and even it assumes independent identically distributed events, which overlapping volatility regimes violate.

**Variance additivity assumes the jump and the diffusion are independent.** If volatility ramps into the announcement, which it empirically does, the pre event baseline is contaminated upward and biases the jump estimate downward. Use `--baseline-gap` to push the baseline window away from the event and test the sensitivity.

---

## Changes from the original prototype

The prototype's modelling logic is the source of truth and is preserved where correct. These changes were deliberate.

| Change | Why |
|---|---|
| **Total volatility now uses zero mean realized variance** rather than demeaned population variance | This is a **bug fix**, not a preference. With `ddof=0`, `np.std([a,b])` equals half the absolute difference of the two values, so it measures the *difference* between the two sessions rather than their magnitude. Two consecutive positive 8 percent days returned zero. That is why **9 of 19 NVDA events printed all zeros**. Those were estimator failures, not zero jump events. Over a two session window the sample mean absorbs the very move being measured. The original behaviour is preserved exactly under `--prototype-mode`, and a test pins the failure mode. |
| **Annualized by root 252** | The prototype reported daily figures. `--no-annualize` restores them. The raw event move is deliberately *not* scaled, since a move is a return rather than a volatility. |
| **Configurable baseline windows of 20, 30, and 60** | The prototype hardcoded 20. The 20 day primary is retained as the default, so headline numbers are unchanged. |
| **Baseline can use Parkinson or Yang Zhang** | The prototype was close to close only. The default is unchanged. The event window is always close to close, since range estimators cannot see the overnight gap that *is* the jump. |
| **Backtest uses full Black Scholes revaluation rather than the dollar gamma approximation** | The gamma form is a second order expansion. On the August 2024 down 26 percent session it produced a negative 546 percent return and a negative 680 percent drawdown, because it treats a piecewise linear payoff as quadratic. Revaluation is exact for the hedging strategy simulated. |
| **Options marked off *remaining* variance rather than a flat volatility** | A flat volatility smears the jump across the term, so the position accrues jump premium on low gamma days and pays it out on one high gamma day, which is a large spurious loss. Marking off remaining variance makes the implied volatility crush emergent. |
| **Event pricing calibrated on the absolute move rather than variance** | See the variance to move conversion above. A variance fair mark is about 20 percent short of P&L fair for a jump, and without this correction `k=1.0` was not edge free. |
| **Event frequency inferred from the data** | Sharpe scales by the square root of events per year. Hardcoding 4 assumes a quarterly reporter and overstates Sharpe by about 40 percent for a semi annual one, which is unacceptable in a package meant to run on any listed name. |
| **Trading day alignment derived from config rather than hardcoded** | The prototype required an index of at least 20 with 10 sessions to spare. Bounds are now derived from the configured windows. The forward rolling rule, first return date on or after the announcement, is unchanged. |
| **Duplicate announcements collapsed** | Two dates rolling onto the same session, for example a Saturday and a Sunday entry, were silently double counted. |
| **Multi ticker and universe support** | The prototype took one symbol from an interactive prompt. The CLI now accepts a list or a CSV universe, reports a cross sectional table, and continues past symbols that fail. |
| **Typed exceptions throughout** | The prototype printed a message and returned `None` on failure. |
| **`lxml` added to requirements** | The yfinance earnings calendar path scrapes an HTML table via `pandas.read_html` and needs it, but does not declare it. Price fetches succeed while earnings fetches fail with a bare `ImportError`. |

### Verified parity

`--prototype-mode` was checked against the original script's published `NVDA` output. All three arrays reproduce to within 1e-7, which is the rounding of the printed originals.

| Metric | Original | Refactor |
|---|---|---|
| Events | 19 | 19 |
| Mean earnings jump vol | 0.022323 | 0.022323 |
| Mean 5 day post event vol | 0.036760 | 0.036760 |
| Mean 10 day post event vol | 0.034575 | 0.034575 |
| Events clamped to zero | 11 of 19 | 11 of 19 |

One presentational difference. The prototype emitted events in the order yfinance returns them, which is reverse chronological, whereas `decompose_events` returns them sorted ascending by session date. The values are identical and only the row order differs. If you are diffing against old output, reverse one of them.

**Preserved unchanged.** The jump isolation identity itself, log return construction and the convention that a return is dated on the later session, forward rolling event alignment, the 20 day baseline as primary, the two session event window and its timing agnostic rationale, the 5 and 10 day post event horizons, and the clamp of negative jump variance to zero.

**Dropped.** The interactive input prompt, replaced by `--ticker`, and printing raw NumPy arrays to stdout, replaced by a labelled per event table and `--save-csv`.

## License

MIT. See [LICENSE](LICENSE).
