# Earnings variance decomposition

A Python research tool that separates event-window variance from a trailing
baseline and simulates daily hedged option scenarios around earnings.
It includes a CLI, numerical API, CSV exports and optional charts.

## Run an offline example

Requires Python 3.9 or newer. The sample is synthetic and needs no credentials
or market-data connection.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
vol-decom --ticker SYNTHETIC --prices-csv examples/synthetic_prices.csv --earnings-csv examples/synthetic_events.csv --save-csv --outdir output
```

The committed sample contains 521 sessions and seven events. The command reports
8.13% mean annualized baseline volatility, 53.35% mean annualized excess-variance
volatility, and 4.79% mean largest absolute session log return. These quantities
have different units of time. Five scenario trades remain after two warm-up events.
Their mean estimated absolute move is 4.79%, versus a 4.70% realized mean over the
same five trades: about **+0.09 percentage points**, before the pricing and hedging
model. Regenerate the tables with the command above; these are illustrative figures.

## What the numbers mean

- Daily excess variance is `max(0, event variance − baseline variance)`.
- Volatility scales by `sqrt(252)` when annualized. The jump volatility is an
  annualized proxy, not an expected announcement move.
- Variance ratio is total divided by baseline, not a fraction of unexplained variance.
- `abs_event_return` selects the largest absolute session log return in the event
  window. `event_window_log_return` sums the whole window.
- Decomposition and the leave-one-out move diagnostic are retrospective.
- Trade inputs use prices through the entry close and completed prior event windows.
  The default expanding historical estimate needs two observations and skips warm-up
  trades. It is a scenario assumption, not observed option-market IV.

## Use your own data

```bash
pip install -e '.[data,plot]'
vol-decom --ticker INTC --earnings-csv your_announcements.csv --plot
vol-decom --help
```

Provider history may be incomplete or unavailable. Offline input is the reproducible
path. [Sample schema](examples/README.md) documents date and price conventions.
[Methodology and API guide](docs/methodology.md) covers timing, estimators, IV inputs,
scenario accounting, plots, universe analysis and historical prototype compatibility.

## Check the package

```bash
pytest
pip wheel --no-deps . -w /tmp/vol-decom-wheels
```

Tests use synthetic inputs, independent pricing calculations and future-data
perturbations. CI installs the built wheel and runs the offline CLI.

MIT license. The [original prototype](prototype/original_prototype.py) is preserved.
