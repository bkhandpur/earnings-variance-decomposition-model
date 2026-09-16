#!/usr/bin/env python3
"""Command-line driver for the earnings variance decomposition model.

Pipeline. Fetch prices and the earnings calendar, decompose each event's
variance into diffusive and jump components -> measure the volatility risk
premium -> backtest the delta-neutral capture trade -> optionally plot.

Why argparse rather than typer
------------------------------
argparse is in the standard library. This package is meant to be cloned and
run by someone evaluating it, and every dependency that is not load-bearing is
a step where that fails. Typer would buy shell completion and slightly terser
option declarations, neither is worth an extra install for a six-flag research
CLI. The subcommand-free flat interface here is small enough that argparse's
verbosity costs nothing.

Examples
--------
Full run on Intel with the defaults.

    python main.py --ticker INTC

Reproduce the original prototype's numbers exactly.

    python main.py --ticker NVDA --prototype-mode

Sweep the mispricing assumption and write charts.

    python main.py --ticker INTC --iv-multiplier 1.15 --plot --outdir figures
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vol_decom import __version__
from vol_decom.backtest import BacktestConfig, run_backtest
from vol_decom.data import (
    DEFAULT_CACHE_DIR,
    clear_cache,
    load_earnings_dates,
    load_earnings_dates_from_csv,
    load_prices,
)
from vol_decom.engine import (
    DecompositionConfig,
    decompose_events,
    summarize_decomposition,
    vol_cone,
    volatility_risk_premium,
)
from vol_decom.exceptions import VolDecomError

LOG_FORMAT = "%(levelname)-8s %(name)s: %(message)s"

RULE = "=" * 78
THIN = "-" * 78


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=textwrap.dedent("""\
            Earnings volatility variance decomposition.

            Separates the one-off jump variance released by an earnings
            announcement from the continuous diffusive volatility around it,
            using  sigma_jump = sqrt(max(0, sigma_total^2 - sigma_baseline^2)),
            then measures the volatility risk premium and backtests a
            delta-neutral trade that harvests it.
            """),
        epilog=textwrap.dedent("""\
            examples
              python main.py --ticker INTC
              python main.py --ticker NVDA --prototype-mode
              python main.py --ticker AAPL --baseline-windows 30 60 --estimator yang_zhang
              python main.py --ticker INTC --iv-multiplier 1.15 --plot --outdir figures

            note on --iv-multiplier
              The backtest has to be told what the options market charged for each
              event. With no option surface supplied, the default k=1.0 prices every
              event at exactly the jump this name historically delivers, so the
              expected edge is zero by construction and the run measures risk, not
              profitability. Raising k asserts that the market overprices earnings by
              that factor, any resulting profit follows from that assumption. See the
              README's Methodology section.
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version",
                        version=f"vol_decom {__version__}")

    data = parser.add_argument_group("data")
    data.add_argument("-t", "--ticker", "--tickers", dest="tickers", nargs="+",
                      metavar="SYM",
                      help="one or more equity symbols, e.g. INTC, or "
                           "'INTC AAPL NVDA TSLA' to run a cross-section")
    data.add_argument("--universe-csv", metavar="PATH",
                      help="read the ticker list from a CSV (first column, or a "
                           "column named 'ticker'/'symbol'). Use for universes "
                           "too large for the command line")
    data.add_argument("--years", type=float, default=7.0, metavar="N",
                      help="years of price history to load (default %(default)s)")
    data.add_argument("--start", metavar="YYYY-MM-DD",
                      help="explicit start date, overrides --years")
    data.add_argument("--end", metavar="YYYY-MM-DD",
                      help="explicit end date (default, today)")
    data.add_argument("--earnings-csv", metavar="PATH",
                      help="load announcement dates from a CSV instead of the "
                           "provider, needed for history beyond what yfinance "
                           "retains (~3 years) or for symbols it does not cover")
    data.add_argument("--earnings-limit", type=int, default=40, metavar="N",
                      help="max announcements to request (default %(default)s)")
    data.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), metavar="DIR",
                      help="on-disk cache location (default %(default)s)")
    data.add_argument("--no-cache", action="store_true",
                      help="bypass the cache and refetch from the provider")
    data.add_argument("--clear-cache", action="store_true",
                      help="delete this ticker's cache entries before running")
    data.add_argument("--strict-gaps", action="store_true",
                      help="abort instead of warning if the price history has "
                           "an anomalous gap")

    model = parser.add_argument_group("decomposition")
    model.add_argument("--baseline-windows", type=int, nargs="+", default=[20, 30, 60],
                       metavar="N",
                       help="baseline lookback windows in trading days, the first "
                            "is used for the jump subtraction (default 20 30 60)")
    model.add_argument("--event-window", type=int, default=2, metavar="N",
                       help="sessions in the event window (default, %(default)s, 2 is "
                            "timing-agnostic across before-open and after-close "
                            "announcements)")
    model.add_argument("--post-windows", type=int, nargs="+", default=[5, 10],
                       metavar="N",
                       help="post-event realized-vol horizons (default 5 10)")
    model.add_argument("--estimator", choices=["close_to_close", "parkinson", "yang_zhang"],
                       default="close_to_close",
                       help="baseline volatility estimator (default %(default)s)")
    model.add_argument("--baseline-gap", type=int, default=0, metavar="N",
                       help="sessions to leave between the baseline window and the "
                            "event, to avoid pre-announcement vol ramp (default 0)")
    model.add_argument("--announcement-offset", type=int, default=0, metavar="N",
                       help="sessions to shift the event forward from the "
                            "announcement date (default 0)")
    model.add_argument("--no-annualize", action="store_true",
                       help="report per-session volatility instead of annualizing "
                            "by sqrt(252)")
    model.add_argument("--prototype-mode", action="store_true",
                       help="reproduce the original prototype exactly, 20-day "
                            "baseline, demeaned population variance on a 2-session "
                            "event window, daily (un-annualized) output. Known to "
                            "zero out roughly half of all events, see the README")

    vrp = parser.add_argument_group("volatility risk premium")
    vrp.add_argument("--implied-vol", type=float, metavar="PCT",
                     help="annualized ATM IV, in percent, applied to every event "
                          "(e.g. 65.6). Without this the VRP falls back to a "
                          "historical proxy whose mean is ~0 by construction")
    vrp.add_argument("--iv-days", type=int, default=21, metavar="N",
                     help="trading days to expiry for the --implied-vol term "
                          "(default %(default)s, i.e. ~30 calendar days)")
    vrp.add_argument("--trim", type=float, default=0.05, metavar="Q",
                     help="two-sided quantile trimmed for the ex-outlier mean "
                          "(default %(default)s, 0 disables)")

    bt = parser.add_argument_group("backtest")
    bt.add_argument("--no-backtest", action="store_true",
                    help="skip the backtest stage")
    bt.add_argument("--structure", choices=["calendar", "short_straddle"],
                    default="calendar",
                    help="trade structure (default %(default)s)")
    bt.add_argument("--entry-offset", type=int, default=5, metavar="N",
                    help="sessions before the event to open (default %(default)s)")
    bt.add_argument("--exit-offset", type=int, default=1, metavar="N",
                    help="sessions after the event to close (default %(default)s)")
    bt.add_argument("--iv-multiplier", type=float, default=1.0, metavar="K",
                    help="how much the market is assumed to overprice the event "
                         "move. 1.0 (default) = priced fairly, zero expected edge")
    bt.add_argument("--cost-bps", type=float, default=25.0, metavar="BPS",
                    help="transaction cost in bps of premium per leg "
                         "(default %(default)s)")
    bt.add_argument("--slippage-vol", type=float, default=0.0, metavar="PTS",
                    help="adverse slippage in vol points, as a decimal "
                         "(0.01 = 1 vol point, default %(default)s)")

    out = parser.add_argument_group("output")
    out.add_argument("--plot", action="store_true",
                     help="render charts (vol cone, jump distribution, "
                          "decomposition timeline, VRP scatter, P&L curve)")
    out.add_argument("--backend", choices=["matplotlib", "plotly"], default="matplotlib",
                     help="charting backend (default %(default)s)")
    out.add_argument("--outdir", default="output", metavar="DIR",
                     help="directory for charts and CSVs (default %(default)s)")
    out.add_argument("--save-csv", action="store_true",
                     help="write the per-event and per-trade tables as CSV")
    out.add_argument("--show-events", action="store_true",
                     help="print the full per-event table rather than a preview")
    out.add_argument("--summary-only", action="store_true",
                     help="with several tickers, print only the cross-sectional "
                          "table and skip the per-ticker detail")
    out.add_argument("--fail-fast", action="store_true",
                     help="abort the whole run on the first ticker that errors "
                          "instead of reporting it and continuing")
    out.add_argument("-v", "--verbose", action="count", default=0,
                     help="-v for INFO, -vv for DEBUG logging")
    out.add_argument("-q", "--quiet", action="store_true",
                     help="suppress warnings")
    return parser


def _fmt(value: float, pct: bool = True, digits: int = 2) -> str:
    """Format a float for the report, handling NaN and infinity."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a" if value is None or np.isnan(value) else ("inf" if value > 0 else "-inf")
    return f"{value * 100:.{digits}f}%" if pct else f"{value:.{digits}f}"


def _section(title: str) -> None:
    """Print a section header."""
    print(f"\n{RULE}\n{title}\n{RULE}")


def _build_decomposition_config(args: argparse.Namespace) -> DecompositionConfig:
    """Translate parsed arguments into a :class:`DecompositionConfig`."""
    if args.prototype_mode:
        return DecompositionConfig.prototype()
    return DecompositionConfig(
        baseline_windows=tuple(args.baseline_windows),
        event_window=args.event_window,
        post_windows=tuple(args.post_windows),
        baseline_gap=args.baseline_gap,
        announcement_offset=args.announcement_offset,
        estimator=args.estimator,
        annualized=not args.no_annualize,
    )


def _report_decomposition(events: pd.DataFrame, cfg: DecompositionConfig,
                          show_all: bool) -> None:
    """Print the decomposition summary and per-event table."""
    summary = summarize_decomposition(events)
    unit = "annualized" if cfg.effective_annualized else "per-session"
    n = int(summary["n_events"])

    _section(f"VARIANCE DECOMPOSITION  ({n} events, vol figures {unit})")
    print(f"  Baseline estimator          {cfg.estimator}")
    print(f"  Baseline windows            {list(cfg.baseline_windows)} "
          f"(primary {cfg.primary_baseline}d)")
    print(f"  Event window                {cfg.event_window} session(s)")
    print(THIN)
    print(f"  Mean baseline vol           {_fmt(summary['mean_sigma_baseline'])}")
    print(f"  Mean total vol (event)      {_fmt(summary['mean_sigma_total'])}")
    print(f"  Mean JUMP vol               {_fmt(summary['mean_sigma_jump'])}")
    print(f"  Median jump vol             {_fmt(summary['median_sigma_jump'])}")
    print(f"  Mean variance ratio         {_fmt(summary['mean_variance_ratio'], pct=False)}x")
    print(THIN)
    print(f"  Mean |event move|           {_fmt(summary['mean_abs_event_return'])}")
    print(f"  Median |event move|         {_fmt(summary['median_abs_event_return'])}")
    print(f"  Downside events             {_fmt(summary['downside_share'])}")
    for key in sorted(k for k in summary if k.startswith("mean_sigma_post_")):
        horizon = key.rsplit("_", 1)[-1]
        print(f"  Mean post-event vol ({horizon:>2}d)    {_fmt(summary[key])}")

    clamp = summary["clamp_rate"]
    print(THIN)
    print(f"  Jump clamped at zero        {_fmt(clamp)} of events")
    if clamp > 0.30:
        print("  ^ WARNING. A high clamp rate means the event-window variance often")
        print("    failed to exceed the baseline. This is an estimator-noise signal,")
        print("    not evidence the name does not jump. Widen --baseline-windows or")
        print("    drop --prototype-mode.")

    align = events.attrs.get("alignment")
    if align is not None:
        print(f"\n  Alignment {align.describe()}")
        if len(align.dropped_insufficient):
            for date, reason in align.dropped_insufficient[:5]:
                print(f"    dropped {pd.Timestamp(date).date()}, {reason}")

    cols = ["announcement_date", "sigma_baseline", "sigma_total", "sigma_jump",
            "abs_event_return", "variance_ratio", "jump_clamped"]
    cols = [c for c in cols if c in events.columns]
    table = events[cols].copy()
    table["announcement_date"] = pd.to_datetime(table["announcement_date"]).dt.date
    print("\n  Per-event detail")
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.max_rows", None if show_all else 12):
        print(textwrap.indent(table.round(4).to_string(), "  "))
    if not show_all and len(table) > 12:
        print(f"  ... {len(table)} events total, pass --show-events for the full table.")


def _report_vrp(result, args: argparse.Namespace) -> None:
    """Print the volatility risk premium block."""
    _section("VOLATILITY RISK PREMIUM  (implied move minus realized move)")
    print(f"  Implied source              {result.implied_source}")
    print(f"  Events compared             {result.n_events}")
    print(THIN)
    print(f"  Mean VRP                    {result.mean_vrp * 100:+.2f} move points")
    print(f"  Median VRP                  {result.median_vrp * 100:+.2f} move points")
    print(f"  Mean VRP ex-outliers        {result.mean_vrp_ex_outliers * 100:+.2f} "
          f"move points  ({result.n_trimmed} trimmed)")
    print(f"  Hit rate (implied > real)   {_fmt(result.hit_rate)}")
    print(f"  t-statistic vs zero         {result.t_stat:+.2f}")
    if result.proxy_note:
        print(THIN)
        for line in textwrap.wrap(result.proxy_note, 74):
            print(f"  ! {line}")


def _report_backtest(result, args: argparse.Namespace) -> None:
    """Print the backtest block."""
    cfg = result.config
    _section(f"BACKTEST  ({cfg.structure}, delta-hedged daily)")
    print(f"  Entry / exit                T-{cfg.entry_offset} / T+{cfg.exit_offset} sessions")
    print(f"  Front leg expiry            T+{cfg.front_expiry_offset}")
    if cfg.structure == "calendar":
        print(f"  Back leg expiry             T-{cfg.back_expiry_offset}")
    print(f"  IV source                   {result.trades.attrs.get('iv_source', 'n/a')}")
    print(f"  Costs                       {cfg.transaction_cost_bps:.0f} bps/leg, "
          f"{cfg.slippage_vol_points * 100:.1f} vol pts slippage")
    if result.n_skipped:
        print(f"  Events skipped              {result.n_skipped} (insufficient history)")
    print(THIN)
    s = result.stats
    print(f"  Trades                      {int(s['n_trades'])}")
    print(f"  Win Rate                    {_fmt(s['win_rate'])}")
    print(f"  Mean Return per Event       {_fmt(s['mean_return'])}")
    print(f"  Median Return per Event     {_fmt(s['median_return'])}")
    print(f"  Std Dev per Event           {_fmt(s['std_return'])}")
    print(f"  Sharpe Ratio (annualized)   {_fmt(s['sharpe_ratio'], pct=False)}")
    print(f"  Profit Factor               {_fmt(s['profit_factor'], pct=False)}")
    print(f"  Max Drawdown                {_fmt(s['max_drawdown'])}"
          f"{'' if cfg.compound else '  (of starting capital)'}")
    label = "Cumulative Return" if cfg.compound else "Sum of Returns   "
    print(f"  {label}           {_fmt(s['total_return'])}")
    print(f"  Best / Worst Trade          {_fmt(s['best_trade'])} / {_fmt(s['worst_trade'])}")
    print(f"  Mean implied / realized     {_fmt(s['mean_implied_move'])} / "
          f"{_fmt(s['mean_realized_move'])} move")

    supplied = result.trades.attrs.get("iv_source") == "supplied"
    print(THIN)
    if supplied:
        print("  ! Priced off the IV you supplied. The edge shown is whatever that")
        print("    level implies versus the moves that actually arrived.")
    elif abs(cfg.implied_move_multiplier - 1.0) < 1e-9:
        print("  ! k=1.0. Every event was priced at the move this name historically")
        print("    delivers, so the expected edge is zero by construction. A positive")
        print("    mean here is the skew of the move distribution (many events below")
        print("    the mean, a few far above), not alpha. Read the win rate and the")
        print("    worst trade together, because that shape is the whole risk of the trade.")
    else:
        print(f"  ! k={cfg.implied_move_multiplier:g}: this run ASSUMES the market prices the event move")
        print(f"    at {cfg.implied_move_multiplier:g}x what actually arrives. The reported edge follows")
        print("    from that assumption, it is not an independent finding.")


def _make_plots(ticker: str, prices: pd.DataFrame, events: pd.DataFrame, vrp, bt,
                args: argparse.Namespace) -> List[str]:
    """Render and save the chart set.

    Returns:
        Paths written.
    """
    from vol_decom.visualizer import (
        plot_decomposition_timeline,
        plot_jump_distribution,
        plot_pnl_curve,
        plot_vol_cone,
        plot_vrp_scatter,
        save_figure,
    )

    outdir = Path(args.outdir)
    ext = "html" if args.backend == "plotly" else "png"
    tag = ticker.upper()
    written: List[str] = []

    jobs = []
    try:
        cone = vol_cone(prices, estimator=args.estimator)
        jobs.append(("vol_cone", lambda: plot_vol_cone(cone, backend=args.backend)))
    except VolDecomError as exc:
        logging.warning("Skipping vol cone, %s", exc)

    jobs.append(("jump_distribution",
                 lambda: plot_jump_distribution(events, backend=args.backend)))
    jobs.append(("decomposition_timeline",
                 lambda: plot_decomposition_timeline(events, backend=args.backend)))
    if vrp is not None:
        jobs.append(("vrp_scatter", lambda: plot_vrp_scatter(vrp, backend=args.backend)))
    if bt is not None:
        jobs.append(("pnl_curve", lambda: plot_pnl_curve(bt, backend=args.backend)))

    for name, maker in jobs:
        try:
            fig = maker()
            path = save_figure(fig, outdir / f"{tag}_{name}.{ext}")
            written.append(path)
        except (ValueError, ImportError, TypeError) as exc:
            logging.warning("Could not render %s, %s", name, exc)
    return written


def _resolve_tickers(args: argparse.Namespace,
                     parser: argparse.ArgumentParser) -> List[str]:
    """Build the ticker list from ``--tickers`` and/or ``--universe-csv``.

    Symbols are upper-cased, stripped, and de-duplicated while preserving the
    order given, so a run over a universe file is reproducible.

    Args:
        args: Parsed arguments.
        parser: Used to raise a usage error when no symbol was supplied.

    Returns:
        Ordered, de-duplicated list of symbols.
    """
    symbols: List[str] = list(args.tickers or [])

    if args.universe_csv:
        try:
            frame = pd.read_csv(args.universe_csv)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            parser.error(f"could not read --universe-csv {args.universe_csv}, {exc}")
            return []
        lowered = {str(c).strip().lower(): c for c in frame.columns}
        column = next((lowered[c] for c in ("ticker", "symbol", "tickers")
                       if c in lowered), frame.columns[0] if len(frame.columns) else None)
        if column is None:
            parser.error(f"--universe-csv {args.universe_csv} has no usable column")
            return []
        symbols += [str(v) for v in frame[column].dropna().tolist()]

    seen: set = set()
    out: List[str] = []
    for raw in symbols:
        sym = str(raw).upper().strip()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    if not out:
        parser.error("no tickers supplied. Use --ticker SYM [SYM ...] or --universe-csv")
    return out


def analyze_ticker(ticker: str,
                   args: argparse.Namespace,
                   cfg: DecompositionConfig,
                   verbose: bool = True) -> Dict[str, Any]:
    """Run the full pipeline for one symbol.

    Args:
        ticker: Equity symbol.
        args: Parsed arguments.
        cfg: Decomposition configuration.
        verbose: Print the per-ticker report blocks.

    Returns:
        Dict with keys ``ticker``, ``prices``, ``events``, ``summary``, ``vrp``
        and ``backtest``. The latter two may be ``None`` when that stage was
        skipped or could not run.

    Raises:
        VolDecomError: Any data, alignment or model failure for this symbol.
            Callers running a universe are expected to catch this per ticker.
    """
    if args.clear_cache:
        removed = clear_cache(args.cache_dir, ticker)
        if verbose:
            print(f"Cleared {removed} cache file(s) for {ticker}.")

    if verbose:
        _section(f"{ticker}  |  earnings variance decomposition  |  vol_decom {__version__}")

    prices = load_prices(
        ticker, start=args.start, end=args.end, years=args.years,
        cache_dir=args.cache_dir, force_refresh=args.no_cache,
        raise_on_gaps=args.strict_gaps,
    )
    if verbose:
        report = prices.attrs.get("gap_report")
        print(f"  Prices     {len(prices)} sessions, "
              f"{prices.index[0].date()} -> {prices.index[-1].date()}")
        if report is not None:
            print(f"  Continuity {report.describe()}")

    if args.earnings_csv:
        edates = load_earnings_dates_from_csv(args.earnings_csv)
        src = f"CSV {args.earnings_csv}"
    else:
        edates = load_earnings_dates(
            ticker, limit=args.earnings_limit, cache_dir=args.cache_dir,
            force_refresh=args.no_cache,
        )
        src = "yfinance"
    if verbose:
        print(f"  Earnings   {len(edates)} announcement(s) from {src}, "
              f"{edates[0].date()} -> {edates[-1].date()}")

    events = decompose_events(prices, edates, cfg)
    summary = summarize_decomposition(events)
    if verbose:
        _report_decomposition(events, cfg, args.show_events)

    vrp = None
    try:
        iv = args.implied_vol / 100.0 if args.implied_vol is not None else None
        vrp = volatility_risk_premium(
            events, implied_vols=iv, days_to_expiry=args.iv_days,
            trim_quantile=args.trim,
        )
        if verbose:
            _report_vrp(vrp, args)
    except ValueError as exc:
        if verbose:
            print(f"\n[VRP skipped] {exc}")

    bt = None
    if not args.no_backtest:
        try:
            bt_cfg = BacktestConfig(
                structure=args.structure,
                entry_offset=args.entry_offset,
                exit_offset=args.exit_offset,
                implied_move_multiplier=args.iv_multiplier,
                transaction_cost_bps=args.cost_bps,
                slippage_vol_points=args.slippage_vol,
            )
            iv_series = args.implied_vol / 100.0 if args.implied_vol is not None else None
            bt = run_backtest(
                prices,
                events,
                bt_cfg,
                implied_vols=iv_series,
                implied_vol_days=args.iv_days,
            )
            if verbose:
                _report_backtest(bt, args)
        except VolDecomError as exc:
            if verbose:
                print(f"\n[Backtest skipped] {exc}")

    if args.save_csv:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written = [str(outdir / f"{ticker}_events.csv")]
        events.to_csv(written[0])
        if bt is not None:
            path = outdir / f"{ticker}_trades.csv"
            bt.trades.to_csv(path)
            written.append(str(path))
        if verbose:
            print(f"\nWrote CSV {', '.join(written)}")

    if args.plot:
        paths = _make_plots(ticker, prices, events, vrp, bt, args)
        if verbose:
            if paths:
                print(f"\nWrote {len(paths)} chart(s)")
                for path in paths:
                    print(f"  {path}")
            else:
                print("\nNo charts were written (see warnings above).")

    return {"ticker": ticker, "prices": prices, "events": events,
            "summary": summary, "vrp": vrp, "backtest": bt}


def _cross_section(results: List[Dict[str, Any]],
                   failures: List[Tuple[str, str]]) -> None:
    """Print a one-row-per-ticker comparison of the whole run.

    This is the view that matters when the model is pointed at a universe
    rather than a single name. Ranking by variance ratio shows which names
    release the most of their volatility as a discrete event, which is the
    screen an earnings vol desk actually wants.

    Args:
        results: Successful :func:`analyze_ticker` payloads.
        failures: ``(ticker, reason)`` for symbols that could not be processed.
    """
    _section(f"CROSS-SECTION  ({len(results)} ticker(s) processed"
             f"{f', {len(failures)} failed' if failures else ''})")

    if results:
        rows = []
        for res in results:
            summary, bt = res["summary"], res["backtest"]
            rows.append({
                "Ticker": res["ticker"],
                "Events": int(summary["n_events"]),
                "Baseline": summary["mean_sigma_baseline"],
                "Jump": summary["mean_sigma_jump"],
                "Total": summary["mean_sigma_total"],
                "VarRatio": summary["mean_variance_ratio"],
                "MeanMove": summary["mean_abs_event_return"],
                "Down%": summary["downside_share"],
                "Clamp%": summary["clamp_rate"],
                "WinRate": bt.stats["win_rate"] if bt else float("nan"),
                "Sharpe": bt.stats["sharpe_ratio"] if bt else float("nan"),
            })
        table = pd.DataFrame(rows).sort_values("VarRatio", ascending=False)
        formatted = table.copy()
        for col in ("Baseline", "Jump", "Total", "MeanMove", "Down%", "Clamp%", "WinRate"):
            formatted[col] = table[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1%}")
        formatted["VarRatio"] = table["VarRatio"].map(lambda v: f"{v:.1f}x")
        formatted["Sharpe"] = table["Sharpe"].map(
            lambda v: "n/a" if pd.isna(v) else f"{v:+.2f}")
        with pd.option_context("display.width", 200, "display.max_columns", 30,
                               "display.max_rows", None):
            print(textwrap.indent(formatted.to_string(index=False), "  "))
        print("\n  Sorted by variance ratio, the share of event-window variance not")
        print("  explained by the diffusive baseline. Higher means more of this name's")
        print("  risk is released as a discrete jump rather than as ordinary vol.")

    if failures:
        print("\n  Not processed")
        for ticker, reason in failures:
            print(f"    {ticker:10s} {reason}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. 0 when every ticker succeeded, 1 when at least one
        failed, 2 on invalid arguments, 130 on interrupt.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.quiet:
        level = logging.ERROR
    elif args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)

    tickers = _resolve_tickers(args, parser)

    try:
        cfg = _build_decomposition_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    multi = len(tickers) > 1
    verbose = not (multi and args.summary_only)

    results: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    try:
        for ticker in tickers:
            try:
                results.append(analyze_ticker(ticker, args, cfg, verbose=verbose))
            except VolDecomError as exc:
                reason = f"[{type(exc).__name__}] {exc}"
                failures.append((ticker, reason))
                if args.fail_fast:
                    print(f"\nERROR {ticker}, {reason}", file=sys.stderr)
                    return 1
                if multi:
                    logging.warning("%s skipped, %s", ticker, exc)
                    if verbose:
                        print(f"\n[{ticker} skipped] {exc}")
                else:
                    print(f"\nERROR {reason}", file=sys.stderr)
                    return 1

        if multi:
            _cross_section(results, failures)

        print(f"\n{RULE}\nDone. {len(results)} succeeded, {len(failures)} failed.\n{RULE}")
        return 0 if not failures else 1

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
