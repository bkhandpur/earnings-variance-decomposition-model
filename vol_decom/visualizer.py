"""Charting for the decomposition, event distributions and backtest results.

Every function **returns** a figure object and never calls ``show()`` or
``savefig()``. That keeps them usable identically from a notebook (where the
returned figure renders inline), from the CLI (which saves them), and from a
test (which just asserts on the object).

Matplotlib is the default backend because it has no browser dependency and
writes PNG/PDF directly. Pass ``backend="plotly"`` to any function for an
interactive figure instead, Plotly is an optional dependency and its absence
raises a clear :class:`ImportError` rather than failing obscurely.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "plot_vol_cone",
    "plot_jump_distribution",
    "plot_decomposition_timeline",
    "plot_pnl_curve",
    "plot_vrp_scatter",
    "save_figure",
]

logger = logging.getLogger(__name__)

_BACKENDS = ("matplotlib", "plotly")


def _require_matplotlib():
    """Import and return the pyplot module, with an actionable error if absent.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma. No cover - environment dependent
        raise ImportError(
            "matplotlib is required for the default charting backend. "
            "Install it with `pip install matplotlib`."
        ) from exc


def _require_plotly():
    """Import and return ``plotly.graph_objects``.

    Raises:
        ImportError: If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as exc:  # pragma. No cover - environment dependent
        raise ImportError(
            "plotly is required for backend='plotly'. Install it with "
            "`pip install plotly`, or use the default matplotlib backend."
        ) from exc


def _check_backend(backend: str) -> None:
    """Validate the backend name.

    Raises:
        ValueError: If ``backend`` is not a known backend.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got '{backend}'")


def _pct(x: float) -> str:
    """Format a decimal as a percentage string."""
    return f"{x * 100:.1f}%"


def plot_vol_cone(cone: pd.DataFrame,
                  current: Optional[pd.Series] = None,
                  title: Optional[str] = None,
                  backend: str = "matplotlib") -> Any:
    """Plot a volatility cone. Realized-vol quantiles against horizon.

    The cone shows the historical distribution of trailing realized vol at
    each horizon. Quantile bands narrow as the horizon lengthens, short
    windows are dominated by noise, long windows average it out. Overlaying
    the current reading shows whether vol is rich or cheap *for its horizon*,
    which a single spot number cannot tell you.

    Args:
        cone: Output of :func:`vol_decom.engine.vol_cone`, indexed by window
            length with ``q05``/``q25``/``q50``/... columns.
        current: Optional series of current vol by window, defaults to the
            ``last`` column of ``cone`` when present.
        title: Figure title. Auto-generated from ``cone.attrs`` if omitted.
        backend: ``"matplotlib"`` or ``"plotly"``.

    Returns:
        A ``matplotlib.figure.Figure`` or ``plotly.graph_objects.Figure``.

    Raises:
        ValueError: If ``cone`` is empty or exposes no quantile columns.
        ImportError: If the requested backend is not installed.
    """
    _check_backend(backend)
    if cone is None or cone.empty:
        raise ValueError("plot_vol_cone requires a non-empty cone frame.")

    qcols = sorted([c for c in cone.columns if c.startswith("q")],
                   key=lambda c: int(c[1:]))
    if not qcols:
        raise ValueError("cone frame exposes no quantile columns (expected 'q05', ...).")

    windows = cone.index.to_numpy()
    if current is None and "last" in cone.columns:
        current = cone["last"]

    ticker = cone.attrs.get("ticker", "")
    est = cone.attrs.get("estimator", "close_to_close")
    title = title or f"Volatility cone{' - ' + ticker if ticker else ''} ({est})"

    if backend == "plotly":
        go = _require_plotly()
        fig = go.Figure()
        # Fill between symmetric quantile pairs, outermost first.
        pairs = list(zip(qcols, reversed(qcols)))[: len(qcols) // 2]
        for lo, hi in pairs:
            fig.add_trace(go.Scatter(
                x=np.concatenate([windows, windows[::-1]]),
                y=np.concatenate([cone[hi].to_numpy(), cone[lo].to_numpy()[::-1]]),
                fill="toself", mode="lines", line=dict(width=0),
                name=f"{lo}-{hi}", opacity=0.25, hoverinfo="skip",
            ))
        mid = "q50" if "q50" in cone.columns else qcols[len(qcols) // 2]
        fig.add_trace(go.Scatter(x=windows, y=cone[mid], mode="lines+markers",
                                 name=f"median ({mid})", line=dict(width=2)))
        if current is not None:
            fig.add_trace(go.Scatter(x=windows, y=current.to_numpy(), mode="lines+markers",
                                     name="current", line=dict(dash="dash", width=2)))
        fig.update_layout(title=title, xaxis_title="Lookback window (trading days)",
                          yaxis_title="Annualized volatility", yaxis_tickformat=".0%",
                          template="plotly_white")
        return fig

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pairs = list(zip(qcols, reversed(qcols)))[: len(qcols) // 2]
    for i, (lo, hi) in enumerate(pairs):
        ax.fill_between(windows, cone[lo], cone[hi], alpha=0.20 + 0.12 * i,
                        color="steelblue", label=f"{lo}-{hi}")
    mid = "q50" if "q50" in cone.columns else qcols[len(qcols) // 2]
    ax.plot(windows, cone[mid], marker="o", color="navy", lw=2, label=f"median ({mid})")
    if current is not None:
        ax.plot(windows, current.to_numpy(), marker="s", ls="--", color="crimson",
                lw=2, label="current")
    ax.set_xlabel("Lookback window (trading days)")
    ax.set_ylabel("Annualized volatility")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_jump_distribution(events: pd.DataFrame,
                           column: str = "sigma_jump",
                           bins: int = 20,
                           show_baseline: bool = True,
                           title: Optional[str] = None,
                           backend: str = "matplotlib") -> Any:
    """Histogram the distribution of isolated earnings jumps.

    Two things to read off this chart. First, the *mass at zero*, where bars stacked
    at exactly 0 are events where the jump variance was clamped, i.e. the
    event window's variance did not exceed the diffusive baseline. Second, the
    *skew*, because a right tail much longer than the body means the average jump is
    a poor summary and the median should be quoted instead.

    Args:
        events: Output of :func:`vol_decom.engine.decompose_events`.
        column: Which column to histogram. ``"sigma_jump"`` for the isolated
            jump vol, ``"abs_event_return"`` for the raw absolute move, or
            ``"event_return"`` to see the directional skew.
        bins: Number of histogram bins.
        show_baseline: Draw a vertical line at the mean baseline vol for scale.
        title: Figure title.
        backend: ``"matplotlib"`` or ``"plotly"``.

    Returns:
        A figure object.

    Raises:
        ValueError: If ``events`` is empty or lacks ``column``.
        ImportError: If the requested backend is not installed.
    """
    _check_backend(backend)
    if events is None or events.empty:
        raise ValueError("plot_jump_distribution requires a non-empty event frame.")
    if column not in events.columns:
        raise ValueError(
            f"column '{column}' not in events frame. Available columns are "
            f"{list(events.columns)}"
        )

    vals = events[column].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        raise ValueError(f"column '{column}' has no non-null values to plot.")

    ticker = events.attrs.get("ticker", "")
    title = title or f"Distribution of {column}{' - ' + ticker if ticker else ''}"
    mean_v, median_v = float(np.mean(vals)), float(np.median(vals))
    n_zero = int(np.sum(vals == 0.0))

    if backend == "plotly":
        go = _require_plotly()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=bins, name=column,
                                   marker_line_width=1, opacity=0.8))
        fig.add_vline(x=mean_v, line_dash="dash", line_color="crimson",
                      annotation_text=f"mean {mean_v:.4f}")
        fig.add_vline(x=median_v, line_dash="dot", line_color="darkgreen",
                      annotation_text=f"median {median_v:.4f}")
        subtitle = f"n={vals.size}" + (f", {n_zero} clamped at zero" if n_zero else "")
        fig.update_layout(title=f"{title}<br><sub>{subtitle}</sub>",
                          xaxis_title=column, yaxis_title="Events",
                          template="plotly_white", bargap=0.05)
        return fig

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean_v, color="crimson", ls="--", lw=2, label=f"mean {mean_v:.4f}")
    ax.axvline(median_v, color="darkgreen", ls=":", lw=2, label=f"median {median_v:.4f}")
    if show_baseline and "sigma_baseline" in events.columns:
        base = float(events["sigma_baseline"].mean())
        ax.axvline(base, color="grey", ls="-.", lw=1.5,
                   label=f"mean baseline {base:.4f}")
    subtitle = f"n = {vals.size}" + (f"   |   {n_zero} clamped at zero" if n_zero else "")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel(column)
    ax.set_ylabel("Events")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_decomposition_timeline(events: pd.DataFrame,
                                title: Optional[str] = None,
                                backend: str = "matplotlib") -> Any:
    """Plot baseline, total and jump volatility per event, in time order.

    This is the decomposition made visible. For each event, how much of the
    observed variance was ordinary diffusion and how much was the jump. The
    lower panel shows the variance ratio ``sigma_total^2 / sigma_baseline^2``,
    where 1.0 marks an event that was statistically indistinguishable from a
    normal session, and grey bands mark events whose jump variance was clamped.

    The baseline and jump bars are drawn **side by side, not stacked**. It is
    tempting to stack them, but volatilities do not add, variances do, so a
    stacked bar would imply ``sigma_total = sigma_baseline + sigma_jump``, which
    is false and would sit visibly above the plotted total. Grouped bars keep
    the comparison honest.

    Args:
        events: Output of :func:`vol_decom.engine.decompose_events`.
        title: Figure title.
        backend: ``"matplotlib"`` or ``"plotly"``.

    Returns:
        A figure object.

    Raises:
        ValueError: If ``events`` is empty or lacks the decomposition columns.
        ImportError: If the requested backend is not installed.
    """
    _check_backend(backend)
    if events is None or events.empty:
        raise ValueError("plot_decomposition_timeline requires a non-empty event frame.")
    needed = ("sigma_baseline", "sigma_total", "sigma_jump")
    missing = [c for c in needed if c not in events.columns]
    if missing:
        raise ValueError(f"events frame is missing column(s) {missing}")

    ticker = events.attrs.get("ticker", "")
    title = title or f"Earnings variance decomposition{' - ' + ticker if ticker else ''}"
    x = events.index

    if backend == "plotly":
        go = _require_plotly()
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.68, 0.32], vertical_spacing=0.08,
                            subplot_titles=("Volatility by event", "Variance ratio"))
        fig.add_trace(go.Bar(x=x, y=events["sigma_baseline"], name="baseline (diffusive)",
                             marker_color="lightsteelblue"), row=1, col=1)
        fig.add_trace(go.Bar(x=x, y=events["sigma_jump"], name="jump",
                             marker_color="crimson"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=events["sigma_total"], name="total (event window)",
                                 mode="lines+markers", line=dict(color="navy", width=2)),
                      row=1, col=1)
        if "variance_ratio" in events.columns:
            fig.add_trace(go.Scatter(x=x, y=events["variance_ratio"], name="variance ratio",
                                     mode="lines+markers", line=dict(color="darkorange")),
                          row=2, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=2, col=1)
        fig.update_layout(title=title, barmode="group", template="plotly_white")
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        return fig

    plt = _require_matplotlib()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )
    # Grouped, not stacked. Volatilities are not additive (see the docstring).
    width = max(4.0, 200.0 / max(len(x), 1))
    offset = pd.Timedelta(days=width / 2.0)
    ax1.bar(x - offset, events["sigma_baseline"], width=width,
            color="lightsteelblue", label="baseline (diffusive)")
    ax1.bar(x + offset, events["sigma_jump"], width=width,
            color="crimson", alpha=0.85, label="jump")
    ax1.plot(x, events["sigma_total"], marker="o", ms=4, color="navy", lw=1.5,
             label="total (event window)")
    ax1.set_ylabel("Annualized volatility")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_title(title)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3, axis="y")

    if "variance_ratio" in events.columns:
        ax2.plot(x, events["variance_ratio"], marker="o", ms=4, color="darkorange", lw=1.5)
        ax2.axhline(1.0, color="grey", ls="--", lw=1)
        ax2.set_ylabel("$\\sigma^2_{total}/\\sigma^2_{base}$")
    if "jump_clamped" in events.columns:
        clamped = events.index[events["jump_clamped"].to_numpy(dtype=bool)]
        for c in clamped:
            ax2.axvline(c, color="grey", alpha=0.35, lw=6)
    ax2.set_xlabel("Event session")
    ax2.grid(alpha=0.3, axis="y")
    fig.autofmt_xdate()
    # No tight_layout here. The explicit gridspec height ratios and hspace
    # already set the geometry, and tight_layout warns on the shared-axis pair.
    return fig


def plot_pnl_curve(result: Any,
                   title: Optional[str] = None,
                   backend: str = "matplotlib") -> Any:
    """Plot the cumulative P&L curve and per-event returns of a backtest.

    The upper panel is the compounded equity curve with its drawdown shaded,
    the lower panel is the per-event return, coloured by sign. Reading them
    together matters. A smooth-looking equity curve built from a handful of
    large wins and many small losses is a very different risk profile from
    one built the other way around, and only the lower panel shows which.

    Args:
        result: A :class:`vol_decom.backtest.BacktestResult`, or a
            ``pd.Series`` equity curve.
        title: Figure title.
        backend: ``"matplotlib"`` or ``"plotly"``.

    Returns:
        A figure object.

    Raises:
        ValueError: If the result carries no trades.
        ImportError: If the requested backend is not installed.
    """
    _check_backend(backend)

    if isinstance(result, pd.Series):
        equity = result
        returns = equity.pct_change().fillna(equity.iloc[0] - 1.0)
        stats: Dict[str, float] = {}
    else:
        equity = getattr(result, "equity_curve", None)
        trades = getattr(result, "trades", None)
        if equity is None or trades is None or trades.empty:
            raise ValueError("plot_pnl_curve requires a backtest result with trades.")
        returns = trades["return"]
        stats = getattr(result, "stats", {}) or {}

    if equity.empty:
        raise ValueError("equity curve is empty.")

    if stats:
        subtitle = (
            f"n={int(stats.get('n_trades', len(equity)))}   "
            f"win rate {_pct(stats.get('win_rate', float('nan')))}   "
            f"mean/event {_pct(stats.get('mean_return', float('nan')))}   "
            f"Sharpe {stats.get('sharpe_ratio', float('nan')):.2f}   "
            f"PF {stats.get('profit_factor', float('nan')):.2f}   "
            f"maxDD {_pct(stats.get('max_drawdown', float('nan')))}"
        )
    else:
        subtitle = f"n={len(equity)}"
    title = title or "Earnings vol-capture backtest"

    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    if backend == "plotly":
        go = _require_plotly()
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.62, 0.38], vertical_spacing=0.09,
                            subplot_titles=("Cumulative equity", "Return per event"))
        fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines+markers",
                                 name="equity", line=dict(color="navy", width=2)),
                      row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=1, col=1)
        colors = ["seagreen" if v >= 0 else "crimson" for v in returns]
        fig.add_trace(go.Bar(x=returns.index, y=returns, name="return/event",
                             marker_color=colors), row=2, col=1)
        fig.update_layout(title=f"{title}<br><sub>{subtitle}</sub>",
                          template="plotly_white", showlegend=False)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        return fig

    plt = _require_matplotlib()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.12},
    )
    ax1.plot(equity.index, equity, marker="o", ms=4, color="navy", lw=2, label="equity")
    ax1.axhline(1.0, color="grey", ls="--", lw=1)
    ax1.fill_between(equity.index, equity, peak, color="crimson", alpha=0.18,
                     label="drawdown")
    ax1.set_ylabel("Equity (1.0 = flat)")
    ax1.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    colors = ["seagreen" if v >= 0 else "crimson" for v in returns]
    width = max(6.0, 300.0 / max(len(returns), 1))
    ax2.bar(returns.index, returns, width=width, color=colors, alpha=0.85)
    ax2.axhline(0.0, color="black", lw=1)
    ax2.set_ylabel("Return / event")
    ax2.set_xlabel("Event session")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.grid(alpha=0.3, axis="y")

    ax3 = ax2.twinx()
    ax3.plot(drawdown.index, drawdown, color="grey", ls=":", lw=1.2)
    ax3.set_ylabel("Drawdown", color="grey", fontsize=8)
    ax3.tick_params(axis="y", labelcolor="grey", labelsize=7)

    fig.autofmt_xdate()
    # tight_layout cannot handle the twinned axis, the gridspec above is explicit.
    return fig


def plot_vrp_scatter(vrp: Any,
                     title: Optional[str] = None,
                     backend: str = "matplotlib") -> Any:
    """Scatter implied against realized event moves, with the 45-degree line.

    Points below the diagonal are events where the market charged more than
    the move that arrived, wins for a vol seller. Points above are losses.
    The vertical distance from the diagonal *is* the per-event premium, so the
    chart shows both the average edge and how lumpily it is distributed.

    Args:
        vrp: A :class:`vol_decom.engine.VRPResult`, or a frame with
            ``implied_move`` and ``realized_move`` columns.
        title: Figure title.
        backend: ``"matplotlib"`` or ``"plotly"``.

    Returns:
        A figure object.

    Raises:
        ValueError: If the required columns are absent or there is no data.
        ImportError: If the requested backend is not installed.
    """
    _check_backend(backend)
    per_event = getattr(vrp, "per_event", vrp)
    if per_event is None or len(per_event) == 0:
        raise ValueError("plot_vrp_scatter requires a non-empty VRP result.")
    for col in ("implied_move", "realized_move"):
        if col not in per_event.columns:
            raise ValueError(f"VRP frame is missing required column '{col}'.")

    imp = per_event["implied_move"].to_numpy(dtype=float)
    rea = per_event["realized_move"].to_numpy(dtype=float)
    lim = float(max(np.nanmax(imp), np.nanmax(rea))) * 1.1

    mean_vrp = getattr(vrp, "mean_vrp", float(np.nanmean(imp - rea)))
    hit = getattr(vrp, "hit_rate", float(np.nanmean(imp > rea)))
    source = getattr(vrp, "implied_source", "unknown")
    subtitle = (f"n={len(per_event)}   mean VRP {mean_vrp * 100:+.2f} pts   "
                f"hit rate {_pct(hit)}   implied source {source}")
    title = title or "Implied vs realized earnings move"

    if backend == "plotly":
        go = _require_plotly()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rea, y=imp, mode="markers", name="events",
                                 marker=dict(size=9, color="steelblue",
                                             line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="fair (45°)",
                                 line=dict(dash="dash", color="grey")))
        fig.update_layout(title=f"{title}<br><sub>{subtitle}</sub>",
                          xaxis_title="Realized move", yaxis_title="Implied move",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          template="plotly_white")
        return fig

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 7))
    below = imp > rea
    ax.scatter(rea[below], imp[below], s=55, color="seagreen", edgecolor="white",
               zorder=3, label="implied > realized (seller wins)")
    ax.scatter(rea[~below], imp[~below], s=55, color="crimson", edgecolor="white",
               zorder=3, label="realized > implied (seller loses)")
    ax.plot([0, lim], [0, lim], ls="--", color="grey", lw=1.5, label="fair (45°)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Realized move")
    ax.set_ylabel("Implied move")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def save_figure(fig: Any, path: Union[str, Path], dpi: int = 150) -> str:
    """Write a figure to disk, dispatching on its type.

    Matplotlib figures are written with ``savefig``, Plotly figures are
    written as standalone HTML if the extension is ``.html``, otherwise via
    ``write_image`` (which needs ``kaleido``).

    Args:
        fig: A matplotlib or plotly figure.
        path: Destination path. The extension selects the format.
        dpi: Raster resolution for matplotlib output.

    Returns:
        The path written, as a string.

    Raises:
        TypeError: If ``fig`` is not a recognized figure type.
        ImportError: If a Plotly static export is requested without kaleido.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(fig, "savefig"):  # matplotlib
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        return str(p)
    if hasattr(fig, "write_html"):  # plotly
        if p.suffix.lower() == ".html":
            fig.write_html(str(p), include_plotlyjs="cdn")
            return str(p)
        try:
            fig.write_image(str(p), scale=dpi / 72.0)
        except (ImportError, ValueError) as exc:
            raise ImportError(
                f"Static export of a Plotly figure to '{p.suffix}' needs kaleido "
                f"(`pip install kaleido`), or save as .html instead. ({exc})"
            ) from exc
        return str(p)
    raise TypeError(f"Unrecognized figure type {type(fig)!r}")
