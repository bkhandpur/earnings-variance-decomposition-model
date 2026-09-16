"""Realized-volatility estimators.

All estimators return **annualized** volatility (a standard deviation, not a
variance) unless ``annualize=False`` is passed, in which case the per-session
figure is returned.  The annualization convention throughout the package is

    sigma_annualized = sigma_daily * sqrt(252)

252 being the conventional count of US equity trading sessions per year.

Three estimators are provided, in increasing statistical efficiency and
increasing sensitivity to data quality.

``close_to_close``
    The textbook sample standard deviation of log returns.  Unbiased and
    robust, but throws away the intraday range, so it is the noisiest of the
    three for a given window length.

``parkinson``
    Uses the high-low range.  Roughly 5x more efficient than close-to-close
    for a diffusion without drift, but systematically *understates* vol when
    a large part of the move happens overnight, which is precisely the case
    for earnings gaps.

``yang_zhang``
    Combines overnight, open-to-close and Rogers-Satchell range terms.  It is
    the only one of the three that is simultaneously drift-independent and
    gap-aware, which makes it the preferred baseline estimator for
    earnings-event work.

Notation used in the docstrings below, for session ``i`` of ``n``.

    O_i, H_i, L_i, C_i   open / high / low / close
    r_i  = ln(C_i / C_{i-1})     close-to-close log return
    o_i  = ln(O_i / C_{i-1})     overnight (gap) log return
    c_i  = ln(C_i / O_i)         open-to-close (intraday) log return
    u_i  = ln(H_i / O_i)
    d_i  = ln(L_i / O_i)
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .exceptions import InsufficientDataError, SchemaValidationError

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "annualize",
    "log_returns",
    "close_to_close",
    "parkinson",
    "yang_zhang",
    "rolling_close_to_close",
    "rolling_parkinson",
    "rolling_yang_zhang",
    "realized_variance",
    "ESTIMATORS",
]

#: Conventional number of US equity trading sessions in a year.
TRADING_DAYS_PER_YEAR: int = 252

_OHLC = ("Open", "High", "Low", "Close")


def annualize(sigma_daily: Union[float, np.ndarray, pd.Series],
              periods_per_year: int = TRADING_DAYS_PER_YEAR
              ) -> Union[float, np.ndarray, pd.Series]:
    """Scale a per-session volatility to an annual figure.

    Formula::

        sigma_annualized = sigma_daily * sqrt(periods_per_year)

    Args:
        sigma_daily: Per-session standard deviation of log returns. Scalar,
            array or Series.
        periods_per_year: Sessions per year. Defaults to 252.

    Returns:
        The annualized volatility, same type as the input.

    Raises:
        ValueError: If ``periods_per_year`` is not strictly positive.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    return sigma_daily * np.sqrt(periods_per_year)


def _require_columns(df: pd.DataFrame, columns: tuple = _OHLC) -> None:
    """Raise :class:`SchemaValidationError` if any required column is absent."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            "OHLC frame is missing required column(s).",
            violations=[f"missing column '{c}'" for c in missing],
        )


def _require_length(n: int, required: int, what: str) -> None:
    """Raise :class:`InsufficientDataError` if ``n < required``."""
    if n < required:
        raise InsufficientDataError(
            f"Not enough observations to compute {what}.",
            required=required, available=n,
        )


def log_returns(close: pd.Series) -> pd.Series:
    """Daily close-to-close log returns, dated on the *later* of the two sessions.

    Formula::

        r_t = ln(C_t) - ln(C_{t-1})

    The alignment convention matters and is deliberate, ``r_t`` carries the
    index label ``t``, so a return that spans the earnings gap is dated on the
    session in which the market reacted, not the session before it.  Every
    windowing routine in :mod:`vol_decom.engine` relies on this.

    Args:
        close: Close prices indexed by a monotonic ``DatetimeIndex``.

    Returns:
        Log returns with the first observation dropped (length ``len(close)-1``).

    Raises:
        InsufficientDataError: If fewer than 2 prices are supplied.
        SchemaValidationError: If any price is non-positive, which would make
            the logarithm undefined.
    """
    _require_length(len(close), 2, "log returns")
    if (close <= 0).any():
        bad = close[close <= 0]
        raise SchemaValidationError(
            "Close prices must be strictly positive to take logarithms.",
            violations=[f"{idx.date()} has {val}" for idx, val in bad.items()][:5],
        )
    return np.log(close).diff().dropna()


def realized_variance(returns: Union[pd.Series, np.ndarray],
                      zero_mean: bool = True,
                      ddof: int = 0) -> float:
    """Realized variance of a return sample, per session.

    Two conventions are supported.

    ``zero_mean=True`` (default, the realized-variance convention)::

        sigma^2 = (1 / n) * sum_i r_i^2

    ``zero_mean=False`` (the sample-variance convention)::

        sigma^2 = (1 / (n - ddof)) * sum_i (r_i - rbar)^2

    The zero-mean form is the right choice for short event windows.  Over a
    2- or 3-day earnings window the sample mean absorbs the very move being
    measured: with ``n=2`` the demeaned estimator collapses to
    ``|r_1 - r_2| / 2``, which returns *zero* for two identical large moves.
    Assuming zero drift over a handful of sessions costs almost nothing and
    removes that failure mode entirely.

    Args:
        returns: Log returns.
        zero_mean: If True, use the zero-drift realized-variance form.
        ddof: Delta degrees of freedom, used only when ``zero_mean`` is False.

    Returns:
        Per-session variance as a float.

    Raises:
        InsufficientDataError: If the sample is empty, or if the demeaned form
            is requested with ``n <= ddof``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    _require_length(arr.size, 1, "realized variance")
    if zero_mean:
        return float(np.mean(arr ** 2))
    _require_length(arr.size, ddof + 1, f"demeaned variance with ddof={ddof}")
    return float(np.var(arr, ddof=ddof))


def close_to_close(close: pd.Series,
                   annualized: bool = True,
                   zero_mean: bool = False,
                   ddof: int = 1,
                   periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Close-to-close realized volatility.

    Formula (default, demeaned sample form)::

        r_i   = ln(C_i / C_{i-1})
        rbar  = (1 / n) * sum_i r_i
        sigma = sqrt( (1 / (n - ddof)) * sum_i (r_i - rbar)^2 )
        sigma_annualized = sigma * sqrt(252)

    With ``zero_mean=True`` the drift term is dropped::

        sigma = sqrt( (1 / n) * sum_i r_i^2 )

    Args:
        close: Close prices indexed by a monotonic ``DatetimeIndex``.
        annualized: Multiply by ``sqrt(periods_per_year)`` if True.
        zero_mean: Use the zero-drift realized-variance form.
        ddof: Degrees of freedom for the demeaned form. 1 gives the unbiased
            sample variance.
        periods_per_year: Annualization factor.

    Returns:
        Volatility as a decimal (0.45 == 45%).

    Raises:
        InsufficientDataError: If fewer than ``ddof + 2`` prices are supplied.
        SchemaValidationError: If any close price is non-positive.
    """
    rets = log_returns(close)
    var = realized_variance(rets, zero_mean=zero_mean, ddof=ddof)
    sigma = float(np.sqrt(var))
    return float(annualize(sigma, periods_per_year)) if annualized else sigma


def parkinson(ohlc: pd.DataFrame,
              annualized: bool = True,
              periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Parkinson (1980) high-low range volatility.

    Formula::

        sigma^2 = 1 / (4 * n * ln(2)) * sum_i [ ln(H_i / L_i) ]^2
        sigma_annualized = sqrt(sigma^2) * sqrt(252)

    The ``1 / (4 ln 2)`` factor is the scaling that makes the expected squared
    log-range an unbiased estimator of the diffusion variance for a driftless
    geometric Brownian motion.

    Caveat for earnings work, Parkinson sees only the intraday range and is
    blind to the overnight gap.  Since earnings are announced outside market
    hours, the bulk of the event move is an opening gap that this estimator
    does not observe, so it will understate event volatility. It is included
    as a clean diffusive-baseline estimator and for cross-checking, not as
    the event-window estimator.

    Args:
        ohlc: Frame with at least ``High`` and ``Low`` columns.
        annualized: Multiply by ``sqrt(periods_per_year)`` if True.
        periods_per_year: Annualization factor.

    Returns:
        Volatility as a decimal.

    Raises:
        SchemaValidationError: If ``High``/``Low`` are absent or non-positive.
        InsufficientDataError: If the frame has no usable rows.
    """
    _require_columns(ohlc, ("High", "Low"))
    high = ohlc["High"].to_numpy(dtype=float)
    low = ohlc["Low"].to_numpy(dtype=float)
    if np.nanmin(high) <= 0 or np.nanmin(low) <= 0:
        raise SchemaValidationError("High/Low prices must be strictly positive.")
    log_hl = np.log(high / low)
    log_hl = log_hl[~np.isnan(log_hl)]
    _require_length(log_hl.size, 1, "Parkinson volatility")
    var = float(np.sum(log_hl ** 2) / (4.0 * log_hl.size * np.log(2.0)))
    sigma = float(np.sqrt(var))
    return float(annualize(sigma, periods_per_year)) if annualized else sigma


def yang_zhang(ohlc: pd.DataFrame,
               annualized: bool = True,
               periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Yang-Zhang (2000) drift-independent, gap-aware volatility.

    The estimator is a convex combination of an overnight term, an
    open-to-close term and the Rogers-Satchell range term::

        o_i = ln(O_i / C_{i-1})            overnight return
        c_i = ln(C_i / O_i)                intraday return
        u_i = ln(H_i / O_i)
        d_i = ln(L_i / O_i)

        sigma_o^2  = 1/(n-1) * sum_i (o_i - obar)^2
        sigma_c^2  = 1/(n-1) * sum_i (c_i - cbar)^2
        sigma_rs^2 = 1/n     * sum_i [ u_i * (u_i - c_i) + d_i * (d_i - c_i) ]

        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        sigma^2 = sigma_o^2 + k * sigma_c^2 + (1 - k) * sigma_rs^2
        sigma_annualized = sqrt(sigma^2) * sqrt(252)

    ``k`` is chosen to minimize the estimator's variance.  Because the
    overnight term enters directly, this is the estimator to use when gaps
    carry real information, as they do around earnings.

    Note the ``n`` convention, ``n`` here is the number of sessions for which
    an overnight return exists, i.e. one fewer than the number of rows in
    ``ohlc``, since ``o_1`` requires the prior close.

    Args:
        ohlc: Frame with ``Open``, ``High``, ``Low``, ``Close`` columns.
        annualized: Multiply by ``sqrt(periods_per_year)`` if True.
        periods_per_year: Annualization factor.

    Returns:
        Volatility as a decimal.

    Raises:
        SchemaValidationError: If OHLC columns are absent or non-positive.
        InsufficientDataError: If fewer than 3 rows are supplied (the ``k``
            weight is undefined for ``n < 2``).
    """
    _require_columns(ohlc, _OHLC)
    _require_length(len(ohlc), 3, "Yang-Zhang volatility")

    o = ohlc["Open"].to_numpy(dtype=float)
    h = ohlc["High"].to_numpy(dtype=float)
    l = ohlc["Low"].to_numpy(dtype=float)
    c = ohlc["Close"].to_numpy(dtype=float)
    if min(np.nanmin(o), np.nanmin(h), np.nanmin(l), np.nanmin(c)) <= 0:
        raise SchemaValidationError("OHLC prices must be strictly positive.")

    # Drop the first row's contribution. It has no prior close.
    prev_close = c[:-1]
    o, h, l, c = o[1:], h[1:], l[1:], c[1:]

    overnight = np.log(o / prev_close)
    intraday = np.log(c / o)
    u = np.log(h / o)
    d = np.log(l / o)

    n = overnight.size
    _require_length(n, 2, "Yang-Zhang volatility (need n >= 2 overnight returns)")

    var_o = float(np.nanvar(overnight, ddof=1))
    var_c = float(np.nanvar(intraday, ddof=1))
    var_rs = float(np.nanmean(u * (u - intraday) + d * (d - intraday)))

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = var_o + k * var_c + (1.0 - k) * var_rs
    # Rogers-Satchell is non-negative in expectation but a small sample can
    # produce a marginally negative combination, clamp at zero rather than
    # returning a NaN from sqrt.
    sigma = float(np.sqrt(max(var, 0.0)))
    return float(annualize(sigma, periods_per_year)) if annualized else sigma


# --------------------------------------------------------------------------- #
# Rolling variants, fully vectorized, no Python-level loop over windows.
# --------------------------------------------------------------------------- #

def rolling_close_to_close(close: pd.Series,
                           window: int,
                           annualized: bool = True,
                           zero_mean: bool = False,
                           ddof: int = 1,
                           periods_per_year: int = TRADING_DAYS_PER_YEAR
                           ) -> pd.Series:
    """Rolling close-to-close volatility, dated on the last session of each window.

    Vectorized via :meth:`pandas.Series.rolling`, no per-window Python loop.

    Args:
        close: Close prices indexed by a monotonic ``DatetimeIndex``.
        window: Number of returns per window (trading days, not calendar days).
        annualized: Multiply by ``sqrt(periods_per_year)`` if True.
        zero_mean: Use the zero-drift realized-variance form.
        ddof: Degrees of freedom for the demeaned form.
        periods_per_year: Annualization factor.

    Returns:
        Volatility series, right-aligned, leading ``window-1`` entries NaN.

    Raises:
        ValueError: If ``window < 2``.
        InsufficientDataError: If fewer than 2 prices are supplied.
        SchemaValidationError: If any close price is non-positive.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    rets = log_returns(close)
    if zero_mean:
        var = rets.pow(2).rolling(window).mean()
    else:
        var = rets.rolling(window).var(ddof=ddof)
    sigma = np.sqrt(var)
    return annualize(sigma, periods_per_year) if annualized else sigma


def rolling_parkinson(ohlc: pd.DataFrame,
                      window: int,
                      annualized: bool = True,
                      periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Rolling Parkinson volatility, dated on the last session of each window.

    Formula per window, as in :func:`parkinson`::

        sigma^2 = 1 / (4 * ln(2)) * mean_i [ ln(H_i / L_i) ]^2

    Args:
        ohlc: Frame with ``High`` and ``Low`` columns.
        window: Number of sessions per window.
        annualized: Multiply by ``sqrt(periods_per_year)`` if True.
        periods_per_year: Annualization factor.

    Returns:
        Volatility series, right-aligned.

    Raises:
        ValueError: If ``window < 1``.
        SchemaValidationError: If ``High``/``Low`` are absent.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    _require_columns(ohlc, ("High", "Low"))
    log_hl_sq = np.log(ohlc["High"] / ohlc["Low"]) ** 2
    var = log_hl_sq.rolling(window).mean() / (4.0 * np.log(2.0))
    sigma = np.sqrt(var)
    return annualize(sigma, periods_per_year) if annualized else sigma


def rolling_yang_zhang(ohlc: pd.DataFrame,
                       window: int,
                       annualized: bool = True,
                       periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Rolling Yang-Zhang volatility, dated on the last session of each window.

    Same decomposition as :func:`yang_zhang`, evaluated on every trailing
    window of ``window`` sessions.  Implemented with three rolling reductions
    rather than a loop, so cost is linear in the length of the series.

    Args:
        ohlc: Frame with ``Open``, ``High``, ``Low``, ``Close`` columns.
        window: Number of overnight returns per window. Must be >= 2 for the
            ``k`` weight to be defined.
        annualized: Multiply by ``sqrt(periods_per_year)`` if True.
        periods_per_year: Annualization factor.

    Returns:
        Volatility series, right-aligned.

    Raises:
        ValueError: If ``window < 2``.
        SchemaValidationError: If OHLC columns are absent.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    _require_columns(ohlc, _OHLC)

    prev_close = ohlc["Close"].shift(1)
    overnight = np.log(ohlc["Open"] / prev_close)
    intraday = np.log(ohlc["Close"] / ohlc["Open"])
    u = np.log(ohlc["High"] / ohlc["Open"])
    d = np.log(ohlc["Low"] / ohlc["Open"])

    var_o = overnight.rolling(window).var(ddof=1)
    var_c = intraday.rolling(window).var(ddof=1)
    var_rs = (u * (u - intraday) + d * (d - intraday)).rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = var_o + k * var_c + (1.0 - k) * var_rs
    sigma = np.sqrt(var.clip(lower=0.0))
    return annualize(sigma, periods_per_year) if annualized else sigma


#: Registry mapping estimator name -> (point estimator, rolling estimator).
#: ``close_to_close`` takes a Series, the range estimators take a DataFrame.
ESTIMATORS: dict = {
    "close_to_close": (close_to_close, rolling_close_to_close),
    "parkinson": (parkinson, rolling_parkinson),
    "yang_zhang": (yang_zhang, rolling_yang_zhang),
}
