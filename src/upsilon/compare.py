"""Goodness of fit and model comparison for :class:`FitResult` s.

- :func:`gof`: Baker-Cousins likelihood-ratio chi2 of the binned data against the
  fitted model, and its p-value. The main "does this model describe the data" test.
- :func:`runs_test`: Wald-Wolfowitz runs test on the signs of the bin residuals.
  Sensitive to *structured* misfit (e.g. model low in the tail, high in the core)
  that chi2 dilutes over many bins.
- :func:`likelihood_ratio`: Wilks test between two *nested* fits to the same data.
- :func:`information_criteria`: AIC and BIC; only differences between fits to the
  same data mean anything. Use for non-nested pairs (Crystal Ball vs double Gaussian).
- :func:`table`: all of the above for several fits, for picking a baseline.

``FitResult.nll`` is iminuit's ``fval`` = -2 ln L + const, where the constant depends
only on the data (and binning), so differences between fits on the same data are
exact -2 Delta ln L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .fitting import FitResult


@dataclass
class TestResult:
    stat: float
    ndof: int
    pvalue: float


@dataclass
class Runs:
    runs: int
    n_pos: int
    n_neg: int
    expected: float
    variance: float
    pvalue: float  # one-sided: probability of this few runs or fewer


@dataclass
class InformationCriteria:
    k: int  # free parameters
    n: int  # events in the window
    m2lnl: float  # -2 ln L + const (FitResult.nll)
    aic: float
    bic: float


def n_free(result: FitResult) -> int:
    """Number of floating parameters (shared parameters count once, fixed ones not at all)."""
    return sum(not f for f in result.minuit.fixed)


def _counts_and_expected(result: FitResult, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Observed and expected counts per bin; a binned fit's own edges take precedence over ``bins``."""
    lo, hi = result.model.window
    edges = result.edges if result.edges is not None else np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(result.data, bins=edges)
    expected = sum(result.model.component_counts(edges, result.values).values())
    return counts, expected


def gof(result: FitResult, bins: int = 250) -> TestResult:
    """Baker-Cousins chi2 = 2 sum[mu - n + n ln(n/mu)], with ndof = nbins - free parameters.

    Asymptotically chi2(ndof) when expected counts per bin are large (>~ 5).
    For a binned fit this equals ``result.nll``. For an unbinned fit the parameters
    did not minimise this statistic, so it sits slightly above the binned minimum
    and the p-value is a little conservative.
    """
    n, mu = _counts_and_expected(result, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(n > 0, n * np.log(n / mu), 0.0)
    stat = float(2 * np.sum(mu - n + log_term))
    ndof = len(n) - n_free(result)
    return TestResult(stat, ndof, float(stats.chi2.sf(stat, ndof)))


def _runs(residuals: np.ndarray) -> Runs:
    signs = np.sign(residuals)
    signs = signs[signs != 0]
    n_pos, n_neg = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    n = n_pos + n_neg
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    expected = 2 * n_pos * n_neg / n + 1
    variance = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n) / (n**2 * (n - 1))
    pvalue = float(stats.norm.cdf((runs - expected) / np.sqrt(variance)))
    return Runs(runs, n_pos, n_neg, expected, variance, pvalue)


def runs_test(result: FitResult, bins: int = 250) -> Runs:
    """Runs of same-sign residuals across the mass window.

    A good model gives signs that alternate at random; too few runs (long stretches
    above or below the data) means the shape is wrong somewhere. Normal approximation,
    fine for tens of bins or more. Independent of chi2, so the two complement each other.
    """
    n, mu = _counts_and_expected(result, bins)
    return _runs(n - mu)


def likelihood_ratio(simple: FitResult, complex_: FitResult) -> TestResult:
    """-2 Delta ln L between nested fits, p-value from chi2(Delta k) (Wilks).

    Caveat: when the simple model sits on the *boundary* of the complex one
    (double-Gaussian frac -> 0, Crystal Ball alpha -> inf) Wilks' conditions fail and
    the chi2 p-value is only approximate; calibrate with toys if it matters.
    """
    if (simple.binned != complex_.binned or len(simple.data) != len(complex_.data)
            or simple.model.window != complex_.model.window
            or (simple.binned and not np.array_equal(simple.edges, complex_.edges))):
        raise ValueError("fits must use the same data, window and binning")
    ndof = n_free(complex_) - n_free(simple)
    if ndof <= 0:
        raise ValueError(f"complex model must have more free parameters (Delta k = {ndof})")
    stat = simple.nll - complex_.nll
    return TestResult(stat, ndof, float(stats.chi2.sf(max(stat, 0.0), ndof)))


def information_criteria(result: FitResult) -> InformationCriteria:
    """AIC = -2lnL + 2k, BIC = -2lnL + k ln N, with N the number of events (also for binned fits)."""
    k, n = n_free(result), len(result.data)
    return InformationCriteria(k, n, result.nll, result.nll + 2 * k, result.nll + k * np.log(n))


def table(results: dict[str, FitResult], bins: int = 250) -> str:
    """One row per fit: k, chi2/ndof and p, runs p, and AIC/BIC relative to the best fit.

    All fits must be to the same data in the same window for the Delta columns to mean anything.
    """
    rows = {name: (information_criteria(r), gof(r, bins), runs_test(r, bins)) for name, r in results.items()}
    aic0 = min(ic.aic for ic, _, _ in rows.values())
    bic0 = min(ic.bic for ic, _, _ in rows.values())
    head = f"{'model':<28}{'k':>4}{'chi2/ndof':>16}{'p(chi2)':>11}{'p(runs)':>11}{'dAIC':>10}{'dBIC':>10}"
    lines = [head, "-" * len(head)]
    for name, (ic, g, ru) in rows.items():
        lines.append(f"{name:<28}{ic.k:>4}{f'{g.stat:.1f}/{g.ndof}':>16}{g.pvalue:>11.3g}{ru.pvalue:>11.3g}"
                     f"{ic.aic - aic0:>10.1f}{ic.bic - bic0:>10.1f}")
    return "\n".join(lines)
