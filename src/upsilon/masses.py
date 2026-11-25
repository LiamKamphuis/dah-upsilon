"""Mass differences, momentum-scale calibration and signal-shape bias.

The fitted masses sit a few MeV below PDG. Two effects can do that:

- a *momentum scale* error multiplies every mass by the same factor s,
  so it cancels in ratios and almost cancels in differences;
- a *shape bias* b: how far a model's mu lands from the true mass because of how
  it handles the radiative tail. Measured on simulation, where the true mass is known.

Everything works on the vector of the three masses and its 3x3 covariance
(order ``PEAKS``), so the correlations between peaks carry through.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats

from .fitting import FitResult, fit
from .models import MC_MASS, PDG_MASS, build_model

PEAKS = ("1S", "2S", "3S")


@dataclass
class Bias:
    value: float  # mu_fit(MC) - m_true [GeV]
    error: float
    sigma: float  # MC core width, to scale the bias to other peaks / data


@dataclass
class ScaleFit:
    s: float
    error: float
    chi2: float
    ndof: int
    pvalue: float


def mass_vector(result: FitResult, peaks: tuple[str, ...] = PEAKS) -> tuple[np.ndarray, np.ndarray]:
    """Fitted masses and their covariance (HESSE) from a fit."""
    names = [f"mu_{p}" for p in peaks]
    mu = np.array([result.values[n] for n in names])
    cov = np.array([[result.minuit.covariance[a, b] for b in names] for a in names])
    return mu, cov


def widths(result: FitResult, peaks: tuple[str, ...] = PEAKS) -> np.ndarray:
    """Each peak's core width (scaled from sigma_1S when the widths are tied)."""
    return np.array([result.model.sigma(p, result.values) for p in peaks])


def effective_widths(results: list[FitResult], peaks: tuple[str, ...] = PEAKS) -> np.ndarray:
    """Each peak's width as seen by a simultaneous fit in cells: sum_c w_c sigma_c / sum_c w_c.

    A shared mass is a weighted mean of what each cell measures, with weight w_c = N_c / sigma_c^2
    (the information on a peak position from N_c signal events of width sigma_c; background
    ignored). A bias proportional to the width (see ``apply_bias``) averages the same way, so
    this width in ``apply_bias`` gives the bias of the shared mass.
    """
    out = []
    for p in peaks:
        n = np.array([r.values[next(k for k in r.model.names if k.startswith(f"N_{p}"))] for r in results])
        s = np.array([r.model.sigma(p, r.values) for r in results])
        w = n / s**2
        out.append(np.sum(w * s) / np.sum(w))
    return np.array(out)


def shape_bias(signal: str, mc_mass: np.ndarray, m_true: float = MC_MASS["1S"],
               window: tuple[float, float] = (9.0, 9.9)) -> Bias:
    """Fit simulation (pure Upsilon(1S)) with ``signal`` and return mu_fit - m_true.

    ``m_true`` is the mass the simulation was *generated* with: the PDG 2016 value
    (``MC_MASS``), not today's ``PDG_MASS``, which is 0.1 MeV higher for the 1S.
    """
    model = build_model(("1S",), signal=signal, window=window, background=None)
    r = fit(model, mc_mass, progress=False)
    return Bias(r.values["mu_1S"] - m_true, r.errors["mu_1S"], r.values["sigma_1S"])


def apply_bias(mu: np.ndarray, cov: np.ndarray, bias: Bias, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the shape bias, scaled to each peak's width: b_k = b * sigma_k / sigma_MC.

    The scaling assumes the peak shape is the same in units of its width, so the
    bias is a fixed fraction of the resolution. All three corrections come from one MC
    number, so their errors are fully correlated: cov += k k^T * error^2, k = sigma/sigma_MC.
    (The uncertainty on sigma_k itself is neglected.)
    """
    k = np.asarray(sigma) / bias.sigma
    return mu - bias.value * k, cov + np.outer(k, k) * bias.error**2


def mass_difference(mu: np.ndarray, cov: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """Difference in mass peak fits and its error, including the correlation between the two masses."""
    return mu[i] - mu[j], np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j])


def scale_corrected_masses(mu: np.ndarray, cov: np.ndarray, ref: int = 0,
                           m_ref: float = PDG_MASS["1S"]) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the momentum scale on peak ``ref``: s = mu[ref] / m_ref, return (mu / s, covariance).

    The corrected ``ref`` mass is m_ref by construction, with zero error. The others
    inherit the error on s, which is correlated with their own fitted mass.
    """
    s = mu[ref] / m_ref
    J = np.eye(len(mu)) / s
    J[:, ref] = -mu / (s * mu[ref])
    J[ref, :] = 0
    return mu / s, J @ cov @ J.T


def common_scale(mu: np.ndarray, cov: np.ndarray, m_pdg: np.ndarray) -> ScaleFit:
    """Best single scale s with mu_k = s * m_pdg_k for all peaks, and how well it fits.

    Minimize chi2(s) = (r - s)^T V^-1 (r - s), with r_k = mu_k / m_pdg_k and V the
    covariance of r. ndof = number of peaks - 1. A small p-value means one momentum
    scale cannot explain all the offsets.
    """
    r = mu / m_pdg
    V = cov / np.outer(m_pdg, m_pdg)
    w = np.linalg.solve(V, np.ones(len(r)))
    s = (w @ r) / w.sum()
    error = np.sqrt(1 / w.sum())
    d = r - s
    chi2 = d @ np.linalg.solve(V, d)
    ndof = len(mu) - 1
    pvalue = scipy.stats.chi2.sf(chi2, ndof)
    return ScaleFit(s, error, chi2, ndof, pvalue)
