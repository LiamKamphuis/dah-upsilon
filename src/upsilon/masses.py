"""Mass differences, momentum-scale calibration and signal-shape bias.

The fitted masses sit a few MeV below PDG. Two effects can do that:

- a *momentum scale* error multiplies every mass by the same factor s,
  so it cancels in ratios and almost cancels in differences;
- a *shape bias* b: how far a model's mu lands from the true mass because of how
  it handles the radiative tail. Measured on simulation, where the true mass is known.

Everything works on the vector of the three masses and its 3x3 covariance
(order ``PEAKS``), so the correlations between peaks carry through.

TODO (you): ``mass_difference``, ``scale_corrected_masses``, ``common_scale``.
Each has a test in ``tests/test_masses.py`` marked xfail until it is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting import FitResult, fit
from .models import PDG_MASS, build_model

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
    return np.array([result.values[f"sigma_{p}"] for p in peaks])


def shape_bias(signal: str, mc_mass: np.ndarray, m_true: float = PDG_MASS["1S"],
               window: tuple[float, float] = (9.0, 9.9)) -> Bias:
    """Fit simulation (pure Upsilon(1S)) with ``signal`` and return mu_fit - m_true.

    ``m_true`` is the mass the simulation was *generated* with. We assume PDG;
    change it here if you find out otherwise.
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


# --- TODO: your error propagation ------------------------------------------------

def mass_difference(mu: np.ndarray, cov: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """mu[i] - mu[j] and its error, including the correlation between the two masses."""
    raise NotImplementedError


def scale_corrected_masses(mu: np.ndarray, cov: np.ndarray, ref: int = 0,
                           m_ref: float = PDG_MASS["1S"]) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the momentum scale on peak ``ref``: s = mu[ref] / m_ref, return (mu / s, covariance).

    The corrected ``ref`` mass is m_ref by construction, with zero error. The others
    inherit the error on s, which is correlated with their own fitted mass.
    """
    raise NotImplementedError


def common_scale(mu: np.ndarray, cov: np.ndarray, m_pdg: np.ndarray) -> ScaleFit:
    """Best single scale s with mu_k = s * m_pdg_k for all peaks, and how well it fits.

    Minimise chi2(s) = (r - s)^T V^-1 (r - s), with r_k = mu_k / m_pdg_k and V the
    covariance of r. ndof = number of peaks - 1. A small p-value means one momentum
    scale cannot explain all the offsets.
    """
    raise NotImplementedError
