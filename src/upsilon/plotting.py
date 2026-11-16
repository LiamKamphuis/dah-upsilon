"""Plots: input variables, fit overlays and residuals/pulls.

Every function returns the matplotlib Figure; pass ``save="name.png"`` to also
write it into ``output/figures/``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .fitting import FitResult
from .io import COLUMNS, FIGURE_DIR, LABELS

MASS_LABEL = LABELS["mass"]


def _save(fig, save):
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / save, dpi=150, bbox_inches="tight")
    return fig


def plot_variables(events: np.ndarray, bins: int = 200, ranges: dict | None = None, save: str | None = None):
    """Histogram each observable (task 1). ``ranges`` maps column -> (lo, hi)."""
    ranges = ranges or {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, COLUMNS):
        ax.hist(events[col], bins=bins, range=ranges.get(col), histtype="step", color="k")
        ax.set(xlabel=LABELS[col], ylabel="Candidates / bin", title=f"Distribution of {LABELS[col]}")
    fig.tight_layout()
    return _save(fig, save)


def _binned(result: FitResult, bins: int):
    lo, hi = result.model.window
    edges = result.edges if result.edges is not None else np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(result.data, bins=edges)
    comps = result.model.component_counts(edges, result.values)
    expected = sum(comps.values())
    return edges, counts, comps, expected


def plot_fit(result: FitResult, bins: int = 250, pulls: bool = True, log: bool = False,
             title: str | None = None, save: str | None = None):
    """Data with fitted model and its components; optional pull panel below."""
    edges, counts, comps, expected = _binned(result, bins)
    centres = 0.5 * (edges[1:] + edges[:-1])

    if pulls:
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="k.", ms=3, label="Data")
    ax.stairs(expected, edges, color="C0", lw=1.5, label="Total fit")
    for k, c in comps.items():
        ax.stairs(c, edges, ls="--", label=k)
    ax.set(ylabel=f"Candidates / ({1e3 * (edges[1] - edges[0]):.0f} MeV/$c^2$)", title=title or "Mass fit")
    if log:
        ax.set_yscale("log")
    ax.legend()

    if pulls:
        plot_residuals(result, bins, kind="pull", ax=axp)
        axp.set_title("")
    else:
        ax.set_xlabel(MASS_LABEL)
    fig.tight_layout()
    return _save(fig, save)


def plot_residuals(result: FitResult, bins: int = 250, kind: str = "residual", ax=None, save: str | None = None):
    """Residuals (data - model) or pulls ((data - model)/sqrt(model)) per bin (task 4)."""
    edges, counts, _, expected = _binned(result, bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    fig = ax.figure if ax is not None else plt.subplots(figsize=(9, 3.5))[0]
    ax = ax if ax is not None else fig.axes[0]

    if kind == "pull":
        y, yerr, ylabel = (counts - expected) / np.sqrt(expected), None, "Pull"
        ax.axhspan(-2, 2, color="0.9")
    else:
        y, yerr, ylabel = counts - expected, np.sqrt(expected), "Data $-$ fit"
    ax.errorbar(centres, y, yerr=yerr, fmt="k.", ms=3)
    ax.axhline(0, color="C0")
    ax.set(xlabel=MASS_LABEL, ylabel=ylabel, title=f"{ylabel} per bin")
    return _save(fig, save)


def plot_profiles(results: dict[str, FitResult], name: str, ref: float = 0.0, scale: float = 1e3,
                  unit: str = "MeV/$c^2$", size: int = 31, save: str | None = None):
    """Overlay profile likelihoods of parameter ``name`` from several fits.

    The x axis is ``(value - ref) * scale``, e.g. ``ref=PDG_MASS["1S"]`` for the mass offset in MeV.
    """
    from .compare import profile_scan

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, r in results.items():
        x, d = profile_scan(r, name, size=size)
        ax.plot((x - ref) * scale, d, label=label)
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.set(xlabel=f"{name} $-$ {ref:g} [{unit}]" if ref else f"{name} [{unit}]",
           ylabel=r"$\Delta(-2\ln L)$", ylim=(0, 9), title=f"Profile likelihood of {name}")
    ax.legend()
    return _save(fig, save)


def plot_vs_mass(events: np.ndarray, column: str, bins=(250, 100), range_=None, save: str | None = None):
    """2D histogram of ``column`` against dimuon mass (task 6)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(events["mass"], events[column], bins=bins, range=range_, cmap="viridis", norm="log")
    fig.colorbar(h[3], ax=ax, label="Candidates")
    ax.set(xlabel=MASS_LABEL, ylabel=LABELS[column], title=f"{LABELS[column]} vs dimuon mass")
    return _save(fig, save)
