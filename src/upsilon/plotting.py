"""Plots: input variables, fit overlays and residuals/pulls.

Every function returns the matplotlib Figure; pass ``save="name.png"`` (or
``"small/name.png"``) to also write it into ``output/figures/``.

Style: mplhep's LHCb style (serif font, inward ticks on all sides), set on import so
the scripts' own plots match. Fit plots follow Fig. 5 of the LHCb momentum-scale
paper (arXiv:2312.01772): black data points, total fit in solid red, signal peaks in
dashed blue, background in dotted black, experiment label inside the top right.
These are not collaboration-approved plots, so the label says "LHCb Unofficial".
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from .fitting import FitResult
from .io import COLUMNS, FIGURE_DIR, LABELS

MASS_LABEL = LABELS["mass"]

hep.style.use("LHCb2")


def _experiment_label(ax, simulation: bool = False):
    """"LHCb Unofficial" (data: "2015 (13 TeV)") or "LHCb Simulation Unofficial" in the top left."""
    hep.lhcb.label("Unofficial", ax=ax, loc=1, data=not simulation, rlabel="(13 TeV)" if simulation else "2015 (13 TeV)")


def save_figure(fig, save: str | None):
    """Write ``fig`` to ``output/figures/<save>`` (e.g. ``"full/fit.png"``); no-op when ``save`` is None."""
    if save:
        path = FIGURE_DIR / save
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_variables(events: np.ndarray, bins: int = 200, ranges: dict | None = None, save: str | None = None):
    """Histogram each observable (task 1). ``ranges`` maps column -> (lo, hi)."""
    ranges = ranges or {}
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    for ax, col in zip(axes.flat, COLUMNS):
        ax.hist(events[col], bins=bins, range=ranges.get(col), histtype="step", color="k")
        ax.set(xlabel=LABELS[col], ylabel="Candidates / bin", title=f"Distribution of {LABELS[col]}")
    fig.tight_layout()
    return save_figure(fig, save)


def _binned(result: FitResult, bins: int):
    lo, hi = result.model.window
    edges = result.edges if result.edges is not None else np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(result.data, bins=edges)
    comps = result.model.component_counts(edges, result.values)
    expected = sum(comps.values())
    return edges, counts, comps, expected


def plot_fit(result: FitResult, bins: int = 250, pulls: bool = True, log: bool = False,
             title: str | None = "Mass fit", legend_title: str | None = None, simulation: bool = False,
             save: str | None = None):
    """Data with fitted model and its components; optional pull panel below.

    ``simulation=True`` labels the plot as LHCb simulation (fits to the MC sample).
    ``legend_title`` heads the legend, e.g. the kinematic cell of a fit in bins.
    """
    edges, counts, comps, expected = _binned(result, bins)
    centres = 0.5 * (edges[1:] + edges[:-1])

    if pulls:
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots()  # style default size

    data = ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="k.", ms=4)
    total, = ax.plot(centres, expected, color="red", lw=2)
    handles, labels = [data, total], ["Simulation" if simulation else "Data", "Total fit"]
    peaks = [k for k in comps if k != "bkg"]
    for k, c in comps.items():
        if k == "bkg":
            bkg, = ax.plot(centres, c, color="k", ls=":", lw=2)
        else:
            sig, = ax.plot(centres, c, color="blue", ls=(0, (8, 4)), lw=1.5)
    # One peak: name the decay, as in Fig. 4 of arXiv:2312.01772; several: one entry for all
    handles.append(sig)
    labels.append(rf"$\Upsilon({peaks[0]})\to\mu^+\mu^-$" if len(peaks) == 1 else r"$\Upsilon$ signal")
    if "bkg" in comps:
        handles.append(bkg)
        labels.append("Background")
    ax.set(ylabel=f"Candidates / ({1e3 * (edges[1] - edges[0]):.0f} MeV/$c^2$)", title=title or "")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.4 * counts.max())  # headroom for the label and legend, as in Fig. 5
    compact = {"fontsize": 0.75 * plt.rcParams["legend.fontsize"], "handlelength": 1.2, "labelspacing": 0.3} if len(peaks) == 1 else {}
    ax.legend(handles, labels, loc="upper right", title=legend_title, **compact)
    _experiment_label(ax, simulation)

    if pulls:
        plot_residuals(result, bins, kind="pull", ax=axp)
        axp.set_title("")
    else:
        ax.set_xlabel(MASS_LABEL)
    fig.tight_layout()
    return save_figure(fig, save)


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
    ax.errorbar(centres, y, yerr=yerr, fmt="k.", ms=4)
    ax.axhline(0, color="red")
    ax.set(xlabel=MASS_LABEL, ylabel=ylabel, title=f"{ylabel} per bin")
    return save_figure(fig, save)


def plot_profiles(results: dict[str, FitResult], name: str, ref: float = 0.0, scale: float = 1e3,
                  unit: str = "MeV/$c^2$", size: int = 31, xlabel: str | None = None, save: str | None = None):
    """Overlay profile likelihoods of parameter ``name`` from several fits.

    The x axis is ``(value - ref) * scale``, e.g. ``ref=PDG_MASS["1S"]`` for the mass offset in MeV.
    """
    from .compare import profile_scan

    fig, ax = plt.subplots(figsize=(16, 9))
    for label, r in results.items():
        x, d = profile_scan(r, name, size=size)
        ax.plot((x - ref) * scale, d, lw=2, label=label)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.5)
    ax.margins(x=0.03)
    ax.set(xlabel=xlabel or (f"{name} $-$ {ref:g} [{unit}]" if ref else f"{name} [{unit}]"),
           ylabel=r"$\Delta(-2\ln L)$", ylim=(0, 9))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=18)  # outside: curves span the width
    fig.tight_layout()
    return save_figure(fig, save)


def plot_error_scans(result: FitResult, names: tuple[str, ...], scale: float = 1e3, unit: str = "MeV/$c^2$",
                     save: str | None = None):
    """Delta(-2 ln L) scans of each parameter in ``names`` (brief task 2: errors by hand).

    Solid: profiled (others re-minimised, = MINOS); dashed: others fixed at the best fit.
    The 68% interval is where a curve crosses Delta = 1; the x axis is the offset from
    the best-fit value, times ``scale``.
    """
    from .compare import crossings, fixed_scan, profile_scan

    fig, axes = plt.subplots(1, len(names), figsize=(10 * len(names), 9))
    for ax, name in zip(np.atleast_1d(axes), names):
        best = result.values[name]
        for (x, d), style, label in ((profile_scan(result, name, size=41), "-", "profiled"),
                                     (fixed_scan(result, name), "--", "others fixed")):
            lo, hi = crossings(x, d)
            ax.plot((x - best) * scale, d, style, color="C0" if style == "-" else "C1",
                    label=rf"{label}: $^{{+{(hi - best) * scale:.3f}}}_{{-{(best - lo) * scale:.3f}}}$")
        err = result.errors[name] * scale
        ax.axvspan(-err, err, color="0.9", label=rf"HESSE $\pm{err:.3f}$")
        ax.axhline(1.0, color="k", lw=1, ls=":")
        ax.set(xlabel=rf"{name} $-$ best fit [{unit}]", ylabel=r"$\Delta(-2\ln L)$", ylim=(0, 4),
               title=f"Likelihood scan of {name}")
        ax.legend(fontsize=16)
    fig.tight_layout()
    return save_figure(fig, save)


def plot_vs_mass(events: np.ndarray, column: str, bins=(250, 100), range_=None, save: str | None = None):
    """2D histogram of ``column`` against dimuon mass (task 6)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(events["mass"], events[column], bins=bins, range=range_, cmap="viridis", norm="log")
    fig.colorbar(h[3], ax=ax, label="Candidates")
    ax.set(xlabel=MASS_LABEL, ylabel=LABELS[column], title=f"{LABELS[column]} vs dimuon mass")
    return save_figure(fig, save)
