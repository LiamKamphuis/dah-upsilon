"""Extended maximum-likelihood fits of a :class:`MassModel` with iminuit.

Unbinned is the default (uses every event's exact mass); binned is offered for
speed on the full dataset and as a cross-check, since the binning choice is
itself a potential source of bias.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, ExtendedUnbinnedNLL
from tqdm.auto import tqdm

from .models import MassModel


@dataclass
class FitResult:
    model: MassModel
    minuit: Minuit
    data: np.ndarray  # masses inside the window
    binned: bool
    edges: np.ndarray | None = None

    @property
    def values(self) -> dict[str, float]:
        return dict(zip(self.minuit.parameters, self.minuit.values))

    @property
    def errors(self) -> dict[str, float]:
        """Symmetric (HESSE) errors; 0 for fixed parameters (Minuit keeps only a step size there)."""
        return {n: 0.0 if fixed else e
                for n, e, fixed in zip(self.minuit.parameters, self.minuit.errors, self.minuit.fixed)}

    @property
    def valid(self) -> bool:
        return self.minuit.valid

    @property
    def nll(self) -> float:
        return self.minuit.fval

    def summary(self) -> str:
        lines = [f"{'parameter':<12}{'value':>14}{'error':>14}", "-" * 40]
        for n, v, e, fixed in zip(self.minuit.parameters, self.minuit.values, self.minuit.errors, self.minuit.fixed):
            lines.append(f"{n:<12}{v:>14.6g}" + (f"{'(fixed)':>14}" if fixed else f"{e:>14.3g}"))
        lines.append(f"valid={self.valid}  fval (-2 lnL + const)={self.nll:.3f}  ({'binned' if self.binned else 'unbinned'})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        out = {"values": self.values, "errors": self.errors, "valid": self.valid,
               "fval": self.nll, "binned": self.binned, "model": repr(self.model)}
        if self.minuit.merrors:
            out["minos"] = {k: [me.lower, me.upper] for k, me in self.minuit.merrors.items()}
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def fit(
    model: MassModel,
    mass: np.ndarray,
    *,
    binned: bool = False,
    bins: int = 250,
    start: dict[str, float] | None = None,
    fix: dict[str, float] | None = None,
    minos: bool = False,
    progress: bool = True,
) -> FitResult:
    """Fit ``model`` to the masses inside its window.

    ``start`` overrides starting values; ``fix`` fixes parameters to given
    values (e.g. Crystal Ball tails from an MC fit). Unless given, yields start
    with ``bkg_fraction`` of the events in background and the rest split evenly
    over the signal peaks: an even split over all components lets wide signal
    Gaussians absorb the background (a false minimum). ``progress`` shows a
    live count of likelihood evaluations while Minuit runs.
    """
    lo, hi = model.window
    x = np.asarray(mass, dtype=float)
    x = x[(x >= lo) & (x <= hi)]

    starts = {n: p.start for n, p in model.params.items()}
    yields = [n for n in starts if n.startswith("N_")]
    n_sig = sum(n != "N_bkg" for n in yields)
    bkg_fraction = 0.8 if "N_bkg" in yields else 0.0
    for n in yields:
        starts[n] = len(x) * (bkg_fraction if n == "N_bkg" else (1 - bkg_fraction) / max(n_sig, 1))
    starts.update(start or {})
    starts.update(fix or {})

    edges = None
    if binned:
        counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
        cost = ExtendedBinnedNLL(counts, edges, model.scaled_cdf, name=model.names)
    else:
        cost = ExtendedUnbinnedNLL(x, model)

    counted = _CountedCost(cost)
    m = Minuit(counted, **starts)
    for n, p in model.params.items():  # set explicitly: the binned cost cannot read limits from scaled_cdf
        lo_lim, hi_lim = p.limits
        m.limits[n] = (-np.inf if lo_lim is None else lo_lim, np.inf if hi_lim is None else hi_lim)
    for n in fix or {}:
        m.fixed[n] = True

    with tqdm(desc="migrad", unit=" calls", disable=not progress) as counted.bar:
        m.migrad()
        counted.bar.set_description_str("hesse")
        m.hesse()
        if not m.valid:
            # Large-N unbinned fits: MIGRAD's approximate covariance can declare
            # convergence that the exact HESSE matrix then puts above the EDM goal.
            # Restarting MIGRAD from the HESSE matrix converges.
            counted.bar.set_description_str("migrad (restart)")
            m.migrad()
            counted.bar.set_description_str("hesse")
            m.hesse()
        if minos:
            counted.bar.set_description_str("minos")
            m.minos()
        counted.bar.set_description_str("fit done")
    counted.bar = None  # later use of result.minuit (profiles, contours) runs without a bar
    return FitResult(model, m, x, binned, edges)


class _CountedCost:
    """Wraps an iminuit cost to tick a progress bar on every evaluation.

    MIGRAD's number of calls is not known in advance, so the bar counts calls
    (with rate and elapsed time) rather than showing a percentage. All other
    attributes (errordef, parameter names/limits, ndata) pass through to the cost.
    """

    def __init__(self, cost):
        self.cost = cost
        self.bar = None

    def __call__(self, *par):
        if self.bar is not None:
            self.bar.update()
        return self.cost(*par)

    def __getattr__(self, name):
        return getattr(self.cost, name)
