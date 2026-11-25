"""Extended maximum-likelihood fits of a :class:`MassModel` with iminuit.

Unbinned is the default (uses every event's exact mass); binned is offered for
speed on the full dataset and as a cross-check, since the binning choice is
itself a potential source of bias.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, ExtendedUnbinnedNLL
from tqdm.auto import tqdm

from .io import rapidity
from .models import TAIL_PARAMS, MassModel

# Kinematic cells (brief task 7): the reference analysis's fiducial region 2.0 < y < 4.5,
# pT < 15 GeV/c, cut at round edges with at least ~2000 Upsilon(3S) in every cell.
Y_EDGES = (2.0, 3.0, 3.5, 4.5)
PT_EDGES = (0.0, 3.0, 6.0, 10.0, 15.0)


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
        if self.minuit.covariance is not None:
            free = [n for n, f in zip(self.minuit.parameters, self.minuit.fixed) if not f]
            corr = self.minuit.covariance.correlation()
            out["correlation"] = {a: {b: corr[a, b] for b in free} for a in free}
        if self.minuit.merrors:
            out["minos"] = {k: [me.lower, me.upper] for k, me in self.minuit.merrors.items()}
        return out


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

    ``start`` overrides starting values (see :func:`_starts` for the defaults); ``fix``
    fixes parameters to given values (e.g. tails from an MC fit). ``progress`` shows a
    live count of likelihood evaluations while Minuit runs.
    """
    x = _in_window(model, mass)
    if binned:
        cost, edges = _binned_cost(model, x, bins)
    else:
        cost, edges = ExtendedUnbinnedNLL(x, model), None
    m = _minuit(cost, {**_starts(model, len(x)), **(start or {})}, model.params, fix, minos, progress)
    return FitResult(model, m, x, binned, edges)


def fit_cells(
    models: list[MassModel],
    masses: list[np.ndarray],
    *,
    bins: int = 250,
    start: dict[str, float] | None = None,
    fix: dict[str, float] | None = None,
    progress: bool = True,
) -> list[FitResult]:
    """Simultaneous binned fit: the sum of each cell's extended binned -2 ln L.

    Parameters with the same name in several models are one parameter (see
    :meth:`MassModel.per_cell`). Returns one :class:`FitResult` per cell; they share the
    same Minuit object, so each cell's plots and goodness of fit work as for a single fit.
    """
    starts, params, cells = {}, {}, []
    for model, mass in zip(models, masses):
        x = _in_window(model, mass)
        cost, edges = _binned_cost(model, x, bins)
        cells.append((model, x, edges, cost))
        starts = {**_starts(model, len(x)), **starts}  # a shared parameter keeps its first start
        params.update(model.params)
    total = sum((cost for *_, cost in cells[1:]), cells[0][3])  # CostSum: equal names are shared
    m = _minuit(total, {**starts, **{k: v for k, v in (start or {}).items() if k in starts}},
                params, fix, False, progress)
    return [FitResult(model, m, x, True, edges) for model, x, edges, _ in cells]


def fiducial(events: np.ndarray, y_edges=Y_EDGES, pt_edges=PT_EDGES) -> np.ndarray:
    """Mask of events inside the outer edges of the kinematic cells."""
    y = rapidity(events)
    return (y > y_edges[0]) & (y < y_edges[-1]) & (events["pt"] < pt_edges[-1])


def fit_kinematic_cells(model: MassModel, events: np.ndarray, signal: str, reference: dict[str, float], *,
                        y_edges=Y_EDGES, pt_edges=PT_EDGES, bins: int = 250,
                        progress: bool = False) -> tuple[list[dict], list[FitResult]]:
    """One simultaneous fit of ``model`` in (y, pT) cells of the fiducial region.

    The masses and the tail shape (of ``signal``) are common to all cells; each cell has its
    own yields, sigma_1S and background. Shape and background start from ``reference``, the
    values of a fit to all cells together (from the default start the double CB can collapse
    into a false minimum); yields start from each cell's event count. Returns the cells
    (name, y and pT range, event count) and one :class:`FitResult` per cell.
    """
    y, pt = rapidity(events), events["pt"]
    keep = fiducial(events, y_edges, pt_edges)
    peaks = [k for k in model.components if k != "bkg"]
    common = {f"mu_{p}" for p in peaks} | {f"{t}_{peaks[0]}" for t in TAIL_PARAMS[signal]}
    cells, models, masses, start = [], [], [], {}
    for i, (ylo, yhi) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for j, (plo, phi) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
            name = f"y{i}pt{j}"
            sel = keep & (y >= ylo) & (y < yhi) & (pt >= plo) & (pt < phi)
            cells.append({"cell": name, "y": (ylo, yhi), "pt": (plo, phi), "events": int(sel.sum())})
            models.append(model.per_cell(name, common))
            masses.append(events["mass"][sel])
            for n, v in reference.items():
                if not n.startswith("N_"):
                    start[n if n in common else f"{n}_{name}"] = v
    return cells, fit_cells(models, masses, bins=bins, start=start, progress=progress)


def _in_window(model: MassModel, mass: np.ndarray) -> np.ndarray:
    lo, hi = model.window
    x = np.asarray(mass, dtype=float)
    return x[(x >= lo) & (x <= hi)]


def _binned_cost(model: MassModel, x: np.ndarray, bins: int):
    counts, edges = np.histogram(x, bins=bins, range=model.window)
    return ExtendedBinnedNLL(counts, edges, model.scaled_cdf, name=model.names), edges


def _starts(model: MassModel, n_events: int) -> dict[str, float]:
    """Shape parameters at their defaults; yields with 80% of the events in the background
    and the rest split over the peaks. An even split over all components lets wide signal
    shapes absorb the background (a false minimum)."""
    yields = [n for n in model.names if n.startswith("N_")]
    n_sig = sum(not n.startswith("N_bkg") for n in yields)
    bkg = 0.8 if len(yields) > n_sig else 0.0
    return {n: (n_events * (bkg if n.startswith("N_bkg") else (1 - bkg) / max(n_sig, 1))
                if n.startswith("N_") else p.start) for n, p in model.params.items()}


def _minuit(cost, starts, params, fix, minos, progress) -> Minuit:
    """Minuit on ``cost`` with limits from ``params`` (set explicitly: the binned cost cannot
    read them from ``scaled_cdf``), ``fix`` applied, minimised by :func:`_minimise`."""
    counted = _CountedCost(cost)
    m = Minuit(counted, **{n: {**starts, **(fix or {})}[n] for n in cost._parameters})
    for n, p in params.items():
        lo, hi = p.limits
        m.limits[n] = (-np.inf if lo is None else lo, np.inf if hi is None else hi)
    for n in fix or {}:
        m.fixed[n] = True
    _minimise(m, counted, minos, progress)
    _warn_unconstrained(m)
    return m


def _minimise(m: Minuit, counted: "_CountedCost", minos: bool, progress: bool) -> None:
    """MIGRAD + HESSE, with the restarts below when the minimum is not converged."""
    with tqdm(desc="migrad", unit=" calls", disable=not progress) as counted.bar:
        m.migrad()
        counted.bar.set_description_str("hesse")
        m.hesse()
        if not _converged(m):
            # Large-N unbinned fits: MIGRAD's approximate covariance can declare
            # convergence that the exact HESSE matrix then puts above the EDM goal.
            # Restarting MIGRAD from the HESSE matrix converges.
            counted.bar.set_description_str("migrad (restart)")
            m.migrad()
            counted.bar.set_description_str("hesse")
            m.hesse()
        if not _converged(m):
            # A start far from the minimum can leave MIGRAD stuck in a bad region (tied-width
            # Crystal Ball on full data: n -> 1.5 instead of 8.7). SIMPLEX needs no
            # derivatives and gets out; MIGRAD then converges from there.
            counted.bar.set_description_str("simplex")
            m.simplex()
            counted.bar.set_description_str("migrad (after simplex)")
            m.migrad()
            counted.bar.set_description_str("hesse")
            m.hesse()
        if minos:
            counted.bar.set_description_str("minos")
            m.minos()
        counted.bar.set_description_str("fit done")
    counted.bar = None  # later use of result.minuit (profiles, contours) runs without a bar


def _converged(m: Minuit) -> bool:
    """Valid minimum, a covariance HESSE did not force positive-definite, and no parameter
    pinned at a limit.

    The last two catch false minima Minuit still calls valid: a double Crystal Ball that
    collapses to one Crystal Ball (ratio -> its limit 1, frac meaningless). Seen at a 3 GeV
    softer-muon cut (-2 ln L worse by 649) and on the fiducial sample (worse by 548). A
    parameter legitimately at a limit only costs one extra SIMPLEX + MIGRAD pass.
    """
    return m.valid and m.fmin.has_accurate_covar and not m.fmin.has_parameters_at_limit


def _warn_unconstrained(m: Minuit) -> None:
    """Warn about free parameters whose error exceeds half their allowed range.

    Such a parameter barely changes the likelihood (e.g. a Crystal Ball ``n`` once
    ``alpha`` has run off the window), which makes the HESSE matrix near-singular:
    Minuit can still report a valid fit while every error, even the yields', is garbage.
    """
    bad = [p.name for p in m.params
           if not p.is_fixed and p.lower_limit is not None and p.upper_limit is not None
           and p.error > 0.5 * (p.upper_limit - p.lower_limit)]
    if bad:
        warnings.warn(f"parameter(s) {bad} unconstrained by the data (error > half their range): "
                      "the covariance and all errors are unreliable; fix or share them", stacklevel=3)


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
