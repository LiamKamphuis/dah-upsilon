"""Mass models: shapes normalized on the fit window, their extended sum, and a builder.

A :class:`Shape` is a 1D PDF normalized to unit area on ``[lo, hi]``.
Normalizing each component on the window (not the real line) keeps the
composite PDF normalized for any parameters, as the extended likelihood
requires. Shapes provide ``pdf`` (unbinned fits) and ``cdf`` (binned fits and
expected bin contents for plots).

:class:`MassModel` is ``f(x) = sum_k N_k * shape_k(x)``. Component ``k``'s
parameters are named ``<param>_<k>`` (``mu_1S``, ``c_bkg``); ``share`` makes
parameters common across components, e.g. ``{"alpha_2S": "alpha_1S"}``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
from particle import Particle
from scipy import stats

Window = tuple[float, float]
Limits = tuple[float | None, float | None]

#: PDG Monte Carlo IDs of the Upsilon(nS) states, used as starting vals.
PDG_ID = {"1S": 553, "2S": 100553, "3S": 200553}
PDG_MASS = {peak: Particle.from_pdgid(pid).mass / 1e3 for peak, pid in PDG_ID.items()}
#: Masses the simulation was generated with, in GeV/c^2: the PDG values of its time.
#: C. Patrignani et al. (Particle Data Group), Chin. Phys. C 40, 100001 (2016), listings:
#: 9460.30 +- 0.26, 10023.26 +- 0.31, 10355.2 +- 0.5 MeV (1S identical in the 2014 edition).
#: Which edition the generator used is our assumption (2015 data, 2015/16 simulation).
MC_MASS = {"1S": 9.46030, "2S": 10.02326, "3S": 10.3552}


@dataclass(frozen=True)
class Param:
    """A shape parameter: local name, starting value and limits."""

    name: str
    start: float
    limits: Limits = (None, None)


class Shape(ABC):
    """A PDF on ``window`` with parameters ``PARAMS`` (in positional order).

    Starting values and limits can be overridden per instance, e.g.
    ``Gaussian(window, mu=10.02, limits={"mu": (9.9, 10.1)})``.
    """

    PARAMS: tuple[Param, ...] = ()

    def __init__(self, window: Window, limits: dict[str, Limits] | None = None, **starts: float):
        self.window = (float(window[0]), float(window[1]))
        unknown = (set(starts) | set(limits or {})) - {p.name for p in self.PARAMS}
        if unknown:
            raise ValueError(f"{type(self).__name__} has no parameter(s) {sorted(unknown)}")
        self.params = tuple(
            replace(p, start=starts.get(p.name, p.start), limits=(limits or {}).get(p.name, p.limits))
            for p in self.PARAMS
        )

    @abstractmethod
    def pdf(self, x: np.ndarray, *p: float) -> np.ndarray:
        """Density at ``x``, integrating to 1 over the window."""

    @abstractmethod
    def cdf(self, x: np.ndarray, *p: float) -> np.ndarray:
        """Integral of ``pdf`` from ``lo`` to ``x`` (so ``cdf(hi) == 1``)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{p.name}={p.start:g}' for p in self.params)})"


# --- signal shapes -------------------------------------------------------------

class Gaussian(Shape):
    """Gaussian truncated to the fit window."""

    PARAMS = (
        Param("mu", 9.46, (8.5, 11.0)),
        Param("sigma", 0.04, (5e-3, 0.2))
    )

    def _norm(self, mu, sigma):
        lo, hi = self.window
        return stats.norm.cdf(hi, mu, sigma) - stats.norm.cdf(lo, mu, sigma)

    def pdf(self, x, mu, sigma):
        return stats.norm.pdf(x, mu, sigma) / self._norm(mu, sigma)

    def cdf(self, x, mu, sigma):
        lo = self.window[0]
        return (stats.norm.cdf(x, mu, sigma) - stats.norm.cdf(lo, mu, sigma)) / self._norm(mu, sigma)


class DoubleGaussian(Shape):
    """Narrow + wide Gaussian sharing a mean.

    ``ratio = sigma_wide / sigma`` (>1) and ``frac`` (narrow fraction). A ratio
    rather than two free widths reduces their correlation and stops the
    components swapping roles: worth discussing as a model choice.

    Each component is truncated and normalized on the window separately, so
    ``frac`` is the narrow fraction *inside the window*, not on the real line.
    """

    PARAMS = (
        Param("mu", 9.46, (8.5, 11.0)),
        Param("sigma", 0.04, (5e-3, 0.2)),
        Param("ratio", 2.0, (1.0, 10.0)),
        Param("frac", 0.7, (0.0, 1.0)),
    )

    def __init__(self, window: Window, limits: dict[str, Limits] | None = None, **starts: float):
        super().__init__(window, limits, **starts)
        self._gauss = Gaussian(window)

    def pdf(self, x, mu, sigma, ratio, frac):
        return frac * self._gauss.pdf(x, mu, sigma) + (1.0 - frac) * self._gauss.pdf(x, mu, ratio * sigma)

    def cdf(self, x, mu, sigma, ratio, frac):
        return frac * self._gauss.cdf(x, mu, sigma) + (1.0 - frac) * self._gauss.cdf(x, mu, ratio * sigma)


class CrystalBall(Shape):
    """Gaussian core with a low-mass power-law tail from final-state radiation.

    ``alpha`` is where the tail starts, in units of ``sigma`` below ``mu``, and
    ``n`` is the power-law exponent (scipy's ``beta`` and ``m``). The shape is
    truncated and normalized on the window via scipy's cdf. ``alpha`` and ``n`` are
    strongly anti-correlated (a tail starting nearer the core but falling faster looks
    alike), and on data the tail also trades against the background under it.
    """

    PARAMS = (
        Param("mu", 9.46, (8.5, 11.0)),
        Param("sigma", 0.04, (5e-3, 0.2)),
        Param("alpha", 1.5, (0.1, 10.0)),
        Param("n", 3.0, (1.01, 50.0)),
    )

    def _norm(self, mu, sigma, alpha, n):
        lo, hi = self.window
        return (
            stats.crystalball.cdf(hi, alpha, n, loc=mu, scale=sigma)
            - stats.crystalball.cdf(lo, alpha, n, loc=mu, scale=sigma)
        )

    def pdf(self, x, mu, sigma, alpha, n):
        return stats.crystalball.pdf(x, alpha, n, loc=mu, scale=sigma) / self._norm(mu, sigma, alpha, n)

    def cdf(self, x, mu, sigma, alpha, n):
        lo = self.window[0]
        return (
            stats.crystalball.cdf(x, alpha, n, loc=mu, scale=sigma)
            - stats.crystalball.cdf(lo, alpha, n, loc=mu, scale=sigma)
        ) / self._norm(mu, sigma, alpha, n)


class DoubleCrystalBall(Shape):
    """Two Crystal Balls sharing ``mu``, ``alpha`` and ``n``, with widths ``sigma`` and
    ``ratio * sigma`` and narrow fraction ``frac``.

    Why: the mass resolution varies with the kinematics (sigma_1S ~ 32-65 MeV across the
    (pT, y) cells of scripts/04_kinematic_bins.py), so the inclusive peak is a blend of widths:
    a sharper core and heavier wings than one Crystal Ball. The LHCb paper uses this shape
    for its mass-model systematic.

    ``alpha`` is in units of each component's own sigma, so the two tails start at
    different masses (a modelling choice; an absolute tail start is the alternative).
    """

    PARAMS = (
        Param("mu", 9.46, (8.5, 11.0)),
        Param("sigma", 0.035, (5e-3, 0.2)),
        Param("alpha", 1.5, (0.1, 10.0)),
        Param("n", 3.0, (1.01, 50.0)),
        Param("ratio", 1.6, (1.0, 5.0)),
        Param("frac", 0.5, (0.0, 1.0)),
    )

    def __init__(self, window: Window, limits: dict[str, Limits] | None = None, **starts: float):
        super().__init__(window, limits, **starts)
        self._cb = CrystalBall(window)

    def pdf(self, x, mu, sigma, alpha, n, ratio, frac):
        return frac * self._cb.pdf(x, mu, sigma, alpha, n) + (1.0 - frac) * self._cb.pdf(x, mu, ratio * sigma, alpha, n)

    def cdf(self, x, mu, sigma, alpha, n, ratio, frac):
        return frac * self._cb.cdf(x, mu, sigma, alpha, n) + (1.0 - frac) * self._cb.cdf(x, mu, ratio * sigma, alpha, n)


# --- background shapes ---------------------------------------------------------

class Exponential(Shape):
    """Falling exponential ``exp(-c (x - lo))`` normalized on the window.

    The ``c -> 0`` (flat) limit is handled with ``expm1`` for stability.
    """

    PARAMS = (Param("c", 0.5, (-20.0, 20.0)),)

    def _x(self, x):
        return np.asarray(x, dtype=float) - self.window[0]

    def pdf(self, x, c):
        width = self.window[1] - self.window[0]
        if abs(c) < 1e-9:
            return np.full_like(self._x(x), 1.0 / width)
        return -c * np.exp(-c * self._x(x)) / np.expm1(-c * width)

    def cdf(self, x, c):
        width = self.window[1] - self.window[0]
        if abs(c) < 1e-9:
            return self._x(x) / width
        return np.expm1(-c * self._x(x)) / np.expm1(-c * width)


class Quadratic(Shape):
    """Second-order polynomial ``1 + a1 P1(t) + a2 P2(t)`` in Legendre polynomials of
    t = 2 (x - lo) / (hi - lo) - 1, normalized on the window.

    The alternative background for the systematic: it can curve where the exponential
    cannot. P1 and P2 integrate to zero on [-1, 1], so the normalization does not depend
    on a1, a2 (but the density goes negative if they are too large: check the fit).
    """

    PARAMS = (Param("a1", -0.3, (-1.0, 1.0)), Param("a2", 0.0, (-1.0, 1.0)))

    def _t(self, x):
        lo, hi = self.window
        return 2 * (np.asarray(x, dtype=float) - lo) / (hi - lo) - 1

    def pdf(self, x, a1, a2):
        t = self._t(x)
        return (1 + a1 * t + a2 * 0.5 * (3 * t**2 - 1)) / (self.window[1] - self.window[0])

    def cdf(self, x, a1, a2):
        t = self._t(x)
        return 0.5 * ((t + 1) + a1 * 0.5 * (t**2 - 1) + a2 * 0.5 * (t**3 - t))


class ScaledWidth(Shape):
    """``shape`` with its ``sigma`` multiplied by a fixed ``factor``.

    Ties a peak's width to another peak's: share ``sigma_2S -> sigma_1S`` and wrap the
    2S shape with ``factor = m_2S / m_1S``. Same parameters as ``shape``.
    """

    def __init__(self, shape: Shape, factor: float):
        self.window, self.params, self.shape, self.factor = shape.window, shape.params, shape, float(factor)
        self._i = [p.name for p in shape.params].index("sigma")

    def _scaled(self, p):
        p = list(p)
        p[self._i] *= self.factor
        return p

    def pdf(self, x, *p):
        return self.shape.pdf(x, *self._scaled(p))

    def cdf(self, x, *p):
        return self.shape.cdf(x, *self._scaled(p))

    def __repr__(self) -> str:
        return f"{self.shape!r} x sigma*{self.factor:.5f}"


SIGNALS: dict[str, type[Shape]] = {
    "gauss": Gaussian, "double_gauss": DoubleGaussian, "crystal_ball": CrystalBall, "double_cb": DoubleCrystalBall,
}

# Signal shape for the final measurement (tails shared across peaks, see ``share_tails``).
BASELINE_SIGNAL = "double_cb"
TAIL_PARAMS: dict[str, tuple[str, ...]] = {
    "gauss": (), "double_gauss": ("ratio", "frac"), "crystal_ball": ("alpha", "n"),
    "double_cb": ("alpha", "n", "ratio", "frac"),
}
BACKGROUNDS: dict[str, type[Shape]] = {"exp": Exponential, "poly2": Quadratic}


# --- composite model -----------------------------------------------------------

class MassModel:
    """Extended sum of shapes, usable directly as the ``scaled_pdf`` of
    ``iminuit.cost.ExtendedUnbinnedNLL``; :meth:`scaled_cdf` serves ``ExtendedBinnedNLL``.
    """

    def __init__(self, window: Window, components: dict[str, Shape], share: dict[str, str] | None = None):
        self.window = (float(window[0]), float(window[1]))
        self.components = dict(components)
        share = dict(share or {})
        self.share = share
        params: dict[str, Param] = {}

        def add(name: str, p: Param) -> int:
            name = share.get(name, name)
            if name not in params:
                params[name] = Param(name, p.start, p.limits)
            return list(params).index(name)

        self._yield_index: dict[str, int] = {}
        self._index: dict[str, list[int]] = {}
        for k, shape in self.components.items():
            if shape.window != self.window:
                raise ValueError(f"component {k!r} has window {shape.window}, model has {self.window}")
            self._yield_index[k] = add(f"N_{k}", Param("N", 1.0, (0.0, None)))
            self._index[k] = [add(f"{p.name}_{k}", p) for p in shape.params]

        unused = set(share) - {n for k, s in self.components.items() for n in (f"N_{k}", *(f"{p.name}_{k}" for p in s.params))}
        if unused:
            raise ValueError(f"share refers to unknown parameter(s) {sorted(unused)}")
        self.params = params
        # iminuit reads names/limits from ``_parameters``; the first entry is the observable.
        self._parameters = {"x": None, **{n: p.limits for n, p in params.items()}}

    @property
    def names(self) -> list[str]:
        return list(self.params)

    def per_cell(self, cell: str, common: set[str]) -> MassModel:
        """This model with every parameter *not* in ``common`` renamed ``<name>_<cell>``.

        For simultaneous fits in kinematic cells: add the cells' costs and the parameters
        that keep their name (e.g. the masses) are shared, the renamed ones are per cell.
        """
        rename = {}
        for k, shape in self.components.items():
            for name in [f"N_{k}", *(f"{p.name}_{k}" for p in shape.params)]:
                target = self.share.get(name, name)
                rename[name] = target if target in common else f"{target}_{cell}"
        return MassModel(self.window, self.components, rename)

    def _split(self, par):
        for k, shape in self.components.items():
            yield k, shape, par[self._yield_index[k]], [par[i] for i in self._index[k]]

    def __call__(self, x, *par):
        """(total yield, sum_k N_k PDF_k(x))."""
        total, density = 0.0, 0.0
        for _, shape, n, p in self._split(par):
            total += n
            density = density + n * shape.pdf(x, *p)
        return total, density

    def scaled_cdf(self, xe, *par):
        """sum_k N_k cdf_k(xe)."""
        return sum(n * shape.cdf(xe, *p) for _, shape, n, p in self._split(par))

    def sigma(self, k: str, values: dict[str, float]) -> float:
        """Component ``k``'s effective ``sigma`` at ``values`` (including a ``ScaledWidth`` factor)."""
        shape = self.components[k]
        name = self.names[self._index[k][[p.name for p in shape.params].index("sigma")]]
        return values[name] * getattr(shape, "factor", 1.0)

    def component_counts(self, edges, values: dict[str, float]) -> dict[str, np.ndarray]:
        """Expected counts per bin for each component at ``values``.

        With ``edges=[a, b]`` this is the expected yield in a region (e.g. for purity).
        """
        par = [values[n] for n in self.names]
        return {k: n * np.diff(shape.cdf(np.asarray(edges), *p)) for k, shape, n, p in self._split(par)}

    def __repr__(self) -> str:
        comps = ", ".join(f"{k}={s!r}" for k, s in self.components.items())
        return f"MassModel(window={self.window}, {comps})"


def share_tails(signal: str, peaks: tuple[str, ...] = ("1S", "2S", "3S")) -> dict[str, str]:
    """``share`` mapping that gives every peak the first peak's tail parameters.

    Floating tails per peak is not identifiable on the full dataset: the 2S tail
    runs off (alpha -> large, n unconstrained) and HESSE breaks. One shared set is stable.
    """
    return {f"{p}_{k}": f"{p}_{peaks[0]}" for p in TAIL_PARAMS[signal] for k in peaks[1:]}


def build_model(
    peaks: tuple[str, ...] = ("1S", "2S", "3S"),
    signal: str = "gauss",
    background: str | None = "exp",
    window: Window = (9.0, 11.0),  # data files start at 9.0 GeV (brief says 8.5)
    share: dict[str, str] | None = None,
    mass_range: float = 0.1,
    tie_widths: bool = False,
) -> MassModel:
    """Signal peak(s) + background. Means start at PDG values, limited to ``+- mass_range``.

    ``background=None`` gives a signal-only model, for simulation (pure signal).
    ``tie_widths=True`` fixes sigma_k = sigma_1S * m_k / m_1S (PDG mass ratio), as in
    JHEP06(2013)064: the relative momentum resolution, and so sigma/m, is about the same
    for all three peaks. It removes two width parameters.

    Examples::

        build_model(("1S",), window=(9.0, 9.9))
        build_model(signal="crystal_ball", share={"alpha_2S": "alpha_1S", "alpha_3S": "alpha_1S"})
    """
    components: dict[str, Shape] = {}
    for peak in peaks:
        m = PDG_MASS[peak]
        components[peak] = SIGNALS[signal](window, mu=m, limits={"mu": (m - mass_range, m + mass_range)})
        if tie_widths and peak != peaks[0]:
            components[peak] = ScaledWidth(components[peak], m / PDG_MASS[peaks[0]])
    if tie_widths:
        share = {**(share or {}), **{f"sigma_{p}": f"sigma_{peaks[0]}" for p in peaks[1:]}}
    if background is not None:
        components["bkg"] = BACKGROUNDS[background](window)
    return MassModel(window, components, share)
