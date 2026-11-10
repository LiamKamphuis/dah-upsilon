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
from scipy import stats

Window = tuple[float, float]
Limits = tuple[float | None, float | None]

#: PDG masses in GeV/c^2, used as starting values.
PDG_MASS = {"1S": 9.4603, "2S": 10.02326, "3S": 10.3552}


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

    PARAMS = (Param("mu", 9.46, (8.5, 11.0)), Param("sigma", 0.04, (5e-3, 0.2)))
    
    def _norm(self, mu, sigma):
        lo, hi = self.window
        return stats.norm.cdf(hi, mu, sigma) - stats.norm.cdf(lo, mu, sigma)

    def pdf(self, x, mu, sigma):
        return stats.norm.pdf(x, mu, sigma) / self._norm(mu, sigma)

    def cdf(self, x, mu, sigma):
        lo = self.window[0]
        return (stats.norm.cdf(x, mu, sigma) - stats.norm.cdf(lo, mu, sigma)) / self._norm(mu, sigma)


class DoubleGaussian(Shape):
    """Narrow + wide Gaussian sharing a mean (TODO).

    ``ratio = sigma_wide / sigma`` (>1) and ``frac`` (narrow fraction). A ratio
    rather than two free widths reduces their correlation and stops the
    components swapping roles: worth discussing as a model choice.
    """

    PARAMS = (
        Param("mu", 9.46, (8.5, 11.0)),
        Param("sigma", 0.04, (5e-3, 0.2)),
        Param("ratio", 2.0, (1.0, 10.0)),
        Param("frac", 0.7, (0.0, 1.0)),
    )

    def pdf(self, x, mu, sigma, ratio, frac):
        raise NotImplementedError("frac * N(mu, sigma) + (1-frac) * N(mu, ratio*sigma), each truncated")

    def cdf(self, x, mu, sigma, ratio, frac):
        raise NotImplementedError


class CrystalBall(Shape):
    """Gaussian core with a low-mass power-law tail from final-state radiation (TODO).

    Hint: ``scipy.stats.crystalball(beta=alpha, m=n, loc=mu, scale=sigma)``,
    normalised on the window via its cdf. ``alpha`` and ``n`` are strongly
    correlated and weakly constrained by data, hence fixing them from MC.
    """

    PARAMS = (
        Param("mu", 9.46, (8.5, 11.0)),
        Param("sigma", 0.04, (5e-3, 0.2)),
        Param("alpha", 1.5, (0.1, 10.0)),
        Param("n", 3.0, (1.01, 50.0)),
    )

    def pdf(self, x, mu, sigma, alpha, n):
        raise NotImplementedError("Crystal Ball normalised on the fit window")

    def cdf(self, x, mu, sigma, alpha, n):
        raise NotImplementedError


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


SIGNALS: dict[str, type[Shape]] = {"gauss": Gaussian, "double_gauss": DoubleGaussian, "crystal_ball": CrystalBall}
BACKGROUNDS: dict[str, type[Shape]] = {"exp": Exponential}


# --- composite model -----------------------------------------------------------

class MassModel:
    """Extended sum of shapes, usable directly as the ``scaled_pdf`` of
    ``iminuit.cost.ExtendedUnbinnedNLL``; :meth:`scaled_cdf` serves ``ExtendedBinnedNLL``.
    """

    def __init__(self, window: Window, components: dict[str, Shape], share: dict[str, str] | None = None):
        self.window = (float(window[0]), float(window[1]))
        self.components = dict(components)
        share = dict(share or {})
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

        unused = set(share) - {f"{p.name}_{k}" for k, s in self.components.items() for p in s.params}
        if unused:
            raise ValueError(f"share refers to unknown parameter(s) {sorted(unused)}")
        self.params = params
        # iminuit reads names/limits from ``_parameters``; the first entry is the observable.
        self._parameters = {"x": None, **{n: p.limits for n, p in params.items()}}

    @property
    def names(self) -> list[str]:
        return list(self.params)

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

    def component_counts(self, edges, values: dict[str, float]) -> dict[str, np.ndarray]:
        """Expected counts per bin for each component at ``values``.

        With ``edges=[a, b]`` this is the expected yield in a region (e.g. for purity).
        """
        par = [values[n] for n in self.names]
        return {k: n * np.diff(shape.cdf(np.asarray(edges), *p)) for k, shape, n, p in self._split(par)}

    def __repr__(self) -> str:
        comps = ", ".join(f"{k}={s!r}" for k, s in self.components.items())
        return f"MassModel(window={self.window}, {comps})"


def build_model(
    peaks: tuple[str, ...] = ("1S", "2S", "3S"),
    signal: str = "gauss",
    background: str = "exp",
    window: Window = (9.0, 11.0),  # data files start at 9.0 GeV (brief says 8.5)
    share: dict[str, str] | None = None,
    mass_range: float = 0.1,
) -> MassModel:
    """Signal peak(s) + background. Means start at PDG values, limited to ``+- mass_range``.

    Examples::

        build_model(("1S",), window=(9.0, 9.9))
        build_model(signal="crystal_ball", share={"alpha_2S": "alpha_1S", "alpha_3S": "alpha_1S"})
    """
    components: dict[str, Shape] = {}
    for peak in peaks:
        m = PDG_MASS[peak]
        components[peak] = SIGNALS[signal](window, mu=m, limits={"mu": (m - mass_range, m + mass_range)})
    components["bkg"] = BACKGROUNDS[background](window)
    return MassModel(window, components, share)
