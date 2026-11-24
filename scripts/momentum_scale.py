"""Is one momentum scale enough? Fit the Upsilon(1S) peak in kinematic bins.

A global momentum-scale error p -> (1 + alpha) p moves every mass by alpha * m
(LHCb momentum-scale paper, arXiv:2312.01772, Eq. 8), the same in every bin. A bias
that grows with momentum (e.g. a curvature bias, which the paper says dominates at
high p) would instead make alpha_eff = mu / m_PDG - 1 drift across the bins.

The radiative-tail shape bias also moves mu, by an amount that scales with the
resolution sigma, and sigma varies with the kinematics. Simulation has the same
columns and (presumably) a perfect momentum scale, so it is fitted in the same bins:
the shape bias shows up in the MC points, and data - MC is the detector effect up to
one constant (the unknown generated mass shifts every MC point equally).
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import upsilon as ups
from upsilon.io import FIGURE_DIR, LABELS, RESULT_DIR, output_subdir
from upsilon.models import BASELINE_SIGNAL, PDG_MASS, TAIL_PARAMS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="full")
parser.add_argument("--bins", type=int, default=6, help="equal-population bins per variable")
args = parser.parse_args()

M = PDG_MASS["1S"]
WINDOW = (9.0, 9.9)
PAPER_ALPHA_ERR = 3e-4  # momentum-scale uncertainty quoted in arXiv:2312.01772


def variables(events: np.ndarray) -> dict[str, np.ndarray]:
    lo, hi = np.minimum(events["pt1"], events["pt2"]), np.maximum(events["pt1"], events["pt2"])
    return {"p": events["p"], "pt": events["pt"], "eta": events["eta"],
            "pt_min": lo, "pt_asym": (hi - lo) / (hi + lo)}


LABEL = {**LABELS, "pt_min": r"min($p_{\perp,1}, p_{\perp,2}$) [GeV/$c$]",
         "pt_asym": r"$|p_{\perp,1}-p_{\perp,2}|/(p_{\perp,1}+p_{\perp,2})$"}


def make_fitter(mass, background):
    """Return fit_cell(selection) for one sample: mu and sigma float per cell, the tail
    shape (in units of sigma) is fixed to the sample's inclusive fit. Floated per cell,
    the four tail parameters are not constrained and the fits fail."""
    model = ups.build_model(("1S",), signal=BASELINE_SIGNAL, window=WINDOW,
                            background="exp" if background else None)
    # Simulation unbinned, as in signal_shape.py: binned double-CB fits to it do not always converge
    inclusive = ups.fit(model, mass, binned=background, progress=False)
    tails = {f"{p}_1S": inclusive.values[f"{p}_1S"] for p in TAIL_PARAMS[BASELINE_SIGNAL]}

    def fit_cell(sel, where=""):
        r = ups.fit(model, mass[sel], binned=background, fix=tails, progress=False)
        if not r.valid:
            print(f"  warning: fit {where} not valid")
        return {"valid": r.valid, "alpha": r.values["mu_1S"] / M - 1, "alpha_err": r.errors["mu_1S"] / M,
                "sigma": r.values["sigma_1S"], "sigma_err": r.errors["sigma_1S"]}
    return fit_cell


def quantile_edges(x, n):
    """Equal-population edges with open outer edges."""
    edges = np.quantile(x, np.linspace(0, 1, n + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


data, mc = ups.load(args.dataset), ups.load("mc")
vd, vm = variables(data), variables(mc)
fit_data, fit_mc = make_fitter(data["mass"], True), make_fitter(mc["mass"], False)
# Bin edges from the data's Upsilon(1S) region, shared with MC
in_peak = (data["mass"] > 9.36) & (data["mass"] < 9.56)
results = {}
for name in vd:
    edges = quantile_edges(vd[name][in_peak], args.bins)
    print(f"{name}: fitting {args.bins} bins (data, MC)")
    results[name] = {"data": [], "mc": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel_d, sel_m = (vd[name] >= lo) & (vd[name] < hi), (vm[name] >= lo) & (vm[name] < hi)
        where = f"{name} in [{lo:.3g}, {hi:.3g})"
        results[name]["data"].append({"lo": lo, "hi": hi, "centre": float(np.median(vd[name][sel_d])),
                                      **fit_data(sel_d, f"data {where}")})
        results[name]["mc"].append({"lo": lo, "hi": hi, "centre": float(np.median(vm[name][sel_m])),
                                    **fit_mc(sel_m, f"MC {where}")})

(RESULT_DIR / output_subdir(args.dataset)).mkdir(parents=True, exist_ok=True)
(RESULT_DIR / output_subdir(args.dataset) / "momentum_scale.json").write_text(json.dumps(results, indent=2, default=float))

fig, axes = plt.subplots(3, len(results), figsize=(4.2 * len(results), 10), sharex="col", sharey="row")
for col, (name, res) in enumerate(results.items()):
    get = {k: {key: np.array([r[key] for r in rows]) for key in rows[0]} for k, rows in res.items()}
    d, m = get["data"], get["mc"]
    x = d["centre"]
    axes[0, col].errorbar(x, 1e4 * d["alpha"], 1e4 * d["alpha_err"], fmt="ko", label="Data")
    axes[0, col].errorbar(m["centre"], 1e4 * m["alpha"], 1e4 * m["alpha_err"], fmt="s", color="C1", label="Simulation")
    diff, diff_err = d["alpha"] - m["alpha"], np.hypot(d["alpha_err"], m["alpha_err"])
    mean = np.average(diff, weights=diff_err**-2)
    axes[1, col].errorbar(x, 1e4 * diff, 1e4 * diff_err, fmt="ko")
    axes[1, col].axhspan(1e4 * (mean - PAPER_ALPHA_ERR), 1e4 * (mean + PAPER_ALPHA_ERR), color="C0", alpha=0.15,
                         label=r"mean $\pm\,3\times10^{-4}$ (paper)")
    axes[1, col].axhline(1e4 * mean, color="C0")
    axes[2, col].errorbar(x, 1e3 * d["sigma"], 1e3 * d["sigma_err"], fmt="ko")
    axes[2, col].errorbar(m["centre"], 1e3 * m["sigma"], 1e3 * m["sigma_err"], fmt="s", color="C1")
    axes[2, col].set_xlabel(LABEL[name])
    chi2 = np.sum(((diff - mean) / diff_err) ** 2)
    axes[1, col].set_title(rf"flat: $\chi^2$/ndof = {chi2:.0f}/{len(diff) - 1}", fontsize=10)
    print(f"  {name}: data - MC alpha spread {1e4 * np.ptp(diff):.1f}e-4, chi2/ndof (flat) = {chi2:.1f}/{len(diff) - 1}")
for col in range(len(results)):  # p and pt have long tails: log scale keeps the bins readable
    if list(results)[col] in ("p", "pt"):
        axes[2, col].set_xscale("log")
axes[0, 0].set_ylabel(r"$\alpha_\mathrm{eff} = \mu/m_\mathrm{PDG} - 1$ [$10^{-4}$]")
axes[1, 0].set_ylabel(r"$\alpha_\mathrm{data} - \alpha_\mathrm{MC}$ [$10^{-4}$]")
axes[2, 0].set_ylabel(r"core $\sigma$ [MeV/$c^2$]")
axes[0, 0].legend()
axes[1, 0].legend(fontsize=8)
fig.suptitle(rf"$\Upsilon(1S)$ peak position and resolution in kinematic bins ({args.dataset} data, {BASELINE_SIGNAL})")
fig.tight_layout()
(FIGURE_DIR / output_subdir(args.dataset)).mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_DIR / output_subdir(args.dataset) / "momentum_scale.png", dpi=150, bbox_inches="tight")

# --- 2D: p within eta bins, and a linear model for alpha(p) ----------------------
# p and eta are correlated (p = pT cosh eta), so the 1D trends above cannot tell which
# one drives the shift. Bin eta first, then p by quantile *within* each eta bin: every
# eta band spans its own p range, and alpha vs p at fixed eta separates the two.
N_ETA, N_P = 4, 4
cells = []
eta_edges = quantile_edges(vd["eta"][in_peak], N_ETA)
print(f"2D: fitting {N_ETA} eta x {N_P} p cells (data, MC)")
for i, (elo, ehi) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):
    band_d, band_m = (vd["eta"] >= elo) & (vd["eta"] < ehi), (vm["eta"] >= elo) & (vm["eta"] < ehi)
    p_edges = quantile_edges(vd["p"][in_peak & band_d], N_P)
    for plo, phi in zip(p_edges[:-1], p_edges[1:]):
        sel_d = band_d & (vd["p"] >= plo) & (vd["p"] < phi)
        sel_m = band_m & (vm["p"] >= plo) & (vm["p"] < phi)
        where = f"eta [{elo:.3g}, {ehi:.3g}) p [{plo:.3g}, {phi:.3g})"
        d, m = fit_data(sel_d, f"data {where}"), fit_mc(sel_m, f"MC {where}")
        cells.append({"eta_bin": i, "eta": float(np.median(vd["eta"][sel_d])), "p": float(np.median(vd["p"][sel_d])),
                      "data": d, "mc": m, "diff": d["alpha"] - m["alpha"],
                      "diff_err": float(np.hypot(d["alpha_err"], m["alpha_err"]))})

y = np.array([c["diff"] for c in cells])
err = np.array([c["diff_err"] for c in cells])
p_cell, eta_cell = np.array([c["p"] for c in cells]), np.array([c["eta"] for c in cells])
P0, ETA0 = 100.0, 4.0  # reference point: alpha0 is the offset at p = 100 GeV, eta = 4
designs = {
    "constant": np.column_stack([np.ones_like(y)]),
    "alpha0 + k (p - 100)": np.column_stack([np.ones_like(y), p_cell - P0]),
    "alpha0 + k (p - 100) + j (eta - 4)": np.column_stack([np.ones_like(y), p_cell - P0, eta_cell - ETA0]),
}
fits = {}
print("\nalpha_data - alpha_MC in 2D cells: weighted least squares (cells are independent)")
for name, X in designs.items():
    Xw, yw = X / err[:, None], y / err
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.inv(Xw.T @ Xw)
    chi2 = float(np.sum((yw - Xw @ coef) ** 2))
    ndof = len(y) - X.shape[1]
    fits[name] = {"coef": coef.tolist(), "err": np.sqrt(np.diag(cov)).tolist(), "chi2": chi2, "ndof": ndof,
                  "pvalue": float(stats.chi2.sf(chi2, ndof))}
    terms = ["alpha0", "k [per GeV]", "j [per unit eta]"][: X.shape[1]]
    print(f"  {name:<36} chi2/ndof = {chi2:5.1f}/{ndof}  p = {fits[name]['pvalue']:.2g}   " + "  ".join(
        f"{t} = {v:.3g} +- {e:.2g}" for t, v, e in zip(terms, coef, fits[name]["err"])))

(RESULT_DIR / output_subdir(args.dataset) / "momentum_scale_2d.json").write_text(
    json.dumps({"cells": cells, "fits": fits}, indent=2, default=float))

lin = fits["alpha0 + k (p - 100)"]
fig, (ax, axr) = plt.subplots(1, 2, figsize=(12, 4.8))
for i in range(N_ETA):
    s = np.array([c["eta_bin"] == i for c in cells])
    label = rf"$\eta$ bin {i + 1} (median {np.median(eta_cell[s]):.2f})"
    ax.errorbar(p_cell[s], 1e4 * y[s], 1e4 * err[s], fmt="o", label=label)
    resid = y[s] - (lin["coef"][0] + lin["coef"][1] * (p_cell[s] - P0))
    axr.errorbar(eta_cell[s], 1e4 * resid, 1e4 * err[s], fmt="o")
grid = np.linspace(p_cell.min(), p_cell.max(), 100)
ax.plot(grid, 1e4 * (lin["coef"][0] + lin["coef"][1] * (grid - P0)), "k-",
        label=rf"$\alpha_0 + k(p-100)$: $\chi^2$/ndof = {lin['chi2']:.0f}/{lin['ndof']}")
ax.set(xscale="log", xlabel=LABELS["p"], ylabel=r"$\alpha_\mathrm{data} - \alpha_\mathrm{MC}$ [$10^{-4}$]",
       title=r"Momentum dependence at fixed $\eta$")
ax.legend(fontsize=8)
axr.axhline(0, color="k", lw=0.8)
axr.set(xlabel=LABELS["eta"], ylabel=r"residual from $\alpha(p)$ [$10^{-4}$]",
        title=r"Anything left for $\eta$ once $p$ is accounted for?")
fig.suptitle(rf"$\Upsilon(1S)$: $p\times\eta$ cells ({args.dataset} data)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / output_subdir(args.dataset) / "momentum_scale_2d.png", dpi=150, bbox_inches="tight")
