"""Momentum scale: does the Upsilon(1S) peak position in data drift with the kinematics?

A momentum-scale error p -> (1 + alpha) p moves every mass by alpha * m (LHCb momentum-scale
paper, arXiv:2312.01772). Fit the 1S peak in bins of a variable and quote
alpha_eff = mu / m_PDG - 1 per bin. The radiative-tail shape bias also moves mu, by an amount
that depends on the resolution and so on the kinematics, so simulation (perfect momentum scale)
is fitted in the same bins: alpha_data - alpha_MC is the detector effect, up to one constant.

This is a systematic check, not a model of alpha: per variable, data - MC is fitted with a
constant, and its chi2 and spread (max - min over the bins) are compared with the 3e-4 the paper
assigns for the variation of alpha across decay modes and kinematics (its Sec. 6).

- p: the paper shows alpha against momentum (its Fig. 5). Curvature (alignment) biases are
  charge dependent and mostly cancel in a dimuon mass, so no functional form in p is assumed.
- softer-muon pT: the selection variable. If data - MC is flat here, the mass drift across
  pT slices in 03_selection.py is a shape/resolution effect, present in simulation too, not a
  calibration problem.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import upsilon as ups
from upsilon.io import LABELS, output_subdir
from upsilon.models import BASELINE_SIGNAL, PDG_MASS, TAIL_PARAMS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--bins", type=int, default=6, help="equal-population bins per variable")
args = parser.parse_args()

M = PDG_MASS["1S"]
PAPER_ALPHA_ERR = 3e-4  # momentum-scale uncertainty of arXiv:2312.01772
VARIABLES = {"p": LABELS["p"], "pt_min": r"min($p_{\perp,1}, p_{\perp,2}$) [GeV/$c$]"}


def variables(events):
    return {"p": events["p"], "pt_min": np.minimum(events["pt1"], events["pt2"])}


def make_fitter(mass, background):
    """fit_cell(selection) for one sample: mu and sigma float per bin, the tail shape is fixed
    to the sample's inclusive fit (floated per bin, the four tail parameters are not constrained)."""
    model = ups.build_model(("1S",), signal=BASELINE_SIGNAL, window=(9.0, 9.9),
                            background="exp" if background else None)
    # Simulation unbinned, as in 02_signal_shape.py: binned double-CB fits to it do not always converge
    inclusive = ups.fit(model, mass, binned=background, progress=False)
    tails = {f"{t}_1S": inclusive.values[f"{t}_1S"] for t in TAIL_PARAMS[BASELINE_SIGNAL]}

    def fit_cell(sel):
        r = ups.fit(model, mass[sel], binned=background, fix=tails, progress=False)
        if not r.valid:
            print("  warning: a bin fit is not valid")
        return r.values["mu_1S"] / M - 1, r.errors["mu_1S"] / M
    return fit_cell


data, mc = ups.load(args.dataset), ups.load("mc")
vd, vm = variables(data), variables(mc)
fit_data, fit_mc = make_fitter(data["mass"], True), make_fitter(mc["mass"], False)
in_peak = (data["mass"] > 9.36) & (data["mass"] < 9.56)

results = {}
fig, axes = plt.subplots(2, len(VARIABLES), figsize=(20, 16), sharex="col")
for col, (name, label) in enumerate(VARIABLES.items()):
    edges = np.quantile(vd[name][in_peak], np.linspace(0, 1, args.bins + 1))  # equal 1S population
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel_d, sel_m = (vd[name] >= lo) & (vd[name] < hi), (vm[name] >= lo) & (vm[name] < hi)
        (ad, ed), (am, em) = fit_data(sel_d), fit_mc(sel_m)
        rows.append({"centre": float(np.median(vd[name][sel_d & in_peak])), "alpha_data": ad, "err_data": ed,
                     "alpha_mc": am, "err_mc": em, "diff": ad - am, "diff_err": float(np.hypot(ed, em))})
    x, y, err = (np.array([r[k] for r in rows]) for k in ("centre", "diff", "diff_err"))

    # Weighted mean (constant fit); the spread across bins is what is compared with the 3e-4
    w = err ** -2.0
    const, const_err = float(np.sum(w * y) / np.sum(w)), float(np.sum(w) ** -0.5)
    chi2, ndof, spread = float(np.sum(((y - const) / err) ** 2)), len(y) - 1, float(np.ptp(y))
    results[name] = {"bins": rows, "constant": const, "constant_err": const_err, "chi2": chi2, "ndof": ndof,
                     "spread": spread}
    print(f"{name}: data - MC = constant {1e4 * const:.2f} +- {1e4 * const_err:.2f} [1e-4]; "
          f"chi2/ndof = {chi2:.1f}/{ndof} (p = {stats.chi2.sf(chi2, ndof):.2g}); "
          f"spread (max - min) = {1e4 * spread:.2f}e-4 vs {1e4 * PAPER_ALPHA_ERR:.0f}e-4")

    top, bottom = axes[0, col], axes[1, col]
    top.errorbar(x, 1e4 * np.array([r["alpha_data"] for r in rows]), 1e4 * np.array([r["err_data"] for r in rows]),
                 fmt="ko", label="Data")
    top.errorbar(x, 1e4 * np.array([r["alpha_mc"] for r in rows]), 1e4 * np.array([r["err_mc"] for r in rows]),
                 fmt="s", color="C1", label="Simulation")
    bottom.errorbar(x, 1e4 * y, 1e4 * err, fmt="ko")
    # Pad the x range (the LHCb style sets zero margins, which puts end points on the frame)
    xlim = (x.min() / 1.15, x.max() * 1.15) if name == "p" else (x.min() - 0.08 * np.ptp(x), x.max() + 0.08 * np.ptp(x))
    grid = (np.geomspace if name == "p" else np.linspace)(*xlim, 100)
    bottom.set_xlim(xlim)
    bottom.set_ylim(1e4 * (const - 1.6 * PAPER_ALPHA_ERR), 1e4 * (const + 1.6 * PAPER_ALPHA_ERR))  # band edges visible
    top.margins(y=0.08)
    fit_line = np.full_like(grid, const)
    bottom.plot(grid, 1e4 * fit_line, "C0", label=rf"constant: $\chi^2$/ndof = {chi2:.0f}/{ndof}")
    bottom.fill_between(grid, 1e4 * (fit_line - PAPER_ALPHA_ERR), 1e4 * (fit_line + PAPER_ALPHA_ERR),
                        color="C0", alpha=0.15, label=r"$\pm3\times10^{-4}$ (LHCb calibration)")
    bottom.set_xlabel(label)
    bottom.legend(fontsize=16)
    if name == "p":
        top.set_xscale("log")
axes[0, 0].set_ylabel(r"$\alpha_\mathrm{eff} = \mu/m_\mathrm{PDG} - 1$ [$10^{-4}$]")
axes[1, 0].set_ylabel(r"$\alpha_\mathrm{data} - \alpha_\mathrm{MC}$ [$10^{-4}$]")
axes[0, 0].legend()
fig.suptitle(rf"$\Upsilon(1S)$ peak position in kinematic bins ({args.dataset} data, {BASELINE_SIGNAL})", fontsize=24)
fig.tight_layout()
ups.plotting.save_figure(fig, f"{output_subdir(args.dataset)}/momentum_scale.png")
ups.save_json(results, "momentum_scale.json", args.dataset)
