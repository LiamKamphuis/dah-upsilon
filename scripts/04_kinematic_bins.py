"""Brief task 7: one simultaneous fit in bins of dimuon pT and rapidity y.

The resolution depends on the kinematics (the muon momentum resolution degrades with p,
and p ~ pT cosh y), so an inclusive fit blends peaks of different widths. Here every
(y, pT) cell has its own resolution, yields and background slope, while the three
masses (and the tail shape) are common to all cells: the fit is the sum of the cells'
binned -2 ln L, so the masses are measured from all cells at once.

Compared with the inclusive fit, look at:
- the masses and their statistical errors (do they move, and is the fit more precise?);
- the goodness of fit per cell and in total (does the shape describe each cell?);
- sigma_1S per cell (the resolution map; the paper parameterizes it as linear in pT and y).

Selection: softer muon pT >= MIN_MUON_PT and the paper's fiducial region 2.0 < y < 4.5,
pT < 15 GeV/c (which also drops the unphysical pT ~ TeV outliers). The cells and the fit are
``upsilon.fitting.fit_kinematic_cells``, shared with 06_final_masses.py, where this fit is the baseline.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import upsilon as ups
from upsilon.io import output_subdir
from upsilon.models import BASELINE_SIGNAL, SIGNALS, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--signal", default=BASELINE_SIGNAL, choices=sorted(SIGNALS))
parser.add_argument("--coarse", action="store_true", help="2 x 2 cells, to check the machinery quickly")
args = parser.parse_args()

PEAKS = ("1S", "2S", "3S")
if args.coarse:
    Y_EDGES, PT_EDGES = (2.0, 3.25, 4.5), (0.0, 5.0, 15.0)
else:
    Y_EDGES, PT_EDGES = ups.fitting.Y_EDGES, ups.fitting.PT_EDGES

events = ups.select_muons(ups.load(args.dataset))
fiducial = ups.fitting.fiducial(events, Y_EDGES, PT_EDGES)
print(f"{fiducial.sum()} of {len(events)} selected events in the fiducial region")

model = ups.build_model(signal=args.signal, share=share_tails(args.signal), tie_widths=True)
inclusive = ups.fit(model, events["mass"][fiducial], binned=True, progress=False)

cells, results = ups.fitting.fit_kinematic_cells(model, events, args.signal, inclusive.values,
                                                 y_edges=Y_EDGES, pt_edges=PT_EDGES, progress=True)
print(f"simultaneous fit: {len(cells)} cells ({len(Y_EDGES) - 1} y x {len(PT_EDGES) - 1} pT), {args.signal}")
m = results[0].minuit
print(f"valid={m.valid}, accurate covariance={m.fmin.has_accurate_covar}, {sum(not f for f in m.fixed)} free parameters")
# Over-parametrisation check: the shared tail shape is fixed by all cells together, but could still
# trade off against the per-cell widths (a correlation near +-1 = only a combination is constrained)
correlations = ups.compare.strong_correlations(results[0])
print("|correlation| >= 0.5: " + (", ".join(f"{a}/{b} {rho:+.2f}" for a, b, rho in correlations) or "none"))

# Goodness of fit: Baker-Cousins chi2 per cell; total ndof = all bins - all free parameters
chi2_cells = [ups.compare.gof(r).stat for r in results]
nbins = sum(len(r.edges) - 1 for r in results)
total = {"chi2": float(sum(chi2_cells)), "ndof": nbins - sum(not f for f in m.fixed)}
incl_gof = ups.compare.gof(inclusive)
print(f"\ngoodness of fit: simultaneous chi2/ndof = {total['chi2']:.0f}/{total['ndof']} "
      f"(p = {stats.chi2.sf(total['chi2'], total['ndof']):.2g}); "
      f"inclusive {incl_gof.stat:.0f}/{incl_gof.ndof} (p = {incl_gof.pvalue:.2g})")

v, e = results[0].values, results[0].errors
print(f"\n{'mass':<6}{'inclusive - PDG':>22}{'simultaneous - PDG':>24}{'shift':>10}   [MeV/c^2]")
for p in PEAKS:
    ref = ups.models.PDG_MASS[p]
    print(f"{p:<6}{1e3 * (inclusive.values[f'mu_{p}'] - ref):>+12.2f} +- {1e3 * inclusive.errors[f'mu_{p}']:.3f}"
          f"{1e3 * (v[f'mu_{p}'] - ref):>+14.2f} +- {1e3 * e[f'mu_{p}']:.3f}"
          f"{1e3 * (v[f'mu_{p}'] - inclusive.values[f'mu_{p}']):>+10.2f}")

print(f"\n{'cell':<8}{'y':>12}{'pT [GeV]':>12}{'events':>9}{'N_1S':>9}{'N_3S':>8}{'sigma_1S [MeV]':>17}{'chi2 (250 bins)':>17}")
for c, r, chi2 in zip(cells, results, chi2_cells):
    n = c["cell"]
    c.update({"sigma_1S": v[f"sigma_1S_{n}"], "sigma_1S_err": e[f"sigma_1S_{n}"], "chi2": chi2,
              **{f"N_{p}": v[f"N_{p}_{n}"] for p in PEAKS}})
    print(f"{n:<8}{str(c['y']):>12}{str(c['pt']):>12}{c['events']:>9}{c['N_1S']:>9.0f}{c['N_3S']:>8.0f}"
          f"{1e3 * c['sigma_1S']:>11.1f} +- {1e3 * c['sigma_1S_err']:.1f}{chi2:>14.0f}")

tag = f"{args.signal}{'_coarse' if args.coarse else ''}"
out = output_subdir(args.dataset)
ups.save_json({"y_edges": Y_EDGES, "pt_edges": PT_EDGES, "valid": m.valid, "values": v, "errors": e,
               "inclusive": {"values": inclusive.values, "errors": inclusive.errors, "chi2": incl_gof.stat,
                             "ndof": incl_gof.ndof},
               "total_gof": total, "cells": cells, "strong_correlations": correlations},
              f"kinematic_bins_{tag}.json", args.dataset)

# Where does the model fail? Pulls of the worst cell: a smooth curve across the whole window
# points at the background shape, structure at the peaks at the signal shape.
worst = int(np.argmax(chi2_cells))
c = cells[worst]
ups.plotting.plot_fit(results[worst], title="",
                      legend_title=rf"${c['y'][0]} < y < {c['y'][1]}$, ${c['pt'][0]:g} < p_\perp < {c['pt'][1]:g}$ GeV/$c$",
                      save=f"{out}/kinematic_bins_worst_cell_{tag}.png")

# Resolution map: sigma_1S against pT, one line per rapidity bin
fig, ax = plt.subplots(figsize=(12, 9))
for i, (ylo, yhi) in enumerate(zip(Y_EDGES[:-1], Y_EDGES[1:])):
    row = [c for c in cells if c["y"] == (ylo, yhi)]
    x = [0.5 * sum(c["pt"]) for c in row]
    ax.errorbar(x, [1e3 * c["sigma_1S"] for c in row], [1e3 * c["sigma_1S_err"] for c in row],
                fmt="o-", label=rf"${ylo} < y < {yhi}$")
ax.axhline(1e3 * inclusive.values["sigma_1S"], color="0.5", ls="--", label="inclusive fit")
ax.margins(y=0.08)  # the LHCb style sets zero margins
ax.set(xlim=(PT_EDGES[0], PT_EDGES[-1]), xlabel=r"$p_\perp$ (bin centre) [GeV/$c$]", ylabel=r"$\sigma_{1S}$ [MeV/$c^2$]",
       title=rf"$\Upsilon(1S)$ resolution per cell ({args.signal})")
ax.legend(fontsize=18)
ups.plotting.save_figure(fig, f"{out}/kinematic_bins_sigma_{tag}.png")
