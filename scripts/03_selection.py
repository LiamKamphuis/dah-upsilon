"""Brief task 6: choose a selection that improves the signal purity.

1. 2D plots of each observable against mass.
2. Fit-free sideband estimate: which variable separates signal from background at all?
3. Cut scan on that variable, one three-peak fit per cut: purity, S/sqrt(S+B) and, what the
   measurement needs, the statistical error on each mass.
4. Does the cut move the mass? Fits in independent slices of the cut variable.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import upsilon as ups
from upsilon.io import output_subdir
from upsilon.models import BASELINE_SIGNAL, TAIL_PARAMS, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
args = parser.parse_args()

events = ups.load(args.dataset)
# Step 1: each observable against mass
for col in ("pt", "eta", "p", "pt1", "pt2"):
    ups.plotting.plot_vs_mass(events, col, save=f"{output_subdir(args.dataset)}/{col}_vs_mass.png")

# Step 2, fit-free. Count events in the 1S window and sidebands either side; for a background
# linear in mass, the sidebands (0.4 GeV in total) predict the background under the 0.2 GeV window.
WINDOW, SIDEBANDS = (9.36, 9.56), ((9.0, 9.2), (9.7, 9.9))
mass = events["mass"]
pt_min = np.minimum(events["pt1"], events["pt2"])
S0, _ = ups.sideband_subtract(mass, WINDOW, SIDEBANDS)
sideband = {}
for label, var, cuts in [("min muon pT", pt_min, (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)),
                         ("dimuon pT", events["pt"], (0, 1, 2, 3, 5, 8)),
                         ("dimuon eta", events["eta"], (0, 2.5, 3.0, 3.5))]:
    sideband[label] = []
    for c in cuts:
        s, b = ups.sideband_subtract(mass[var >= c], WINDOW, SIDEBANDS)
        sideband[label].append({"cut": c, "eff": s / S0, "purity": s / (s + b), "signif": s / np.sqrt(s + b)})
        print(f"sideband: {label} >= {c}: signal kept {s / S0:.2f}, purity {s / (s + b):.3f}, "
              f"S/sqrt(S+B) {s / np.sqrt(s + b):.0f}")

fig, ax = plt.subplots(figsize=(12, 9))
for label, rows in sideband.items():
    ax.plot([r["eff"] for r in rows], [r["purity"] for r in rows], "o-", label=label)
ax.set(xlabel=r"$\Upsilon(1S)$ signal kept by the cut", ylabel=r"Purity $S/(S+B)$ in 9.36-9.56 GeV",
       title="Sideband estimate: purity against signal kept")
ax.legend()
ups.plotting.save_figure(fig, f"{output_subdir(args.dataset)}/sideband_purity.png")

# Step 3, with fits. Purity scan: cut on the softer muon's pT (combinatorial background is mostly a soft
# muon paired with anything; dimuon pT and eta barely separate signal from background).
# Each cut gets one three-peak baseline fit, which gives both figures of merit:
#   - S, B in mu +- 2 sigma per peak -> purity S/(S+B) and S/sqrt(S+B)
#   - the HESSE error on each mass, which is what the measurement actually needs.
CUTS = np.round(np.arange(1.0, 4.01, 0.25), 2)  # GeV; 1.0 = no cut (the data start there)
PEAKS = ("1S", "2S", "3S")
model = ups.build_model(signal=BASELINE_SIGNAL, share=share_tails(BASELINE_SIGNAL), tie_widths=True)
# Every cut and slice starts from the no-cut fit: from the default start, MIGRAD sometimes
# lands in a false minimum where the double CB collapses to one CB (ratio -> 1), with
# -2 ln L worse by ~650 but a minimum Minuit still calls valid (e.g. cut at 3.75 GeV).
reference = ups.fit(model, events["mass"], binned=True, progress=False).values

scan = []
for cut in CUTS:
    r = ups.fit(model, events["mass"][pt_min >= cut], binned=True, start=reference, progress=False)
    row = {"cut": cut, "valid": r.valid}
    for p in PEAKS:
        mu, sigma = r.values[f"mu_{p}"], r.model.sigma(p, r.values)
        counts = r.model.component_counts([mu - 2 * sigma, mu + 2 * sigma], r.values)
        s, b = counts[p][0], sum(c[0] for k, c in counts.items() if k != p)  # other peaks count as background
        row[p] = {"mu": mu, "err": r.errors[f"mu_{p}"], "S": s, "B": b,
                  "purity": s / (s + b), "signif": s / np.sqrt(s + b)}
    scan.append(row)
    print(f"min muon pT >= {cut:.2f} GeV  valid={r.valid}  " + "  ".join(
        f"{p}: P={row[p]['purity']:.3f} S/sqrt(S+B)={row[p]['signif']:.0f} "
        f"err={1e3 * row[p]['err']:.3f} MeV" for p in PEAKS))

# Step 4, mass stability. A cut's sample shares most events with the no-cut sample, so the shift
# between the two fits has a correlated error that neither fit's error gives (and Barlow's
# sqrt(err_cut^2 - err_all^2) assumes a random subset, while a cut removes mostly background).
# Independent slices of the cut variable avoid this: if the mass is the same in every slice,
# no cut on it can move the mass. Coarser than CUTS, so every slice has a usable 3S peak.
# Tails fixed to the no-cut fit (as in 05_momentum_scale.py): floated per slice they are not
# constrained in the background-dominated low slices.
tails = {f"{t}_1S": reference[f"{t}_1S"] for t in TAIL_PARAMS[BASELINE_SIGNAL]}
SLICES = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, np.inf)  # below 2 GeV there is almost no 3S
slices = []
for lo, hi in zip(SLICES[:-1], SLICES[1:]):
    sel = (pt_min >= lo) & (pt_min < hi)
    r = ups.fit(model, events["mass"][sel], binned=True, start=reference, fix=tails, progress=False)
    slices.append({"lo": lo, "hi": hi, "centre": float(np.median(pt_min[sel])), "valid": r.valid,
                   **{p: {"mu": r.values[f"mu_{p}"], "err": r.errors[f"mu_{p}"]} for p in PEAKS}})
flatness = {}
for p in PEAKS:
    mu, err = (np.array([sl[p][k] for sl in slices]) for k in ("mu", "err"))
    mean = np.average(mu, weights=err**-2)
    chi2 = float(np.sum(((mu - mean) / err) ** 2))
    flatness[p] = {"mean": mean, "chi2": chi2, "ndof": len(mu) - 1}
    print(f"{p} in softer-muon pT slices: chi2/ndof (flat) = {chi2:.1f}/{len(mu) - 1}, "
          f"p = {stats.chi2.sf(chi2, len(mu) - 1):.2g}")

ups.save_json({"sideband": sideband, "cuts": scan, "slices": slices, "flatness": flatness},
              "selection.json", args.dataset)

cuts = np.array(CUTS)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # sized for the LHCb style's large fonts
for i, p in enumerate(PEAKS):
    get = lambda key: np.array([row[p][key] for row in scan])
    axes[0, 0].plot(cuts, get("purity"), "o-", label=rf"$\Upsilon({p})$")
    axes[0, 1].plot(cuts, get("signif") / get("signif")[0], "o-")
    axes[1, 0].plot(cuts, get("err") / get("err")[0], "o-")
    x = np.array([sl["centre"] for sl in slices])
    axes[1, 1].errorbar(x + 0.04 * (i - 1), 1e3 * (np.array([sl[p]["mu"] for sl in slices]) - flatness[p]["mean"]),
                        yerr=1e3 * np.array([sl[p]["err"] for sl in slices]), fmt="o",
                        label=rf"$\Upsilon({p})$: $\chi^2$/ndof = {flatness[p]['chi2']:.1f}/{flatness[p]['ndof']}")
axes[0, 0].set(ylabel=r"$S/(S+B)$ in $\mu\pm2\sigma$", title=r"Signal purity")
axes[0, 1].set(ylabel=r"$S/\sqrt{S+B}$ / (no cut)", title=r"Signal significance $S/\sqrt{S+B}$")
axes[1, 0].set(ylabel=r"$\delta\mu_\mathrm{stat}$ / (no cut)", title=r"Statistical uncertainty on $\mu$")
axes[1, 1].set(ylabel=r"$\mu - \langle\mu\rangle$ [MeV/$c^2$]",
               title="Mass in independent slices", ylim=(-6, 6), xlim=(1.3, 5.9))
axes[1, 1].legend(fontsize=18, title="consistency with a constant", title_fontsize=18, loc="lower right")
axes[1, 1].axhline(0, color="k", lw=0.8)
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_xlabel(r"Cut on min($p_{\perp,1}, p_{\perp,2}$) [GeV/$c$]")

axes[1, 1].set_xlabel(r"min($p_{\perp,1}, p_{\perp,2}$) in slice (median) [GeV/$c$]")
axes[0, 0].legend()
fig.suptitle(f"Selection on the softer muon's $p_\\perp$ ({args.dataset} data, three-peak {BASELINE_SIGNAL} fits)", fontsize=24)
fig.tight_layout()
ups.plotting.save_figure(fig, f"{output_subdir(args.dataset)}/cut_scan.png")

