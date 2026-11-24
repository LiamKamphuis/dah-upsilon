"""Use the other observables: their correlation with mass, cuts for purity, and kinematic bins."""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import upsilon as ups
from upsilon.io import FIGURE_DIR, RESULT_DIR, output_subdir
from upsilon.models import BASELINE_SIGNAL, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
args = parser.parse_args()

events = ups.load(args.dataset)
for col in ("pt", "eta", "p", "pt1", "pt2"):
    ups.plotting.plot_vs_mass(events, col, save=f"{output_subdir(args.dataset)}/{col}_vs_mass.png")

# Purity scan: cut on the softer muon's pT (combinatorial background is mostly a soft
# muon paired with anything; dimuon pT and eta barely separate signal from background).
# Each cut gets one three-peak baseline fit, which gives both figures of merit:
#   - S, B in mu +- 2 sigma per peak -> purity S/(S+B) and S/sqrt(S+B)
#   - the HESSE error on each mass, which is what the measurement actually needs.
CUTS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)  # GeV; 1.0 = no cut (the data start there)
PEAKS = ("1S", "2S", "3S")
pt_min = np.minimum(events["pt1"], events["pt2"])
model = ups.build_model(signal=BASELINE_SIGNAL, share=share_tails(BASELINE_SIGNAL))

scan = []
for cut in CUTS:
    r = ups.fit(model, events["mass"][pt_min >= cut], binned=True, progress=False)
    row = {"cut": cut, "valid": r.valid}
    for p in PEAKS:
        mu, sigma = r.values[f"mu_{p}"], r.values[f"sigma_{p}"]
        counts = r.model.component_counts([mu - 2 * sigma, mu + 2 * sigma], r.values)
        s, b = counts[p][0], sum(c[0] for k, c in counts.items() if k != p)  # other peaks count as background
        row[p] = {"mu": mu, "err": r.errors[f"mu_{p}"], "S": s, "B": b,
                  "purity": s / (s + b), "signif": s / np.sqrt(s + b)}
    scan.append(row)
    print(f"min muon pT >= {cut:.1f} GeV  valid={r.valid}  " + "  ".join(
        f"{p}: P={row[p]['purity']:.3f} S/sqrt(S+B)={row[p]['signif']:.0f} "
        f"err={1e3 * row[p]['err']:.3f} MeV" for p in PEAKS))

(RESULT_DIR / output_subdir(args.dataset)).mkdir(parents=True, exist_ok=True)
(RESULT_DIR / output_subdir(args.dataset) / "cut_scan.json").write_text(json.dumps(scan, indent=2, default=float))

cuts = np.array(CUTS)
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for i, p in enumerate(PEAKS):
    get = lambda key: np.array([row[p][key] for row in scan])  # noqa: E731
    axes[0, 0].plot(cuts, get("purity"), "o-", label=rf"$\Upsilon({p})$")
    axes[0, 1].plot(cuts, get("signif") / get("signif")[0], "o-")
    axes[1, 0].plot(cuts, get("err") / get("err")[0], "o-")
    # mass stability: shift from the no-cut fit, offset sideways so the error bars stay readable
    axes[1, 1].errorbar(cuts + 0.04 * (i - 1), 1e3 * (get("mu") - get("mu")[0]), yerr=1e3 * get("err"), fmt="o")
axes[0, 0].set(ylabel=r"Purity $S/(S+B)$ in $\mu\pm2\sigma$", title="Purity")
axes[0, 1].set(ylabel=r"$S/\sqrt{S+B}$ relative to no cut", title="Signal significance")
axes[1, 0].set(ylabel=r"$\sigma(\mu)$ relative to no cut", title="Statistical error on the mass (lower is better)")
axes[1, 1].set(ylabel=r"$\mu - \mu_\mathrm{no\ cut}$ [MeV/$c^2$]", title="Mass stability")
axes[1, 1].axhline(0, color="k", lw=0.8)
for ax in axes[1]:
    ax.set_xlabel(r"min($p_{\perp,1}, p_{\perp,2}$) cut [GeV/$c$]")
axes[0, 0].legend()
fig.suptitle(f"Selection scan on the softer muon $p_\\perp$ ({args.dataset} data, {BASELINE_SIGNAL} fits)")
fig.tight_layout()
(FIGURE_DIR / output_subdir(args.dataset)).mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_DIR / output_subdir(args.dataset) / "cut_scan.png", dpi=150, bbox_inches="tight")

#
# TODO kinematic bins: one model per (pt, eta) cell with per-cell parameters
# renamed by cell; masses keep a common name. Summing iminuit costs
# (cost_a + cost_b) shares parameters with equal names.
