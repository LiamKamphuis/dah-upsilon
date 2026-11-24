"""Brief task 8: the Upsilon(1S, 2S, 3S) masses with statistical and systematic uncertainties.

    m = mu_fit - shape bias   +- stat   +- fit model   +- momentum scale

1. Baseline: double Crystal Ball (tails shared, widths tied) + exponential, binned fit to the
   selected data. The shape bias is measured on simulation with the same selection
   (fitted mu - generated mass) and subtracted; its MC error goes into stat.
2. Fit-model systematic: refit with one analysis choice changed at a time and take the shift of
   the bias-corrected masses. A change of signal shape or selection also changes the MC bias, so
   each variation is corrected with its own. Shifts are added in quadrature.
3. Momentum scale: 3e-4 relative (arXiv:2312.01772), the same for every peak.
4. Cross-check: calibrate the scale on the 1S and compare the 2S, 3S masses with PDG.
"""
import argparse
from functools import cache

import numpy as np

import upsilon as ups
from upsilon import masses
from upsilon.models import BASELINE_SIGNAL, PDG_MASS, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--unbinned", action="store_true", help="add an unbinned-fit variation (slow)")
args = parser.parse_args()

SCALE_ERR = 3e-4
MEV = 1e3
PEAKS = masses.PEAKS
pdg = np.array([PDG_MASS[p] for p in PEAKS])
events, mc = ups.load(args.dataset), ups.load("mc")


def model(signal=BASELINE_SIGNAL, **kw):
    return ups.build_model(signal=signal, share=share_tails(signal), tie_widths=True, **kw)


@cache
def bias(signal, cut):
    return masses.shape_bias(signal, ups.select_muons(mc, cut)["mass"])


def corrected(result, signal=BASELINE_SIGNAL, cut=ups.io.MIN_MUON_PT):
    """Bias-corrected masses and covariance (fit + MC statistics) of a fit."""
    mu, cov = masses.mass_vector(result)
    return masses.apply_bias(mu, cov, bias(signal, cut), masses.widths(result))


data = ups.select_muons(events)["mass"]
baseline = ups.fit(model(), data, binned=True)
print(f"baseline valid={baseline.valid}, MC shape bias {MEV * bias(BASELINE_SIGNAL, ups.io.MIN_MUON_PT).value:+.2f} MeV")
mu, cov = corrected(baseline)


def refit(m, x, **kw):
    """Binned fit started from the baseline values: from the default start the double CB can
    fall into a false minimum where it collapses to one CB."""
    return ups.fit(m, x, binned=True, start={k: v for k, v in baseline.values.items() if k in m.names},
                   progress=False, **kw)


def cut(c):
    return lambda: corrected(refit(model(), ups.select_muons(events, c)["mass"]), cut=c)


variations = {
    "fit window 9.1-10.9": lambda: corrected(refit(model(window=(9.1, 10.9)), data)),
    "bin width x2": lambda: corrected(refit(model(), data, bins=125)),
    "Crystal Ball signal": lambda: corrected(refit(model("crystal_ball"), data), signal="crystal_ball"),
    "quadratic background": lambda: corrected(refit(model(background="poly2"), data)),
    "muon pT cut 2.5 GeV": cut(2.5),
    "muon pT cut 3.5 GeV": cut(3.5),
}
if args.unbinned:
    variations["unbinned"] = lambda: corrected(ups.fit(model(), data, start=baseline.values))

shifts = {name: run()[0] - mu for name, run in variations.items()}
fit_model = np.sqrt(sum(s**2 for s in shifts.values()))
stat = np.sqrt(np.diag(cov))
scale = SCALE_ERR * mu

print(f"\n{'[MeV/c^2]':<24}" + "".join(f"{p:>10}" for p in PEAKS))
print(f"{'stat (fit + MC bias)':<24}" + "".join(f"{MEV * v:>10.3f}" for v in stat))
for name, s in shifts.items():
    print(f"{name:<24}" + "".join(f"{MEV * v:>+10.3f}" for v in s))
print(f"{'fit model (quadrature)':<24}" + "".join(f"{MEV * v:>10.3f}" for v in fit_model))
print(f"{'momentum scale (3e-4)':<24}" + "".join(f"{MEV * v:>10.3f}" for v in scale))

print("\nmass [MeV/c^2] = value +- stat +- fit model +- momentum scale;  - PDG")
for k, p in enumerate(PEAKS):
    print(f"  {p}: {MEV * mu[k]:.2f} +- {MEV * stat[k]:.2f} +- {MEV * fit_model[k]:.2f} +- {MEV * scale[k]:.2f}"
          f"   ({MEV * (mu[k] - pdg[k]):+.2f})")

# Cross-check: one momentum scale for all peaks? Calibrating on the 1S assumes the peaks share
# a momentum spectrum, since alpha depends on p (05_momentum_scale.py): check the mean p per peak.
selected = ups.select_muons(events)
print("\ncross-check: sideband-subtracted mean dimuon momentum per peak")
mean_p = {}
for p, m in zip(PEAKS, pdg):
    side = ((m - 0.12, m - 0.07), (m + 0.07, m + 0.12))
    n, _ = ups.sideband_subtract(selected["mass"], (m - 0.05, m + 0.05), side)
    total, _ = ups.sideband_subtract(selected["mass"], (m - 0.05, m + 0.05), side, weights=selected["p"])
    mean_p[p] = total / n
    print(f"  {p}: <p> = {mean_p[p]:.1f} GeV/c")
smu, scov = masses.scale_corrected_masses(mu, cov)
for i in (1, 2):
    print(f"  calibrated on 1S: {PEAKS[i]} - PDG = {MEV * (smu[i] - pdg[i]):+.2f} +- {MEV * np.sqrt(scov[i, i]):.2f} MeV")
sf = masses.common_scale(mu, cov, pdg)
print(f"  common scale s - 1 = ({1e4 * (sf.s - 1):+.2f} +- {1e4 * sf.error:.2f}) x 1e-4, "
      f"chi2/ndof = {sf.chi2:.2f}/{sf.ndof}, p = {sf.pvalue:.2g}")

ups.save_json({"peaks": PEAKS, "mass": mu, "stat": stat, "fit_model": fit_model, "scale": scale,
               "shifts": shifts, "pdg": pdg, "valid": baseline.valid, "mean_p": mean_p,
               "calibrated_on_1S": smu, "calibrated_err": np.sqrt(np.diag(scov)),
               "common_scale": vars(sf)}, "final_masses.json", args.dataset)
