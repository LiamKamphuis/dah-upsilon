"""Brief task 8: the Upsilon(1S, 2S, 3S) masses with statistical and systematic uncertainties.

    m = mu_fit - shape bias   +- stat   +- fit model   +- momentum scale

1. Baseline: the simultaneous fit in (y, pT) cells of 04_kinematic_bins.py, on the selected
   events in the fiducial region: double Crystal Ball (tails shared, widths tied, one sigma_1S
   per cell) + exponential per cell; the three masses are common to all cells.
2. Shape bias: measured on simulation with the same selection and fiducial region (fitted mu -
   generated mass), scaled to each peak by the width the cell fit effectively sees
   (masses.effective_widths) and subtracted; its MC error goes into stat.
3. Fit-model systematic: refit with one analysis choice changed at a time and take the shift of
   the bias-corrected masses: signal shape (Crystal Ball in the cells), the softer-muon cut, and
   no kinematic cells (one inclusive fit). A change of
   signal shape or selection also changes the MC bias, so each variation is corrected with its
   own. Shifts are added in quadrature.
4. Momentum scale: 3e-4 relative (arXiv:2312.01772), the same for every peak.
5. Cross-check: calibrate the scale on the 1S and compare the 2S, 3S masses with PDG.
"""
import argparse
from functools import cache

import numpy as np

import upsilon as ups
from upsilon import masses
from upsilon.fitting import fiducial, fit_kinematic_cells
from upsilon.models import BASELINE_SIGNAL, PDG_MASS, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
args = parser.parse_args()

SCALE_ERR = 3e-4
MEV = 1e3
PEAKS = masses.PEAKS
pdg = np.array([PDG_MASS[p] for p in PEAKS])
events, mc = ups.load(args.dataset), ups.load("mc")


def selected(sample, cut=ups.io.MIN_MUON_PT):
    """Softer-muon cut and the fiducial region of the kinematic cells."""
    sample = ups.select_muons(sample, cut)
    return sample[fiducial(sample)]


def model(signal=BASELINE_SIGNAL):
    return ups.build_model(signal=signal, share=share_tails(signal), tie_widths=True)


@cache
def bias(signal, cut):
    return masses.shape_bias(signal, selected(mc, cut)["mass"])


reference = {}  # shape values of the baseline inclusive fit: the start for every later fit


def measure(signal=BASELINE_SIGNAL, cut=ups.io.MIN_MUON_PT, cells=True):
    """Bias-corrected masses and covariance (fit + MC statistics), and whether the fit converged.

    First one inclusive fit to all selected events (started from the baseline's shape: from the
    default start the double CB can collapse into a false minimum); with ``cells``, that fit
    then starts the simultaneous fit in the kinematic cells.
    """
    sample, m = selected(events, cut), model(signal)
    inclusive = ups.fit(m, sample["mass"], binned=True, progress=False,
                        start={k: v for k, v in reference.items() if k in m.names})
    if not cells:
        mu, cov = masses.mass_vector(inclusive)
        return (*masses.apply_bias(mu, cov, bias(signal, cut), masses.widths(inclusive)), inclusive.valid, inclusive)
    _, results = fit_kinematic_cells(m, sample, signal, inclusive.values)
    mu, cov = masses.mass_vector(results[0])
    sigma = masses.effective_widths(results)
    return (*masses.apply_bias(mu, cov, bias(signal, cut), sigma), results[0].minuit.valid, inclusive)


mu, cov, valid, inclusive = measure()
reference.update({k: v for k, v in inclusive.values.items() if not k.startswith("N_")})
b = bias(BASELINE_SIGNAL, ups.io.MIN_MUON_PT)
print(f"baseline (kinematic cells) valid={valid}; MC shape bias {MEV * b.value:+.3f} +- {MEV * b.error:.3f} MeV "
      f"at sigma_MC = {MEV * b.sigma:.1f} MeV")

variations = {
    "Crystal Ball signal": lambda: measure("crystal_ball"),
    "muon pT cut 2.5 GeV": lambda: measure(cut=2.5),
    "muon pT cut 3.5 GeV": lambda: measure(cut=3.5),
    "no kinematic cells": lambda: measure(cells=False),
}
shifts, converged = {}, {}
for name, run in variations.items():
    v_mu, _, converged[name], _ = run()
    shifts[name] = v_mu - mu
    print(f"  {name}: valid={converged[name]}")
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
sample = selected(events)
print("\ncross-check: sideband-subtracted mean dimuon momentum per peak")
mean_p = {}
for p, m in zip(PEAKS, pdg):
    side = ((m - 0.12, m - 0.07), (m + 0.07, m + 0.12))
    n, _ = ups.sideband_subtract(sample["mass"], (m - 0.05, m + 0.05), side)
    total, _ = ups.sideband_subtract(sample["mass"], (m - 0.05, m + 0.05), side, weights=sample["p"])
    mean_p[p] = total / n
    print(f"  {p}: <p> = {mean_p[p]:.1f} GeV/c")
smu, scov = masses.scale_corrected_masses(mu, cov)
for i in (1, 2):
    print(f"  calibrated on 1S: {PEAKS[i]} - PDG = {MEV * (smu[i] - pdg[i]):+.2f} +- {MEV * np.sqrt(scov[i, i]):.2f} MeV")
sf = masses.common_scale(mu, cov, pdg)
print(f"  common scale s - 1 = ({1e4 * (sf.s - 1):+.2f} +- {1e4 * sf.error:.2f}) x 1e-4, "
      f"chi2/ndof = {sf.chi2:.2f}/{sf.ndof}, p = {sf.pvalue:.2g}")

ups.save_json({"peaks": PEAKS, "mass": mu, "stat": stat, "fit_model": fit_model, "scale": scale,
               "shifts": shifts, "pdg": pdg, "valid": valid, "variations_valid": converged,
               "bias": vars(b), "mean_p": mean_p,
               "calibrated_on_1S": smu, "calibrated_err": np.sqrt(np.diag(scov)),
               "common_scale": vars(sf)}, "final_masses.json", args.dataset)
