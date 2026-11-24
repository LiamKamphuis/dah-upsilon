"""Mass differences and momentum-scale-corrected masses, with signal-shape bias from simulation.

For each signal shape: fit data (one set of tail parameters shared by all peaks), measure the shape bias on
simulation, subtract it, then compare mass differences and scale-corrected masses with PDG.
"""
import argparse

import numpy as np

import upsilon as ups
from upsilon import masses
from upsilon.models import BASELINE_SIGNAL, PDG_MASS, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--binned", action="store_true")
parser.add_argument("--signals", nargs="+", default=[BASELINE_SIGNAL, "crystal_ball", "double_gauss", "gauss"])
args = parser.parse_args()

events = ups.load(args.dataset)
data = events["mass"]
mc = ups.load("mc")["mass"]
pdg = np.array([PDG_MASS[p] for p in masses.PEAKS])
MEV = 1e3

# Calibrating on 1S assumes all peaks see the same momentum scale. The scale drifts with dimuon
# momentum (scripts/momentum_scale.py), so check the peaks' momentum spectra match: mean p in
# +-50 MeV around each peak, minus equal-width sidebands at 70-120 MeV (linear background).
print("sideband-subtracted mean dimuon momentum per peak:")
for p, m_pdg in zip(masses.PEAKS, pdg):
    d = np.abs(data - m_pdg)
    sig, side = d < 0.05, (d > 0.07) & (d < 0.12)
    mean_p = (events["p"][sig].sum() - events["p"][side].sum()) / (sig.sum() - side.sum())
    print(f"  {p}: <p> = {mean_p:.1f} GeV/c")

for signal in args.signals:
    result = ups.fit(ups.build_model(signal=signal, share=share_tails(signal)), data, binned=args.binned)
    if not result.valid:
        print(f"\nWARNING: {signal} data fit is not valid; its covariance (and every error below) is unreliable")
        print(result.minuit.fmin)
    bias = masses.shape_bias(signal, mc)
    raw, raw_cov = masses.mass_vector(result)
    mu, cov = masses.apply_bias(raw, raw_cov, bias, masses.widths(result))

    print(f"\n=== {signal}  (MC shape bias {MEV * bias.value:+.2f} +- {MEV * bias.error:.2f} MeV) ===")
    print(f"{'peak':<6}{'raw - PDG':>16}{'corrected - PDG':>20}   [MeV]")
    for k, p in enumerate(masses.PEAKS):
        print(f"{p:<6}{MEV * (raw[k] - pdg[k]):>+10.2f} +- {MEV * np.sqrt(raw_cov[k, k]):.2f}"
              f"{MEV * (mu[k] - pdg[k]):>+12.2f} +- {MEV * np.sqrt(cov[k, k]):.2f}")

    print("mass differences - PDG:")
    for i in (1, 2):
        d, e = masses.mass_difference(mu, cov, i, 0)
        print(f"  m({masses.PEAKS[i]}) - m(1S): {MEV * (d - (pdg[i] - pdg[0])):+.2f} +- {MEV * e:.2f} MeV")

    smu, scov = masses.scale_corrected_masses(mu, cov)
    print("scale-corrected (calibrated on 1S) - PDG:")
    for i in (1, 2):
        print(f"  {masses.PEAKS[i]}: {MEV * (smu[i] - pdg[i]):+.2f} +- {MEV * np.sqrt(scov[i, i]):.2f} MeV")

    sf = masses.common_scale(mu, cov, pdg)
    w = np.linalg.solve(cov / np.outer(pdg, pdg), np.ones(len(pdg)))
    print(f"common scale: s - 1 = ({1e4 * (sf.s - 1):+.2f} +- {1e4 * sf.error:.2f}) x 1e-4, "
          f"chi2/ndof = {sf.chi2:.2f}/{sf.ndof}, p = {sf.pvalue:.3g}, "
          f"weights ({', '.join(masses.PEAKS)}) = {np.round(w / w.sum(), 3)}")
