"""Measure the Upsilon(1S, 2S, 3S) masses with a signal shape + exponential model.

The default, ``--signal gauss``, gives the single-Gaussian fits of brief tasks 1-4;
``--signal double_cb`` is the baseline shape used everywhere else.

Plots the observables, fits the 1S peak alone, fits all three peaks
simultaneously, and shows the residuals and pulls of the three-peak fit.
"""
import argparse

import numpy as np

import upsilon as ups
from upsilon.io import output_subdir
from upsilon.models import SIGNALS, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small", help="small | full | mc | path to .bin")
parser.add_argument("--binned", action="store_true", help="binned likelihood (much faster)")
parser.add_argument("--signal", default="gauss", choices=sorted(SIGNALS), help="signal shape")
args = parser.parse_args()
out = output_subdir(args.dataset)  # outputs go to output/{figures,results}/<dataset>/
sim = args.dataset == "mc"

events = ups.load(args.dataset)
print(f"{len(events)} events")
ups.plotting.plot_variables(
    events,
    ranges={"pt": (0, 30), "eta": (1.5, 5.5), "p": (0, 500), "pt1": (0, 20), "pt2": (0, 20)},
    save=f"{out}/variables.png",
)

# Upsilon(1S) alone
fit_1s = ups.fit(ups.build_model(("1S",), signal=args.signal, window=(9.0, 9.9)),
                 events["mass"], binned=args.binned, minos=True)
print("\nUpsilon(1S) fit\n" + fit_1s.summary())
ups.save_json(fit_1s.to_dict(), f"fit_1S_{args.signal}.json", args.dataset)
ups.plotting.plot_fit(fit_1s, bins=90, title=r"$\Upsilon(1S)$ fit",
                      simulation=sim, save=f"{out}/fit_1S_{args.signal}.png")

# Check errors by hand from - 2 ln L plot (where it equals 1)
SCAN = ("mu_1S", "sigma_1S")
ups.plotting.plot_error_scans(fit_1s, SCAN, save=f"{out}/error_scans_1S_{args.signal}.png")
N_sig, sigma = fit_1s.values["N_1S"], fit_1s.values["sigma_1S"]
by_hand = {"mu_1S": sigma / np.sqrt(N_sig), "sigma_1S": sigma / np.sqrt(2 * N_sig)} # Gaussian, std. error on mean
print(f"\n1S errors [MeV/c^2]   {'HESSE':>8}{'MINOS -/+':>18}{'scan, profiled':>20}{'scan, fixed':>18}{'sigma/sqrt(N)':>15}")
for name in SCAN:
    best, me = fit_1s.values[name], fit_1s.minuit.merrors[name]
    prof = ups.compare.crossings(*ups.compare.profile_scan(fit_1s, name, size=41))
    fixed = ups.compare.crossings(*ups.compare.fixed_scan(fit_1s, name))
    print(f"{name:<22}{1e3 * fit_1s.errors[name]:>8.4f}{1e3 * me.lower:>9.4f}/{1e3 * me.upper:+.4f}"
          f"{1e3 * (prof[0] - best):>10.4f}/{1e3 * (prof[1] - best):+.4f}"
          f"{1e3 * (fixed[0] - best):>9.4f}/{1e3 * (fixed[1] - best):+.4f}{1e3 * by_hand[name]:>13.4f}")

# All three peaks
# One set of tail parameters for all peaks: per-peak tails are not identifiable on the full data
fit_3 = ups.fit(ups.build_model(signal=args.signal, share=share_tails(args.signal)),
                events["mass"], binned=args.binned)
print("\nThree-peak fit\n" + fit_3.summary())
ups.save_json(fit_3.to_dict(), f"fit_3peaks_{args.signal}.json", args.dataset)
ups.plotting.plot_fit(fit_3, pulls=False, title=r"$\Upsilon(nS)$ fit", simulation=sim,
                      save=f"{out}/fit_3peaks_{args.signal}.png")
ups.plotting.plot_residuals(fit_3, save=f"{out}/residuals_3peaks_{args.signal}.png")
ups.plotting.plot_fit(fit_3, title=r"$\Upsilon(nS)$ fit with pulls", simulation=sim,
                      save=f"{out}/pulls_3peaks_{args.signal}.png")
