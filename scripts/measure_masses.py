"""Measure the Upsilon(1S, 2S, 3S) masses with a signal shape (default: the baseline) + exponential model.

``--signal gauss`` gives the single-Gaussian fits of brief tasks 1-4.

Plots the observables, fits the 1S peak alone, fits all three peaks
simultaneously, and shows the residuals and pulls of the three-peak fit.
"""
import argparse

import upsilon as ups
from upsilon.io import RESULT_DIR
from upsilon.models import BASELINE_SIGNAL, SIGNALS, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small", help="small | full | mc | path to .bin")
parser.add_argument("--binned", action="store_true", help="binned likelihood (much faster)")
parser.add_argument("--signal", default=BASELINE_SIGNAL, choices=sorted(SIGNALS), help="signal shape")
args = parser.parse_args()
tag = f"{args.signal}_{args.dataset}"  # fit outputs per signal shape

events = ups.load(args.dataset)
print(f"{len(events)} events")
ups.plotting.plot_variables(
    events,
    ranges={"pt": (0, 30), "eta": (1.5, 5.5), "p": (0, 500), "pt1": (0, 20), "pt2": (0, 20)},
    save=f"variables_{args.dataset}.png",
)

# Upsilon(1S) alone
fit_1s = ups.fit(ups.build_model(("1S",), signal=args.signal, window=(9.0, 9.9)), events["mass"], binned=args.binned, minos=True)
print("\nUpsilon(1S) fit\n" + fit_1s.summary())
fit_1s.save(RESULT_DIR / f"fit_1S_{tag}.json")
ups.plotting.plot_fit(fit_1s, bins=90, title=r"$\Upsilon(1S)$ fit", save=f"fit_1S_{tag}.png")

# All three peaks
# One set of tail parameters for all peaks: per-peak tails are not identifiable on the full data
fit_3 = ups.fit(ups.build_model(signal=args.signal, share=share_tails(args.signal)), events["mass"], binned=args.binned)
print("\nThree-peak fit\n" + fit_3.summary())
fit_3.save(RESULT_DIR / f"fit_3peaks_{tag}.json")
ups.plotting.plot_fit(fit_3, pulls=False, title=r"$\Upsilon(nS)$ fit", save=f"fit_3peaks_{tag}.png")
ups.plotting.plot_residuals(fit_3, save=f"residuals_3peaks_{tag}.png")
ups.plotting.plot_fit(fit_3, title=r"$\Upsilon(nS)$ fit with pulls", save=f"pulls_3peaks_{tag}.png")
