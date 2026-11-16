"""Systematic uncertainties on the masses: refit with one analysis choice changed at a time.

Each shift from the baseline is a candidate systematic. Data are shared
between fits, so judge each shift against the statistical spread of the
*difference* (Barlow, hep-ex/0207026), not against either fit's error.
"""
import argparse

import numpy as np

import upsilon as ups
from upsilon.models import BASELINE_SIGNAL, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--unbinned", action="store_true",
                    help="add the unbinned-fit variation (slow: ~10-20 min on small with the double CB baseline)")
args = parser.parse_args()

mass = ups.load(args.dataset)["mass"]
PARAMS = ("mu_1S", "mu_2S", "mu_3S")


def model(signal=BASELINE_SIGNAL, **kw):
    return ups.build_model(signal=signal, share=share_tails(signal), **kw)


baseline = ups.fit(model(), mass, binned=True)
variations = {
    "fit window 9.1-10.9": lambda: ups.fit(model(window=(9.1, 10.9)), mass, binned=True),
    "bin width x2": lambda: ups.fit(model(), mass, binned=True, bins=125),
    "Crystal Ball signal": lambda: ups.fit(model("crystal_ball"), mass, binned=True),
    # TODO "momentum scale": lambda: ups.fit(model(), mass * (1 + delta), binned=True),
    # TODO alternative background shape (add to models.BACKGROUNDS)
}
if args.unbinned:
    variations["unbinned"] = lambda: ups.fit(model(), mass)

shifts = {}
for name, run in variations.items():
    result = run()
    shifts[name] = {p: result.values[p] - baseline.values[p] for p in PARAMS}

print(f"{'variation':<22}" + "".join(f"{p:>14}" for p in PARAMS) + "   [MeV/c^2]")
print(f"{'baseline (stat)':<22}" + "".join(f"{1e3 * baseline.errors[p]:>14.3f}" for p in PARAMS))
for name, s in shifts.items():
    print(f"{name:<22}" + "".join(f"{1e3 * s[p]:>14.3f}" for p in PARAMS))
total = {p: np.sqrt(sum(s[p] ** 2 for s in shifts.values())) for p in PARAMS}
print(f"{'quadrature sum':<22}" + "".join(f"{1e3 * total[p]:>14.3f}" for p in PARAMS))
