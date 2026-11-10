"""Systematic uncertainties on the masses: refit with one analysis choice changed at a time.

Each shift from the baseline is a candidate systematic. Data are shared
between fits, so judge each shift against the statistical spread of the
*difference* (Barlow, hep-ex/0207026), not against either fit's error.
"""
import argparse

import numpy as np

import upsilon as ups

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
args = parser.parse_args()

mass = ups.load(args.dataset)["mass"]
PARAMS = ("mu_1S", "mu_2S", "mu_3S")

baseline = ups.fit(ups.build_model(), mass, binned=True)
variations = {
    "fit window 9.1-10.9": lambda: ups.fit(ups.build_model(window=(9.1, 10.9)), mass, binned=True),
    "bin width x2": lambda: ups.fit(ups.build_model(), mass, binned=True, bins=125),
    "unbinned": lambda: ups.fit(ups.build_model(), mass),
    # TODO "Crystal Ball signal": lambda: ups.fit(ups.build_model(signal="crystal_ball"), mass, binned=True),
    # TODO "momentum scale": lambda: ups.fit(ups.build_model(), mass * (1 + delta), binned=True),
    # TODO alternative background shape (add to models.BACKGROUNDS)
}

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
