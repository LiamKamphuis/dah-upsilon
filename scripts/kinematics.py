"""Use the other observables: their correlation with mass, cuts for purity, and kinematic bins."""
import argparse

import upsilon as ups

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
args = parser.parse_args()

events = ups.load(args.dataset)
for col in ("pt", "eta", "p", "pt1", "pt2"):
    ups.plotting.plot_vs_mass(events, col, save=f"{col}_vs_mass_{args.dataset}.png")

# TODO purity: fit a selection, then signal / total in a mass region, e.g.
#   result = ups.fit(ups.build_model(), ups.select(events, pt=(2.0, None))["mass"], binned=True)
#   counts = result.model.component_counts([9.40, 9.52], result.values)
#
# TODO kinematic bins: one model per (pt, eta) cell with per-cell parameters
# renamed by cell; masses keep a common name. Summing iminuit costs
# (cost_a + cost_b) shares parameters with equal names.
