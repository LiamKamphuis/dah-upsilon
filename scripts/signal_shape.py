"""Compare signal shapes beyond a single Gaussian, all in one table by default.

For each shape: fit simulation (pure signal) to get the tail parameters, then fit
data twice, once with the tails fixed from simulation and once with them floating
(shared across peaks). All fits use the same data and binning, so the Delta columns
compare every model against every other. ``--signal`` restricts it to one shape.
"""
import argparse

import upsilon as ups
from upsilon.models import PDG_MASS, SIGNALS, TAIL_PARAMS, share_tails

SHAPES = [s for s in SIGNALS if TAIL_PARAMS[s]]  # shapes with tail parameters (not gauss)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--signal", default="all", choices=["all", *SHAPES])
parser.add_argument("--binned", action="store_true", help="much faster; recommended for --dataset full")
args = parser.parse_args()

mc = ups.load("mc")["mass"]
mass = ups.load(args.dataset)["mass"]
fits = {"gauss": ups.fit(ups.build_model(), mass, binned=args.binned)}

for signal in SHAPES if args.signal == "all" else [args.signal]:
    # Simulation is pure signal: a background term would absorb the radiative tail
    mc_fit = ups.fit(ups.build_model(("1S",), signal=signal, window=(9.0, 9.9), background=None), mc)
    print(f"\n{signal}: simulation\n" + mc_fit.summary())

    fixed = {f"{p}_1S": mc_fit.values[f"{p}_1S"] for p in TAIL_PARAMS[signal]}
    model = ups.build_model(signal=signal, share=share_tails(signal))
    fits[f"{signal} (MC tails)"] = ups.fit(model, mass, fix=fixed, binned=args.binned)
    fits[f"{signal} (shared tails)"] = shared = ups.fit(model, mass, binned=args.binned)
    print(f"\n{signal}: data, shared tails\n" + shared.summary())
    ups.plotting.plot_fit(shared, save=f"fit_3peaks_{signal}_{args.dataset}.png")

print("\nModel comparison\n" + ups.compare.table(fits))

# Nested test with the simple model inside the parameter space (tails at the MC values),
# so Wilks holds: are the data's tails compatible with simulation's?
print("\nAre data tails compatible with MC tails? (likelihood ratio, chi2 with Delta k = tail parameters)")
for signal in SHAPES if args.signal == "all" else [args.signal]:
    lr = ups.compare.likelihood_ratio(fits[f"{signal} (MC tails)"], fits[f"{signal} (shared tails)"])
    print(f"  {signal:<14} -2 Delta lnL = {lr.stat:8.1f}  (Delta k = {lr.ndof})  p = {lr.pvalue:.3g}")

# Does the shape choice move the mass by more than its statistical error?
profiled = {name: r for name, r in fits.items() if name == "gauss" or name.endswith("(shared tails)")}
ups.plotting.plot_profiles(profiled, "mu_1S", ref=PDG_MASS["1S"], save=f"profile_mu_1S_{args.dataset}.png")
