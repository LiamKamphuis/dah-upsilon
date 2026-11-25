"""Compare signal shapes beyond a single Gaussian, all in one table by default.

For each shape: fit simulation (pure signal) to get the tail parameters, then fit
data twice, once with the tails fixed from simulation and once with them floating
(shared across peaks). All fits use the same data and binning, so the Delta columns
compare every model against every other. ``--signal`` restricts it to one shape.
"""
import argparse

import upsilon as ups
from upsilon.io import output_subdir
from upsilon.models import PDG_MASS, SIGNALS, TAIL_PARAMS, share_tails

SHAPES = [s for s in SIGNALS if TAIL_PARAMS[s]]  # shapes with tail parameters (not gauss)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--signal", default="all", choices=["all", *SHAPES])
parser.add_argument("--binned", action="store_true", help="much faster; recommended for --dataset full")
args = parser.parse_args()

mc = ups.select_muons(ups.load("mc"))["mass"]  # same selection on simulation and data
mass = ups.select_muons(ups.load(args.dataset))["mass"]
fits = {"gauss": ups.fit(ups.build_model(), mass, binned=args.binned)}
# Selected-data plots are named shape_fit_* so they do not overwrite 01's unselected fit_3peaks_*
ups.plotting.plot_fit(fits["gauss"], save=f"{output_subdir(args.dataset)}/shape_fit_gauss.png")
mc_fits = {}

# Fit MC first to pin tail parameters, then use fixed tails on dataset.
for signal in SHAPES if args.signal == "all" else [args.signal]:
    # background=None as simulation is pure signal, unbinned as MC dataset is small
    mc_fit = ups.fit(ups.build_model(("1S",), signal=signal, window=(9.0, 9.9), background=None), mc)
    mc_fits[signal] = mc_fit
    print(f"\n{signal}: simulation\n" + mc_fit.summary())

    fixed = {f"{p}_1S": mc_fit.values[f"{p}_1S"] for p in TAIL_PARAMS[signal]}
    model = ups.build_model(signal=signal, share=share_tails(signal), tie_widths=True)
    fits[f"{signal} (MC tails)"] = ups.fit(model, mass, fix=fixed, binned=args.binned)
    fits[f"{signal} (shared tails)"] = shared = ups.fit(model, mass, binned=args.binned)
    print(f"\n{signal}: data, shared tails\n" + shared.summary())
    for label, r in (("simulation", mc_fit), ("data", shared)):  # what the fit cannot separate
        print(f"{signal}, {label}: |correlation| >= 0.5: " + ", ".join(
            f"{a}/{b} {rho:+.2f}" for a, b, rho in ups.compare.strong_correlations(r)))
    ups.plotting.plot_fit(shared, save=f"{output_subdir(args.dataset)}/shape_fit_{signal}.png")

print("\nModel comparison\n" + ups.compare.table(fits))
comparison = {}
for name, r in fits.items():
    ic, g, ru = ups.compare.information_criteria(r), ups.compare.gof(r), ups.compare.runs_test(r)
    comparison[name] = {"k": ic.k, "chi2": g.stat, "ndof": g.ndof, "p": g.pvalue, "runs_p": ru.pvalue,
                        "aic": ic.aic, "bic": ic.bic}

# Nested test with the simple model inside the parameter space (tails at the MC values) so Wilks holds
# Essentially this tests whether the data's tails are compatible with the simulation's
tails_test = {}
print("\nAre data tails compatible with MC tails? (likelihood ratio, chi2 with Delta k = tail parameters)")
for signal in SHAPES if args.signal == "all" else [args.signal]:
    lr = ups.compare.likelihood_ratio(fits[f"{signal} (MC tails)"], fits[f"{signal} (shared tails)"])
    tails_test[signal] = {"m2dlnl": lr.stat, "ndof": lr.ndof, "pvalue": lr.pvalue}
    print(f"  {signal:<14} -2 Delta lnL = {lr.stat:8.1f}  (Delta k = {lr.ndof})  p = {lr.pvalue:.3g}")

# Does the shape choice move the mass by more than its statistical error?
NICE = {"gauss": "Gaussian", "double_gauss": "Double Gaussian", "crystal_ball": "Crystal Ball", "double_cb": "Double CB"}
profiled = {NICE[name.split()[0]]: r for name, r in fits.items() if name == "gauss" or name.endswith("(shared tails)")}
ups.plotting.plot_profiles(profiled, "mu_1S", ref=PDG_MASS["1S"], xlabel=r"$\mu_{1S} - m_\mathrm{PDG}$ [MeV/$c^2$]",
                           save=f"{output_subdir(args.dataset)}/profile_mu_1S.png")

ups.save_json({"mc": {k: r.to_dict() for k, r in mc_fits.items()},
                   "data": {k: r.to_dict() for k, r in fits.items()},
                   "tails_test": tails_test, "comparison": comparison}, "signal_shape.json", args.dataset)
