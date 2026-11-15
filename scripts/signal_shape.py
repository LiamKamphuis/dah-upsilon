"""Compare signal shapes beyond a single Gaussian

Fits the shape to simulation first, where the tails are clean, then fits data
with the tail parameters fixed from simulation and shared across peaks.
"""
import argparse

import upsilon as ups

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--signal", default="crystal_ball", help="crystal_ball | double_gauss")
parser.add_argument("--binned", action="store_true")
args = parser.parse_args()

# Simulation is pure signal: a background term would absorb the radiative tail
mc_model = ups.build_model(("1S",), signal=args.signal, window=(9.0, 9.9), background=None)
mc_fit = ups.fit(mc_model, ups.load("mc")["mass"])
print("Simulation\n" + mc_fit.summary())

tail_params = {"crystal_ball": ("alpha", "n"), "double_gauss": ("ratio", "frac")}[args.signal]
fixed = {f"{p}_1S": mc_fit.values[f"{p}_1S"] for p in tail_params}
share = {f"{p}_{peak}": f"{p}_1S" for p in tail_params for peak in ("2S", "3S")}

mass = ups.load(args.dataset)["mass"]
data_fit = ups.fit(ups.build_model(signal=args.signal, share=share), mass, fix=fixed, binned=args.binned)
print("Data\n" + data_fit.summary())
ups.plotting.plot_fit(data_fit, save=f"fit_3peaks_{args.signal}_{args.dataset}.png")

# Same data, same binning: Gaussian baseline and tails floating per peak, for comparison
fits = {
    "gauss": ups.fit(ups.build_model(), mass, binned=args.binned),
    f"{args.signal} (MC tails)": data_fit,
    f"{args.signal} (free tails)": ups.fit(ups.build_model(signal=args.signal), mass, binned=args.binned),
}
print("\nModel comparison\n" + ups.compare.table(fits))
