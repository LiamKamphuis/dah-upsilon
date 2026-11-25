"""Toy (pseudo-experiment) study: is the baseline fit unbiased, and are its errors right?

Fit the selected data with the baseline model, then treat that fit as the truth:
draw Poisson counts in every bin from its expected counts, refit, and repeat. For each
parameter the pull (fit - true) / error should be a unit Gaussian:

- mean != 0: the fit is biased, by mean * error (a candidate systematic);
- width != 1: the HESSE errors are too large (< 1) or too small (> 1).

Invalid toy fits are counted: a fit that only sometimes converges would show up here.
Binned toys match the binned baseline fit, and each one takes a few seconds.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import upsilon as ups
from upsilon.io import output_subdir
from upsilon.models import BASELINE_SIGNAL, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--toys", type=int, default=300)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

PARAMS = ("mu_1S", "mu_2S", "mu_3S", "sigma_1S")
model = ups.build_model(signal=BASELINE_SIGNAL, share=share_tails(BASELINE_SIGNAL), tie_widths=True)
truth = ups.fit(model, ups.select_muons(ups.load(args.dataset))["mass"], binned=True, progress=False)
edges = truth.edges
expected = sum(truth.model.component_counts(edges, truth.values).values())
centres = 0.5 * (edges[1:] + edges[:-1])
print(f"truth fit valid={truth.valid}, {expected.sum():.0f} expected events in {len(expected)} bins")

rng = np.random.default_rng(args.seed)
rows = []
for i in range(args.toys):
    counts = rng.poisson(expected)
    # One entry per event at its bin centre: the binned fit (same edges) re-histograms it exactly
    r = ups.fit(model, np.repeat(centres, counts), binned=True, start=truth.values, progress=False)
    rows.append({"valid": r.valid, **{p: {"value": r.values[p], "error": r.errors[p]} for p in PARAMS}})
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{args.toys} toys, {sum(not row['valid'] for row in rows)} invalid so far", flush=True)

ok = [row for row in rows if row["valid"]]
summary = {"toys": args.toys, "invalid": args.toys - len(ok), "params": {}}
fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # 2 x 2 reads better on a page than 1 x 4
grid = np.linspace(-4, 4, 200)
print(f"\n{len(ok)}/{args.toys} valid toy fits\n{'parameter':<10}{'pull mean':>16}{'pull width':>16}"
      f"{'bias [MeV]':>14}{'<error> [MeV]':>16}{'KS p':>10}")
for ax, p in zip(axes.flat, PARAMS):
    value = np.array([row[p]["value"] for row in ok])
    error = np.array([row[p]["error"] for row in ok])
    pull = (value - truth.values[p]) / error
    n = len(pull)
    mean, width = pull.mean(), pull.std(ddof=1)
    mean_err, width_err = width / np.sqrt(n), width / np.sqrt(2 * (n - 1))
    bias = mean * error.mean()
    ks = stats.kstest(pull, "norm").pvalue  # shape test against a unit Gaussian
    summary["params"][p] = {"pull_mean": mean, "pull_mean_err": mean_err, "pull_width": width,
                            "pull_width_err": width_err, "bias": bias, "mean_error": error.mean(), "ks_pvalue": ks}
    print(f"{p:<10}{mean:>+9.3f} +- {mean_err:.3f}{width:>9.3f} +- {width_err:.3f}"
          f"{1e3 * bias:>+14.4f}{1e3 * error.mean():>16.4f}{ks:>10.2f}")
    # 17 bins of width 0.5 with one centred on 0 (edges at 0 make a fluctuation look like a spike)
    ax.hist(pull, bins=17, range=(-4.25, 4.25), histtype="step", color="k", lw=2, label=f"{n} toys")
    ax.plot(grid, n * 0.5 * stats.norm.pdf(grid, mean, width), "C3",
            label=rf"mean ${mean:+.2f}\pm{mean_err:.2f}$" + "\n" + rf"width ${width:.2f}\pm{width_err:.2f}$")
    ax.plot(grid, n * 0.5 * stats.norm.pdf(grid), "0.5", ls="--", label=f"unit Gaussian (KS p = {ks:.2f})")
    symbol = rf"$\{p.split('_')[0]}_{{{p.split('_')[1]}}}$"  # mu_1S -> $\mu_{1S}$
    ax.set(xlabel=f"Pull of {symbol}", ylabel="Toys / bin")
    ax.set_ylim(0, 1.6 * ax.get_ylim()[1])  # headroom so the legend sits above the histogram
    ax.legend(loc="upper left", fontsize=16)
fig.tight_layout(w_pad=3)

ups.save_json({"truth": truth.values, "summary": summary, "toys": rows}, "toys.json", args.dataset)
ups.plotting.save_figure(fig, f"{output_subdir(args.dataset)}/toy_pulls.png")
