"""Why the unbinned baseline three-peak fit on data ends INVALID, while the binned fit converges.

Four checks, each ruling a cause in or out:

1. Correlations of the (valid) binned fit: a near-zero eigenvalue of the correlation
   matrix is a near-flat direction of the likelihood, which MIGRAD struggles to converge along.
2. Smoothness of the unbinned -2 ln L: |FCN| ~ 1e7 on the small dataset, so rounding could
   spoil finite-difference derivatives. If d2F/dp2 is stable from 1e-3 to 1 sigma steps, it is not.
3. HESSE alone, at the binned minimum: if it succeeds there, HESSE itself is not what fails.
4. (``--migrad``, slow: ~20 min on small) MIGRAD started at the binned minimum: does it reach
   the EDM goal, and how far do the masses move while it tries?
"""
import argparse
import json
import time

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

import upsilon as ups
from upsilon.io import RESULT_DIR
from upsilon.models import BASELINE_SIGNAL, share_tails

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="small")
parser.add_argument("--bins", type=int, default=250)
parser.add_argument("--migrad", action="store_true", help="also run check 4 (slow)")
args = parser.parse_args()

MASSES = ("mu_1S", "mu_2S", "mu_3S")
SHAPE = ("sigma_1S", "frac_1S", "ratio_1S", "alpha_1S", "n_1S")
MEV = 1e3

model = ups.build_model(signal=BASELINE_SIGNAL, share=share_tails(BASELINE_SIGNAL))
data = ups.load(args.dataset)["mass"]
lo, hi = model.window
x = data[(data >= lo) & (data <= hi)]
results = {"dataset": args.dataset, "bins": args.bins, "events": len(x)}


def fmin_flags(m: Minuit) -> dict:
    f = m.fmin
    return {"valid": f.is_valid, "edm": f.edm, "edm_goal": f.edm_goal, "nfcn": f.nfcn,
            "hesse_failed": f.hesse_failed, "accurate": f.has_accurate_covar,
            "made_posdef": f.has_made_posdef_covar, "at_limit": f.has_parameters_at_limit}


def unbinned_minuit(values, errors) -> Minuit:
    m = Minuit(ExtendedUnbinnedNLL(x, model), *values, name=names)
    for n, p in model.params.items():
        lo_lim, hi_lim = p.limits
        m.limits[n] = (-np.inf if lo_lim is None else lo_lim, np.inf if hi_lim is None else hi_lim)
    m.errors = errors
    return m


# 1. binned reference fit and its correlation structure
binned = ups.fit(model, data, binned=True, bins=args.bins, progress=False).minuit
names = list(binned.parameters)
vals, errs = np.array(binned.values), np.array(binned.errors)
cov = np.array(binned.covariance)
corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
eigval, eigvec = np.linalg.eigh(corr)
print(f"1. binned fit ({args.bins} bins): {fmin_flags(binned)}")
results["binned"] = {"fmin": fmin_flags(binned), "values": dict(zip(names, vals)), "errors": dict(zip(names, errs))}
results["flat_directions"] = []
for k in range(2):
    top = np.argsort(-np.abs(eigvec[:, k]))[:4]
    loadings = {names[i]: eigvec[i, k] for i in top}
    results["flat_directions"].append({"eigenvalue": eigval[k], "loadings": loadings})
    print(f"   correlation-matrix eigenvalue {eigval[k]:.4f} along "
          + ", ".join(f"{n} {v:+.2f}" for n, v in loadings.items()))
strong = {f"{names[i]}~{names[j]}": float(corr[i, j])
          for i in range(len(names)) for j in range(i) if abs(corr[i, j]) > 0.8}
results["strong_correlations"] = strong
print("   |corr| > 0.8:", {k: round(v, 3) for k, v in strong.items()})
mass_corr = {mu: {s: float(corr[names.index(mu), names.index(s)]) for s in SHAPE} for mu in MASSES}
results["mass_shape_correlations"] = mass_corr
print("   mu_1S correlation with shape parameters:", {s: round(v, 3) for s, v in mass_corr["mu_1S"].items()})

# 2. smoothness of the unbinned likelihood at the binned minimum
cost = ExtendedUnbinnedNLL(x, model)
t = time.time()
f0 = cost(*vals)
results["seconds_per_call"] = time.time() - t
results["fcn"] = f0
print(f"\n2. unbinned: {len(x)} events, {results['seconds_per_call']:.2f} s/call, FCN = {f0:.1f}")
print("   d2F/dp2 * sigma_binned^2 for steps h = (1e-3, 1e-2, 0.1, 1) sigma (stable = smooth FCN):")
steps = (1e-3, 1e-2, 1e-1, 1.0)
results["curvature"] = {}
for n in ("mu_1S",) + SHAPE:
    i = names.index(n)
    row = []
    for frac in steps:
        h = frac * errs[i]
        up, down = vals.copy(), vals.copy()
        up[i] += h
        down[i] -= h
        row.append((cost(*up) + cost(*down) - 2 * f0) / h**2 * errs[i] ** 2)
    results["curvature"][n] = dict(zip(map(str, steps), row))
    print(f"   {n:<9}" + "".join(f"{v:>10.3f}" for v in row))

# 3. HESSE alone at the binned minimum
m = unbinned_minuit(vals, errs)
t = time.time()
m.hesse()
results["hesse_at_binned_minimum"] = {
    "fmin": fmin_flags(m), "seconds": time.time() - t,
    "error_ratio_unbinned_over_binned": dict(zip(names, np.array(m.errors) / errs))}
print(f"\n3. HESSE at the binned minimum: {fmin_flags(m)}")
print("   mass errors unbinned/binned:",
      {mu: round(float(m.errors[mu] / binned.errors[mu]), 3) for mu in MASSES})

# 4. MIGRAD from the binned minimum
if args.migrad:
    m = unbinned_minuit(vals, errs)
    t = time.time()
    m.migrad()
    m.hesse()
    out = {"fmin": fmin_flags(m), "seconds": time.time() - t,
           "mass_shift_mev": {mu: MEV * (m.values[mu] - binned.values[mu]) for mu in MASSES},
           "mass_errors_mev": {mu: MEV * m.errors[mu] for mu in MASSES},
           "shape_values": {s: [m.values[s], m.errors[s]] for s in SHAPE}}
    results["migrad_from_binned_minimum"] = out
    print(f"\n4. MIGRAD + HESSE from the binned minimum ({out['seconds']:.0f} s): {out['fmin']}")
    for mu in MASSES:
        print(f"   {mu}: unbinned - binned = {out['mass_shift_mev'][mu]:+.3f} MeV, "
              f"error {out['mass_errors_mev'][mu]:.3f} (binned {MEV * binned.errors[mu]:.3f}) MeV")
    for s in SHAPE:
        print(f"   {s}: {m.values[s]:.4g} +- {m.errors[s]:.3g}  (binned {binned.values[s]:.4g} +- {binned.errors[s]:.3g})")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
path = RESULT_DIR / f"unbinned_diagnostics_{args.dataset}.json"
path.write_text(json.dumps(results, indent=2, default=float))
print(f"\nsaved {path}")
