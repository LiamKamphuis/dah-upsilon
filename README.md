# Upsilon mass fits: DAH Project F1

Extended maximum-likelihood fits of the Υ(1S), Υ(2S), Υ(3S) peaks in LHCb 2015
dimuon data (brief: `docs/project_F1_description.md`). Analysis techniques inspired by:

The LHCb collaboration., Aaij, R., Beteta, C.A. et al. Production of J/ψ and mesons in pp collisions at $\sqrt{s} = 8$
 TeV. J. High Energ. Phys. 2013, 64 (2013). https://doi.org/10.1007/JHEP06(2013)064

## Setup

```bash
uv sync                 # creates .venv (Python 3.12) with numpy, scipy, iminuit, matplotlib, mplhep
```

## API

The API is intended to be fairly intuitive and flexible, allowing for simple model building, fitting, and viewing and comparison. 

Here is an example usage:

```python
import upsilon as ups

events = ups.load("small")                              # structured array: mass, pt, eta, p, pt1, pt2
model  = ups.build_model(("1S",), window=(9.0, 9.9))    # Gaussian + exponential
result = ups.fit(model, events["mass"])                 # extended unbinned ML (binned=True for binned)
print(result.summary())                                 # values, HESSE errors; minos=True for MINOS
ups.plotting.plot_fit(result, save="fit_1S.png")        # data, model, components, pulls
```

Model choice is a set of arguments, so alternatives are one-line changes:

```python
ups.build_model(signal="crystal_ball",
                share={"alpha_2S": "alpha_1S", "alpha_3S": "alpha_1S"})
ups.fit(model, mass, fix={"alpha_1S": 1.7, "n_1S": 2.5})   # e.g. tails fixed from MC
```

## Layout

The package holds the reusable fitting machinery; each script is one section of the report and
writes its figures to `output/figures/<dataset>/` and its numerical results to `output/results/<dataset>/`
(both gitignored).

| Path | Contents |
|---|---|
| `src/upsilon/io.py` | binary reader, the selection (`select_muons`, `MIN_MUON_PT`), rapidity, sideband subtraction, output paths |
| `src/upsilon/models.py` | shapes normalised on the fit window (Gaussian, double Gaussian, Crystal Ball, double Crystal Ball; exponential background), `MassModel` (Σ N_k·shape_k, parameter sharing, per-cell renaming), `build_model()` |
| `src/upsilon/fitting.py` | `fit()` and simultaneous `fit_cells()` → `FitResult`, with convergence checks and restarts |
| `src/upsilon/compare.py` | goodness of fit, likelihood ratio, AIC/BIC, correlations, likelihood scans |
| `src/upsilon/masses.py` | MC shape bias, mass differences, momentum-scale calibration |
| `src/upsilon/plotting.py` | observables, fit overlay, residuals/pulls, likelihood scans; mplhep LHCb style |

| Script | Brief task / report section |
|---|---|
| `scripts/01_measure_masses.py` | tasks 1–4: observables, Gaussian 1S fit, errors by hand (−2 ln L scans), bin-width scan, three-peak fit, residuals |
| `scripts/02_signal_shape.py` | task 5: shapes fitted to MC and data, model comparison, MC-tail test, correlations |
| `scripts/03_selection.py` | task 6: sideband purity, cut scan, mass stability in slices |
| `scripts/04_kinematic_bins.py` | task 7: simultaneous fit in (pT, y) cells with common masses |
| `scripts/05_momentum_scale.py` | data − MC peak position against p and softer-muon pT |
| `scripts/06_final_masses.py` | task 8: masses ± stat ± fit model ± momentum scale from the kinematic-cell fit (bias-corrected, fiducial region); 1S-calibration cross-check |
| `scripts/07_toys.py` | fit validation: pulls from pseudo-experiments |

Scripts are numbered in the order of the brief's tasks. They take `--dataset small|full|mc` (default `small` for quick runs; use `full` for the report).
