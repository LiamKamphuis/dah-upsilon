# Upsilon mass fits: DAH Project F1

Extended maximum-likelihood fits of the Υ(1S), Υ(2S), Υ(3S) peaks in LHCb 2015
dimuon data (brief: `docs/project_F1_description.md`, paper: `docs/JHEP06(2013)064.pdf`).

## Setup

```bash
uv sync                 # creates .venv (Python 3.12) with numpy, scipy, iminuit, matplotlib, mplhep, pytest
uv run pytest           # model normalisation + fit-recovery tests
```

In PyCharm: *Settings → Project → Python Interpreter → Add → Existing → `.venv/bin/python`*.
Put the data files in `data/` (`ups-15.bin`, `ups-15-small.bin`, `mc.bin`; gitignored).

## API

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
writes its figures to `output/figures/<dataset>/` and its numbers to `output/results/<dataset>/`
(both gitignored). Every number in the report comes from one of these scripts.

| Path | Contents |
|---|---|
| `src/upsilon/io.py` | binary reader, the selection (`select_muons`, `MIN_MUON_PT`), rapidity, sideband subtraction, output paths |
| `src/upsilon/models.py` | shapes normalised on the fit window (Gaussian, double Gaussian, Crystal Ball, double Crystal Ball; exponential and quadratic background), `MassModel` (Σ N_k·shape_k, parameter sharing, per-cell renaming), `build_model()` |
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

## Design notes for the discussion

* **Normalisation on the window.** Each shape is normalised on `[lo, hi]` analytically via its CDF,
  so the composite stays normalised for any parameters and fit range. The fit range then becomes
  a clean systematic variation.
* **Fit window.** The data files cover 9.0–11.0 GeV/c² (not 8.5 as the brief states). A window that
  extends past the data makes the normalised background predict events where there are none and
  drives the fit to an unphysical minimum, so the default window is (9.0, 11.0).
* **Model choice = configuration.** Signal/background shapes are chosen by name and parameters
  can be shared across peaks (`share=`). Comparing "independent widths" vs "common tail parameters"
  vs "tails fixed from MC" therefore uses the same code path, which keeps the comparisons fair.
* **Systematics as variations.** Every analysis choice (signal model, background model, window,
  binning, momentum scale) is a refit with one change, and the parameter shift is the systematic.
  A cut shares most events with the baseline, and removes mostly background, so neither fit's
  error (nor Barlow's formula, which assumes a random subset) gives the error on the shift; the
  mass is instead checked for stability in independent slices (`03_selection.py`). Pseudo-experiments
  (`07_toys.py`) separate a fit bias from noise.
* **Residuals/pulls** are the first diagnostic of model inadequacy: a structured pattern
  around the peaks (e.g. low-mass excess from FSR) motivates the Crystal Ball.
