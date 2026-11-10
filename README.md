# Upsilon mass fits: DAH Project F1

Extended maximum-likelihood fits of the Υ(1S), Υ(2S), Υ(3S) peaks in LHCb 2015
dimuon data (brief: `docs/project_F1_description.md`, paper: `docs/JHEP06(2013)064.pdf`).

## Setup

```bash
uv sync                 # creates .venv (Python 3.12) with numpy, scipy, iminuit, matplotlib, pytest
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
ups.fit(model, mass, fix={"alpha_1S": 1.7, "n_1S": 2.5})   # tails fixed from MC
```

## Layout

| Path | Contents |
|---|---|
| `src/upsilon/io.py` | binary reader, `select()` cuts, column labels, project paths |
| `src/upsilon/models.py` | `Shape`s normalised on the fit window (`Gaussian`, `Exponential`; `DoubleGaussian`, `CrystalBall` **TODO**), `MassModel` (Σ N_k·shape_k, parameter sharing), `build_model()` |
| `src/upsilon/fitting.py` | `fit()` → `FitResult` (values, errors, summary, JSON) via iminuit |
| `src/upsilon/plotting.py` | observable histograms, fit overlay, residuals/pulls, 2D vs mass |
| `scripts/measure_masses.py` | the core measurement: observables, 1S fit, three-peak fit, residuals/pulls |
| `scripts/signal_shape.py` | Crystal Ball / double Gaussian, tails fixed from simulation (needs the TODO shapes) |
| `scripts/kinematics.py` | observables vs mass; TODO purity and (pt, η) bins |
| `scripts/systematics.py` | refit with one choice changed; table of mass shifts |
| `output/figures`, `output/results` | generated plots and fit JSON (gitignored) |

Scripts take `--dataset small|full|mc` (default `small`); `measure_masses.py --binned` is much faster.

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
  Judge shifts against the statistical spread of the *difference* (Barlow). Pseudo-experiments
  (generate from a fitted model, refit) can separate a real bias from noise.
* **Residuals/pulls** are the first diagnostic of model inadequacy: a structured pattern
  around the peaks (e.g. low-mass excess from FSR) motivates the Crystal Ball.
