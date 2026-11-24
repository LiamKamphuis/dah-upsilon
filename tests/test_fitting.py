import numpy as np
import pytest

import upsilon as ups

WINDOW = (9.0, 9.9)
TRUE = {"N_1S": 5000, "mu_1S": 9.46, "sigma_1S": 0.045, "N_bkg": 15000, "c_bkg": 1.5}


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(1)
    sig = rng.normal(TRUE["mu_1S"], TRUE["sigma_1S"], TRUE["N_1S"])
    # truncated exponential via inverse cdf
    u = rng.uniform(size=TRUE["N_bkg"])
    w = WINDOW[1] - WINDOW[0]
    bkg = WINDOW[0] - np.log1p(u * np.expm1(-TRUE["c_bkg"] * w)) / TRUE["c_bkg"]
    return np.concatenate([sig, bkg])


@pytest.mark.parametrize("binned", [False, True])
def test_fit_recovers_truth(sample, binned):
    res = ups.fit(ups.build_model(("1S",), window=WINDOW), sample, binned=binned)
    assert res.valid
    for name, true in TRUE.items():
        assert abs(res.values[name] - true) < 4 * res.errors[name], name


def test_fix(sample):
    res = ups.fit(ups.build_model(("1S",), window=WINDOW), sample, fix={"sigma_1S": 0.045})
    assert res.values["sigma_1S"] == 0.045 and res.minuit.fixed["sigma_1S"]


@pytest.mark.parametrize("binned", [False, True])
def test_limits_applied(sample, binned):
    model = ups.build_model(("1S",), window=WINDOW)
    res = ups.fit(model, sample, binned=binned, progress=False)
    assert tuple(res.minuit.limits["sigma_1S"]) == model.params["sigma_1S"].limits
    assert tuple(res.minuit.limits["N_1S"]) == (0.0, np.inf)


def test_signal_only_fit_recovers_truth():
    rng = np.random.default_rng(3)
    x = rng.normal(TRUE["mu_1S"], TRUE["sigma_1S"], TRUE["N_1S"])
    res = ups.fit(ups.build_model(("1S",), window=WINDOW, background=None), x, progress=False)
    assert res.valid
    for name in ("N_1S", "mu_1S", "sigma_1S"):
        assert abs(res.values[name] - TRUE[name]) < 4 * res.errors[name], name


def test_warns_on_unconstrained_parameter():
    """Regression: full-data CB fit with free 2S tails gave yield errors of 2 while
    reporting valid. Same mechanism here: Gaussian data leave the CB tail unconstrained."""
    x = np.random.default_rng(4).normal(TRUE["mu_1S"], TRUE["sigma_1S"], 20000)
    model = ups.build_model(("1S",), signal="crystal_ball", window=WINDOW, background=None)
    with pytest.warns(UserWarning, match="n_1S"):
        ups.fit(model, x, progress=False)


def test_no_warning_on_good_fit(sample):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ups.fit(ups.build_model(("1S",), window=WINDOW), sample, progress=False)


def test_fit_cells_shares_mass_and_separates_widths():
    """Two cells with different resolution and one true mass: mu is shared, sigma is per cell."""
    rng = np.random.default_rng(7)
    model = ups.build_model(("1S",), window=WINDOW)
    cells = {"a": 0.030, "b": 0.060}
    models = [model.per_cell(c, common={"mu_1S"}) for c in cells]
    assert "mu_1S" in models[0].names and {"sigma_1S_a", "N_bkg_a", "c_bkg_a"} <= set(models[0].names)
    masses = [np.concatenate([rng.normal(9.46, s, 20000), WINDOW[0] + rng.exponential(1 / 1.5, 40000)])
              for s in cells.values()]
    results = ups.fit_cells(models, masses, start={"sigma_1S_a": 0.04, "sigma_1S_b": 0.04}, progress=False)
    v, e = results[0].values, results[0].errors
    assert results[0].valid and results[0].minuit is results[1].minuit
    assert abs(v["mu_1S"] - 9.46) < 3 * e["mu_1S"]
    for c, s in cells.items():
        assert abs(v[f"sigma_1S_{c}"] - s) < 3 * e[f"sigma_1S_{c}"]
