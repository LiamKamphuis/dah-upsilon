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
