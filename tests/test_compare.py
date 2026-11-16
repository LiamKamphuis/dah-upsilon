import numpy as np
import pytest

import upsilon as ups
from upsilon import compare

WINDOW = (9.0, 9.9)
TRUE = {"N_1S": 5000, "mu_1S": 9.46, "sigma_1S": 0.045, "N_bkg": 15000, "c_bkg": 1.5}


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(2)
    sig = rng.normal(TRUE["mu_1S"], TRUE["sigma_1S"], TRUE["N_1S"])
    u = rng.uniform(size=TRUE["N_bkg"])
    w = WINDOW[1] - WINDOW[0]
    bkg = WINDOW[0] - np.log1p(u * np.expm1(-TRUE["c_bkg"] * w)) / TRUE["c_bkg"]
    return np.concatenate([sig, bkg])


@pytest.fixture(scope="module")
def fits(sample):
    """Correct model (signal + background) and wrong model (background only), binned and unbinned."""
    out = {}
    for binned in (False, True):
        for name, peaks in (("right", ("1S",)), ("wrong", ())):
            out[name, binned] = ups.fit(ups.build_model(peaks, window=WINDOW), sample,
                                        binned=binned, bins=90, progress=False)
    return out


def test_gof_matches_binned_fval(fits):
    """iminuit's ExtendedBinnedNLL is the Baker-Cousins deviance, so the two must agree."""
    res = fits["right", True]
    g = compare.gof(res)
    assert g.stat == pytest.approx(res.nll, rel=1e-6)
    assert g.ndof == 90 - 5


@pytest.mark.parametrize("binned", [False, True])
def test_gof_separates_right_and_wrong(fits, binned):
    assert compare.gof(fits["right", binned], bins=90).pvalue > 1e-3
    assert compare.gof(fits["wrong", binned], bins=90).pvalue < 1e-6


@pytest.mark.parametrize("binned", [False, True])
def test_runs_separates_right_and_wrong(fits, binned):
    assert compare.runs_test(fits["right", binned], bins=90).pvalue > 1e-3
    assert compare.runs_test(fits["wrong", binned], bins=90).pvalue < 1e-6


def test_runs_counts():
    # signs + + - - - + : 3 runs, n+ = 3, n- = 3
    r = compare._runs(np.array([1.0, 2.0, -1.0, -0.5, -2.0, 3.0]))
    assert (r.runs, r.n_pos, r.n_neg) == (3, 3, 3)
    assert r.expected == pytest.approx(2 * 3 * 3 / 6 + 1)


def test_information_criteria(fits, sample):
    res = fits["right", False]
    ic = compare.information_criteria(res)
    k = 5
    assert ic.k == k
    assert ic.aic == pytest.approx(res.nll + 2 * k)
    assert ic.bic == pytest.approx(res.nll + k * np.log(len(res.data)))


def test_information_criteria_counts_only_free_parameters(sample):
    res = ups.fit(ups.build_model(("1S",), window=WINDOW), sample, fix={"sigma_1S": 0.045}, progress=False)
    assert compare.information_criteria(res).k == 4


@pytest.mark.parametrize("binned", [False, True])
def test_likelihood_ratio(fits, binned):
    lr = compare.likelihood_ratio(fits["wrong", binned], fits["right", binned])
    assert lr.ndof == 3
    assert lr.stat > 100 and lr.pvalue < 1e-10


def test_likelihood_ratio_rejects_mismatched_fits(fits):
    with pytest.raises(ValueError):
        compare.likelihood_ratio(fits["wrong", False], fits["right", True])
    with pytest.raises(ValueError):  # simple model must have fewer free parameters
        compare.likelihood_ratio(fits["right", False], fits["wrong", False])


def test_table(fits):
    text = compare.table({"gauss+exp": fits["right", True], "exp only": fits["wrong", True]})
    assert "gauss+exp" in text and "exp only" in text and "dBIC" in text


def test_profile_scan(fits):
    """Profile of mu: minimum at the fit value, crossing Delta(-2lnL) = 1 at about +-1 HESSE error."""
    res = fits["right", False]
    x, d = compare.profile_scan(res, "mu_1S", size=41, bound=2)
    assert d.min() == pytest.approx(0.0, abs=0.01)
    assert x[np.argmin(d)] == pytest.approx(res.values["mu_1S"], abs=0.1 * res.errors["mu_1S"])
    i = np.argmin(d)
    lo = np.interp(1.0, d[:i + 1][::-1], x[:i + 1][::-1])
    hi = np.interp(1.0, d[i:], x[i:])
    assert (hi - lo) / 2 == pytest.approx(res.errors["mu_1S"], rel=0.05)
