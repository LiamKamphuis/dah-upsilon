"""Tests for masses.py.

The error checks use toys: draw many mass vectors from N(mu, cov), push each through
the formula, and compare the spread with the propagated error.
"""
import numpy as np
import pytest

from upsilon import masses
from upsilon.masses import Bias

MU = np.array([9.4570, 10.0190, 10.3505])
SIG = np.array([0.25, 0.70, 1.20]) * 1e-3
CORR = np.array([[1.0, 0.4, 0.3], [0.4, 1.0, 0.5], [0.3, 0.5, 1.0]])  # strong, so ignoring it fails
COV = CORR * np.outer(SIG, SIG)
PDG = np.array([9.4603, 10.02326, 10.3552])


@pytest.fixture(scope="module")
def toys():
    return np.random.default_rng(0).multivariate_normal(MU, COV, size=200_000)


def test_apply_bias():
    b = Bias(value=1e-3, error=0.1e-3, sigma=0.040)
    sigma = np.array([0.040, 0.044, 0.046])
    mu, cov = masses.apply_bias(MU, COV, b, sigma)
    k = sigma / 0.040
    assert mu == pytest.approx(MU - 1e-3 * k)
    assert cov[0, 2] - COV[0, 2] == pytest.approx(k[0] * k[2] * 1e-8)


def test_mass_difference(toys):
    value, error = masses.mass_difference(MU, COV, 1, 0)
    assert value == pytest.approx(MU[1] - MU[0])
    assert error == pytest.approx(np.std(toys[:, 1] - toys[:, 0]), rel=0.01)


def test_scale_corrected_masses(toys):
    mu, cov = masses.scale_corrected_masses(MU, COV, ref=0, m_ref=PDG[0])
    assert mu == pytest.approx(MU * PDG[0] / MU[0])
    corrected = toys * (PDG[0] / toys[:, [0]])
    assert np.sqrt(np.diag(cov))[1:] == pytest.approx(corrected.std(axis=0)[1:], rel=0.01)
    assert cov[1, 2] == pytest.approx(np.cov(corrected[:, 1], corrected[:, 2])[0, 1], rel=0.02)


def test_common_scale_exact():
    s = 0.99965
    fit = masses.common_scale(s * PDG, COV, PDG)
    assert fit.s == pytest.approx(s, abs=1e-9)
    assert fit.chi2 == pytest.approx(0.0, abs=1e-9)
    assert fit.ndof == 2 and fit.pvalue == pytest.approx(1.0)


def test_common_scale_error(toys):
    fits = [masses.common_scale(t, COV, PDG) for t in toys[:5000]]
    assert fits[0].error == pytest.approx(np.std([f.s for f in fits]), rel=0.03)
    # chi2 of toys drawn from a pure scale model (shifted to s*PDG) follows chi2(2): mean 2
    shifted = toys[:5000] - MU + 0.9997 * PDG
    assert np.mean([masses.common_scale(t, COV, PDG).chi2 for t in shifted]) == pytest.approx(2.0, rel=0.05)
