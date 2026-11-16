import numpy as np
import pytest

from upsilon.models import (CrystalBall, DoubleCrystalBall, DoubleGaussian, Exponential, Gaussian, Hypatia,
                            JohnsonSU, build_model)

todo = pytest.mark.xfail(raises=NotImplementedError, reason="TODO: scaffolded shape")

WINDOW = (9.0, 9.9)
X = np.linspace(*WINDOW, 20001)


@pytest.mark.parametrize("shape, par", [
    (Gaussian(WINDOW), (9.46, 0.04)),
    (Gaussian(WINDOW), (9.05, 0.3)),  # heavily truncated
    (DoubleGaussian(WINDOW), (9.46, 0.04, 2.0, 0.7)),
    (DoubleGaussian(WINDOW), (9.05, 0.1, 5.0, 0.3)),  # wide component heavily truncated
    (CrystalBall(WINDOW), (9.46, 0.04, 1.5, 3.0)),
    (CrystalBall(WINDOW), (9.2, 0.05, 0.5, 1.5)),  # long tail cut off at lo
    (DoubleCrystalBall(WINDOW), (9.46, 0.033, 1.5, 3.3, 1.6, 0.43)),
    (DoubleCrystalBall(WINDOW), (9.2, 0.05, 0.5, 1.5, 3.0, 0.8)),  # wide tail cut off at lo
    pytest.param(JohnsonSU(WINDOW), (9.46, 0.04, -0.5, 2.0), marks=todo),
    pytest.param(Hypatia(WINDOW), (9.46, 0.04, -2.0, 1e-3, 0.0, 2.0, 2.0, 3.0, 5.0), marks=todo),
    (Exponential(WINDOW), (2.0,)),
    (Exponential(WINDOW), (0.0,)),
    (Exponential(WINDOW), (-1.0,)),
])
def test_shape_normalised_and_cdf_consistent(shape, par):
    assert np.trapezoid(shape.pdf(X, *par), X) == pytest.approx(1.0, rel=1e-6)
    assert shape.cdf(np.array(WINDOW), *par) == pytest.approx([0.0, 1.0], abs=1e-9)
    mid = 9.5
    assert shape.cdf(np.array([mid]), *par)[0] == pytest.approx(
        np.trapezoid(shape.pdf(X[X <= mid], *par), X[X <= mid]), rel=1e-4)


def test_composite_integrates_to_total_yield():
    model = build_model(("1S",), window=WINDOW)
    par = {"N_1S": 300.0, "mu_1S": 9.46, "sigma_1S": 0.04, "N_bkg": 700.0, "c_bkg": 1.0}
    total, dens = model(X, *[par[n] for n in model.names])
    assert total == 1000.0
    assert np.trapezoid(dens, X) == pytest.approx(1000.0, rel=1e-6)


def test_share_merges_parameters():
    model = build_model(share={"sigma_2S": "sigma_1S", "sigma_3S": "sigma_1S"})
    assert "sigma_2S" not in model.names and "sigma_1S" in model.names
    with pytest.raises(ValueError):
        build_model(share={"nonexistent_1S": "sigma_1S"})


def test_signal_only_model():
    """background=None: for simulation, which has no combinatorial background."""
    model = build_model(("1S",), window=WINDOW, background=None)
    assert list(model.components) == ["1S"]
    assert "N_bkg" not in model.names


def test_share_tails():
    from upsilon.models import share_tails
    assert share_tails("gauss") == {}
    assert share_tails("crystal_ball") == {"alpha_2S": "alpha_1S", "alpha_3S": "alpha_1S",
                                           "n_2S": "n_1S", "n_3S": "n_1S"}
    model = build_model(signal="crystal_ball", share=share_tails("crystal_ball"))
    assert "alpha_2S" not in model.names and "alpha_1S" in model.names


def test_double_cb_reduces_to_cb():
    x = np.linspace(*WINDOW, 101)
    for frac, ratio in ((1.0, 2.0), (0.3, 1.0)):
        assert DoubleCrystalBall(WINDOW).pdf(x, 9.46, 0.04, 1.5, 3.0, ratio, frac) == pytest.approx(
            CrystalBall(WINDOW).pdf(x, 9.46, 0.04, 1.5, 3.0))


def test_double_cb_shared_tails():
    from upsilon.models import share_tails
    model = build_model(signal="double_cb", share=share_tails("double_cb"))
    assert {"ratio_2S", "frac_3S", "alpha_2S"}.isdisjoint(model.names)
    assert len(model.names) == 3 * 3 + 4 + 2  # (N, mu, sigma) per peak + shared shape + background
