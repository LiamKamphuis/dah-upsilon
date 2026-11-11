import numpy as np
import pytest

from upsilon.models import CrystalBall, DoubleGaussian, Exponential, Gaussian, build_model

WINDOW = (9.0, 9.9)
X = np.linspace(*WINDOW, 20001)


@pytest.mark.parametrize("shape, par", [
    (Gaussian(WINDOW), (9.46, 0.04)),
    (Gaussian(WINDOW), (9.05, 0.3)),  # heavily truncated
    (DoubleGaussian(WINDOW), (9.46, 0.04, 2.0, 0.7)),
    (DoubleGaussian(WINDOW), (9.05, 0.1, 5.0, 0.3)),  # wide component heavily truncated
    (CrystalBall(WINDOW), (9.46, 0.04, 1.5, 3.0)),
    (CrystalBall(WINDOW), (9.2, 0.05, 0.5, 1.5)),  # long tail cut off at lo
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
