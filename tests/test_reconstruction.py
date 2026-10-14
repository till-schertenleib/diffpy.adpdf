"""Smoke tests for S(Q) construction and the Fourier transform to G(r)."""

import numpy as np
import pytest

from diffpy.adpdf.reconstruction import S_to_G, build_S_from_raw


def test_S_to_G_zero_for_flat_S():
    Q = np.linspace(0.5, 20.0, 500)
    r = np.linspace(0.0, 10.0, 101)
    G = S_to_G(Q, np.ones_like(Q), r, window="none", kbeta=6.0)
    assert np.allclose(G, 0.0)


@pytest.mark.parametrize("window", ["none", "lorch", "kaiser"])
def test_S_to_G_peak_position(window):
    # F(Q) = Q (S - 1) = sin(Q r0) transforms to a peak at r = r0
    r0 = 2.5
    Q = np.linspace(1e-3, 40.0, 4000)
    S = 1.0 + np.sin(Q * r0) / Q
    r = np.arange(0.5, 6.0, 0.01)
    G = S_to_G(Q, S, r, window=window, kbeta=6.0)
    assert r[np.argmax(G)] == pytest.approx(r0, abs=0.02)


@pytest.mark.parametrize("bkg_poly_type", ["wls", "getx"])
def test_build_S_from_raw_shapes(bkg_poly_type):
    ne, nq = 4, 300
    Q = np.linspace(0.5, 20.0, nq)
    E = np.linspace(29000.0, 29300.0, ne)
    rng = np.random.default_rng(0)
    I_raw = 1.0 + 0.1 * rng.standard_normal((ne, nq))
    w_tot = np.full((ne, nq), 2.0)
    fself = np.full((ne, nq), 1.5)
    S, beta_o = build_S_from_raw(
        I_raw,
        Q,
        E,
        np.ones(ne),
        np.zeros(ne),
        w_tot,
        fself,
        rpoly=1.0,
        bkg_poly_type=bkg_poly_type,
    )
    assert S.shape == (ne, nq)
    assert beta_o.shape == (ne, nq)
    assert np.all(np.isfinite(S))


def test_build_S_from_raw_rejects_unknown_type():
    Q = np.linspace(0.5, 20.0, 50)
    with pytest.raises(ValueError, match="Unknown bkg_poly_type"):
        build_S_from_raw(
            np.ones((2, 50)),
            Q,
            np.array([1.0, 2.0]),
            np.ones(2),
            np.zeros(2),
            np.ones((2, 50)),
            np.ones((2, 50)),
            rpoly=1.0,
            bkg_poly_type="nope",
        )
