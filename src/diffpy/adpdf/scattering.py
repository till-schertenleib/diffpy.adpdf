"""Scattering factors and partial weights for the resonant and non-resonant
channels."""

from typing import Any, Dict, List

import numpy as np
import xraydb

from diffpy.adpdf.utils import as_1d, safe


def f0_of_Q(element: str, Q: np.ndarray) -> np.ndarray:
    s = as_1d(Q) / (4.0 * np.pi)
    f0 = np.array([xraydb.f0(element, si)[0] for si in s], dtype=float)
    return safe(f0, f"f0({element})")


def compute_scattering(
    elements: List[str],
    comp: Dict[str, float],
    alpha: str,
    Q: np.ndarray,
    E_eV: np.ndarray,
    edge_model: Any,
    fp_offset: float = 0.0,
):
    Q = as_1d(Q)
    E = as_1d(E_eV)
    nq, ne = len(Q), len(E)
    f0_map = {el: f0_of_Q(el, Q) for el in elements}

    fp = edge_model.fp(E, return_f1=False)
    fpp = np.maximum(edge_model.fpp(E), 0.0)

    fr_weights, f2_weights = {}, {}
    for el in elements:
        f0 = f0_map[el]
        if el == alpha:
            fr = (fp + fp_offset)[:, None] + f0[None, :]
            fi = fpp[:, None]
        else:
            fr = f0[None, :]
            fi = np.zeros((ne, 1))
        fr_weights[el] = fr
        f2_weights[el] = fr**2 + fi**2

    fmean_real = np.zeros((ne, nq), float)
    fself = np.zeros((ne, nq), float)
    for el in elements:
        c = float(comp[el])
        fmean_real += c * fr_weights[el]
        fself += c * f2_weights[el]

    fi_alpha = fpp[:, None]
    fmean_imag = float(comp[alpha]) * fi_alpha * np.ones((1, nq))
    w_tot = fmean_real**2 + fmean_imag**2

    return fr_weights, fi_alpha, fmean_real, fmean_imag, w_tot, fself


def weights_alpha_not(
    comp: Dict[str, float],
    alpha: str,
    fr_weights: Dict[str, np.ndarray],
    fi_alpha: np.ndarray,
    fmean_real: np.ndarray,
    fmean_imag: np.ndarray,
    w_tot: np.ndarray,
):
    cA = float(comp[alpha])
    w_A = cA * (fr_weights[alpha] * fmean_real + fi_alpha * fmean_imag)
    eps = 1e-12
    a = w_A / (w_tot + eps)
    b = 1.0 - a

    # TODO(physics): heuristic clip of the channel weights
    a = np.clip(a, -5.0, 5.0)
    b = np.clip(b, -5.0, 5.0)
    return a, b, {"a_min": float(np.nanmin(a)), "a_max": float(np.nanmax(a))}


def alpha_beta_from_fpp(
    fpp_E: np.ndarray,
    alpha_C: float,
    alpha_D: float,
    beta_C: float,
    beta_D: float,
):
    return (
        np.clip(alpha_C + alpha_D * fpp_E, 1e-6, None),
        beta_C + beta_D * fpp_E,
    )
