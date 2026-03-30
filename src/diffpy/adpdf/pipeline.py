import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import h5py
import kkcalc
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import xraydb

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================


@dataclass
class RefinementConfig:
    # S(Q) construction
    rpoly: float = 1.0
    hiQ_frac: float = 0.85
    ridge: float = 8e-4

    # G(r)
    window: str = "lorch"
    kaiser_beta: float = 6.0
    rmin: float = 0.0
    rmax: float = 20.0
    dr: float = 0.01
    Qmin: float = 0.7
    Qmax: Optional[float] = 30.0

    # Outer refinement and early stopping
    n_outer: int = 24
    plot_every: int = 2
    tol_obj: float = 1e-4

    # Output directories
    out_dir: str = "out"
    fig_dir: str = os.path.join("out", "figures")
    frame_dir: str = os.path.join("out", "figures", "refine_frames")

    # Parameter step sizes
    step_rpoly: float = 0.05
    step_dE_eV: float = 2.0
    step_scale: float = 0.05
    step_fpp_offset: float = 0.05
    step_fp_offset: float = 0.05
    step_broaden_fwhm: float = 2.0
    step_alphaC: float = 0.05
    step_alphaD: float = 0.05
    step_betaC: float = 0.05
    step_betaD: float = 0.05

    # Peak step sizes
    step_peak_center_eV: float = 4.0
    step_peak_fwhm_eV: float = 1.0
    step_peak_height: float = 0.05

    # Physical constraints
    min_broaden_fwhm_eV: float = 0.0
    max_broaden_fwhm_eV: float = 40.0
    min_peak_fwhm_eV: float = 0.5
    max_peak_fwhm_eV: float = 30.0
    max_peak_height_factor: float = 1.0

    # Objective pieces
    hiQ_anchor_lambda: float = 1.0
    smooth_lambda: float = 1e-5
    peak_L1_lambda: float = 1e-3

    # Peak birth/death controls
    birth_every: int = 4
    death_height_thresh: float = 1e-2


@dataclass
class RefinementHistory:
    it: int
    obj: float
    params: Dict[str, float]


@dataclass
class Peak:
    center_eV: float
    fwhm_eV: float
    height: float


@dataclass
class EdgeModelParams:
    element: str = "Bi"
    e0_keV: float = 90.526
    dE_eV: float = 0.0
    scale: float = 1.0
    fpp_offset: float = 0.0
    fpp_slope_per_eV: float = 0.0
    broaden_fwhm_eV: float = 0.0
    peaks: List[Peak] = field(default_factory=list)
    use_chantler_f2_baseline: bool = True


def setup_directories(cfg: RefinementConfig):
    for d in [cfg.out_dir, cfg.fig_dir, cfg.frame_dir]:
        if not os.path.exists(d):
            os.makedirs(d)


# =============================================================================
# 2. EDGE MODEL & SCATTERING PHYSICS
# =============================================================================


class _BaseEdgeModel:
    def __init__(self, params: EdgeModelParams):
        self.params = params

    @staticmethod
    def to_eV(E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=float)
        scale = 1.0 if np.nanmax(E) >= 1e3 else 1e3
        return E * scale

    @staticmethod
    def lorentzian_height(
        E: np.ndarray, center: float, fwhm: float, height: float
    ) -> np.ndarray:
        gamma = 0.5 * max(float(fwhm), 1e-18)
        return height * (gamma**2) / ((E - center) ** 2 + gamma**2)

    @staticmethod
    def broadening(
        E_eV: np.ndarray, y: np.ndarray, fwhm_eV: float
    ) -> np.ndarray:
        if fwhm_eV <= 0.0:
            return y
        Emin, Emax = float(E_eV[0]), float(E_eV[-1])
        dmins = float(np.min(np.diff(E_eV)))
        d_helper = max(min(dmins / 4.0, fwhm_eV / 25.0), 1e-3)
        pad = max(5.0 * fwhm_eV, 10.0 * dmins)
        Xu = np.arange(Emin - pad, Emax + pad + d_helper, d_helper)
        yu = np.interp(Xu, E_eV, y)

        gamma = 0.5 * float(fwhm_eV)
        Lx = np.arange(-10.0 * fwhm_eV, 10.0 * fwhm_eV + d_helper, d_helper)
        Lk = (gamma / np.pi) / (Lx**2 + gamma**2)
        Lk /= Lk.sum()
        conv = np.convolve(yu, Lk, mode="same")
        return np.interp(E_eV, Xu, conv)


class KKCalcAdapter:
    def __init__(self, element: str):
        self.stoich = kkcalc.data.ParseChemicalFormula(element)
        self.rc = kkcalc.calc_relativistic_correction(self.stoich)
        self.full_E, self.full_imag_coeffs = kkcalc.data.calculate_asf(
            self.stoich
        )

    def f1_from_f2(
        self,
        energy_eval_eV: np.ndarray,
        near_edge_f2_vals: Optional[np.ndarray],
        merge_points: Optional[Tuple[float, float]] = None,
        add_background: bool = False,
        fix_distortions: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        E = np.asarray(energy_eval_eV, float)
        if near_edge_f2_vals is not None:
            NearEdge_Data = np.column_stack(
                [E, np.asarray(near_edge_f2_vals, float)]
            )
            mp = (
                (float(E[0]), float(E[-1]))
                if merge_points is None
                else merge_points
            )
            Full_E, Imaginary_Spectrum = kkcalc.data.merge_spectra(
                NearEdge_Data,
                self.full_E,
                self.full_imag_coeffs,
                merge_points=mp,
                add_background=add_background,
                fix_distortions=fix_distortions,
            )
        else:
            Full_E, Imaginary_Spectrum = self.full_E, self.full_imag_coeffs

        f1_vals = kkcalc.KK_PP(E, Full_E, Imaginary_Spectrum, self.rc)
        i1 = max(np.searchsorted(Full_E, E[0], side="right") - 1, 0)
        i2 = max(np.searchsorted(Full_E, E[-1], side="right") - 1, i1 + 1)
        f2_vals = kkcalc.data.coeffs_to_ASF(
            E, np.vstack((Imaginary_Spectrum[i1:i2], Imaginary_Spectrum[-1]))
        )
        return f1_vals, f2_vals, float(self.rc)


class EdgeModel(_BaseEdgeModel):
    def __init__(self, params: EdgeModelParams):
        super().__init__(params)
        self.kk = KKCalcAdapter(params.element)

    def fpp(self, E: np.ndarray) -> np.ndarray:
        E_eV = self.to_eV(E)
        E_eval = E_eV + float(self.params.dE_eV)
        if self.params.use_chantler_f2_baseline:
            f2 = np.asarray(
                xraydb.f2_chantler(self.params.element, E_eval), float
            )
        else:
            f2 = np.zeros_like(E_eval)

        f2 = float(self.params.scale) * f2 + float(self.params.fpp_offset)
        f2 += float(self.params.fpp_slope_per_eV) * (
            E_eV - float(self.params.e0_keV) * 1e3
        )

        if self.params.peaks:
            add = np.zeros_like(f2)
            for pk in self.params.peaks:
                add += self.lorentzian_height(
                    E_eval, pk.center_eV, pk.fwhm_eV, pk.height
                )
            f2 = f2 + add
        return self.broadening(E_eV, f2, self.params.broaden_fwhm_eV)

    def fp(
        self,
        E: np.ndarray,
        fpp: Optional[np.ndarray] = None,
        return_f1: bool = False,
        merge_points: Optional[Tuple[float, float]] = None,
        add_background: bool = False,
        fix_distortions: bool = False,
    ) -> np.ndarray:
        E_eV = self.to_eV(E)
        E_eval = E_eV + float(self.params.dE_eV)
        if fpp is None:
            fpp = self.fpp(E_eV)
        f1_vals, f2_vals, f1_inf = self.kk.f1_from_f2(
            energy_eval_eV=E_eval,
            near_edge_f2_vals=fpp,
            merge_points=merge_points,
            add_background=add_background,
            fix_distortions=fix_distortions,
        )
        return f1_vals if return_f1 else (f1_vals - f1_inf)


# =============================================================================
# 3. UTILITIES & PIPELINE FUNCTIONS
# =============================================================================


def _as_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).ravel()


def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def _safe(a: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(a)):
        raise ValueError(
            f"{name} has non-finite entries at {np.where(~np.isfinite(a))}"
        )
    return a


def _lorch(Q: np.ndarray, Qmax: float) -> np.ndarray:
    x = Q / Qmax
    w = np.ones_like(Q)
    m = (x > 0) & (x <= 1.0)
    w[m] = np.sin(np.pi * x[m]) / (np.pi * x[m])
    w[x > 1.0] = 0.0
    return w


def _kaiser(Q: np.ndarray, Qmax: float, beta: float = 6.0) -> np.ndarray:
    from numpy import i0

    x = Q / Qmax
    w = np.zeros_like(Q)
    m = (x >= 0) & (x <= 1.0)
    w[m] = i0(beta * np.sqrt(1.0 - x[m] ** 2)) / i0(beta)
    return w


def _ensure_2d(x: np.ndarray, ne: int, nq: int, name: str) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim == 0:
        return np.full((ne, nq), float(x))
    if x.ndim == 1:
        if x.size == ne:
            return x[:, None] * np.ones((1, nq))
        if x.size == nq:
            return np.ones((ne, 1)) * x[None, :]
        raise ValueError(f"{name} 1D mismatch")
    if x.shape != (ne, nq):
        raise ValueError(f"{name} shape mismatch")
    return x


def _f0_of_Q(element: str, Q: np.ndarray) -> np.ndarray:
    s = _as_1d(Q) / (4.0 * np.pi)
    f0 = np.array([xraydb.f0(element, si)[0] for si in s], dtype=float)
    return _safe(f0, f"f0({element})")


def _compute_scattering(
    elements: List[str],
    comp: Dict[str, float],
    alpha: str,
    Q: np.ndarray,
    E_eV: np.ndarray,
    edge_model: Any,
    fp_offset: float = 0.0,
):
    Q = _as_1d(Q)
    E = _as_1d(E_eV)
    nq, ne = len(Q), len(E)
    f0_map = {el: _f0_of_Q(el, Q) for el in elements}

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

    fmean_real, fmean_abs2, f_not_sum = (
        np.zeros((ne, nq), float),
        np.zeros((ne, nq), float),
        np.zeros((ne, nq), float),
    )
    for el in elements:
        c = float(comp[el])
        fmean_real += c * fr_weights[el]
        fmean_abs2 += c * f2_weights[el]
        if el != alpha:
            f_not_sum += c * fr_weights[el]
    return fr_weights, f0_map, fmean_real, fmean_abs2, fmean_abs2, f_not_sum


def _weights_alpha_not(
    comp: Dict[str, float],
    alpha: str,
    fr_weights: Dict[str, np.ndarray],
    fmean_real: np.ndarray,
    f_not_sum: np.ndarray,
):
    cA = float(comp[alpha])
    eps = 1e-12
    a = np.clip(cA * fr_weights[alpha] / (fmean_real + eps), -5.0, 5.0)
    b = np.clip(f_not_sum / (fmean_real + eps), -5.0, 5.0)
    return a, b, {"a_min": float(np.nanmin(a)), "a_max": float(np.nanmax(a))}


def _alpha_beta_from_fpp(
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


def _build_S_from_raw(
    I_raw: np.ndarray,
    Q: np.ndarray,
    E_eV: np.ndarray,
    alpha_e: np.ndarray,
    beta_f_e: np.ndarray,
    fmean_real: np.ndarray,
    fself: np.ndarray,
    rpoly: float,
):
    I_raw = np.asarray(I_raw, float)
    E = _as_1d(E_eV)
    Q = _as_1d(Q)
    ne, _ = len(E), len(Q)

    I_corr = I_raw * alpha_e[:, None] + beta_f_e[:, None]
    S0 = (I_corr - fself) / (fmean_real**2 + 1e-12) + 1.0

    # WLS Fractional Polynomial Fitting
    xspan = float(Q[-1] - Q[0])
    porder = max(1.0, (rpoly * xspan) / np.pi)
    porderlo, porderhi = int(np.floor(porder)), int(np.ceil(porder))
    poly_weights = (
        [0.5, 0.5]
        if porderlo == porderhi
        else [porderhi - porder, porder - porderlo]
    )

    weight_power = 4.0
    W_sqrt = np.sqrt((Q / Q[-1]) ** weight_power)

    Xlo = np.vstack([Q**k for k in range(porderlo + 1)]).T
    Xhi = np.vstack([Q**k for k in range(porderhi + 1)]).T
    Xlo_w, Xhi_w = Xlo * W_sqrt[:, None], Xhi * W_sqrt[:, None]
    XtXlo = Xlo_w.T @ Xlo_w + 1e-12 * np.eye(Xlo_w.shape[1])
    XtXhi = Xhi_w.T @ Xhi_w + 1e-12 * np.eye(Xhi_w.shape[1])

    beta_o = np.zeros_like(S0)
    for i in range(ne):
        y = (1.0 - S0[i]) * (fmean_real[i] ** 2)
        y_w = y * W_sqrt
        coeff_lo = npl.solve(XtXlo, Xlo_w.T @ y_w)
        beta_lo = (Xlo @ coeff_lo) / (fmean_real[i] ** 2 + 1e-12)

        if porderlo == porderhi:
            beta_o[i] = beta_lo
        else:
            coeff_hi = npl.solve(XtXhi, Xhi_w.T @ y_w)
            beta_hi = (Xhi @ coeff_hi) / (fmean_real[i] ** 2 + 1e-12)
            beta_o[i] = poly_weights[0] * beta_lo + poly_weights[1] * beta_hi

    return S0 + beta_o, beta_o


def _solve_two_channel_column(
    Aj: np.ndarray, yj: np.ndarray, ridge: float, anchor: float
):
    ATA = Aj.T @ Aj + ridge * np.eye(2) + anchor * np.eye(2)
    ATy = Aj.T @ yj + anchor * np.array([[1.0], [1.0]])
    try:
        sol = npl.solve(ATA, ATy)
    except npl.LinAlgError:
        sol = npl.lstsq(ATA, ATy, rcond=None)[0]
    return float(sol[0, 0]), float(sol[1, 0])


def _reconstruct_Salpha_Snot(
    S_meas: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    Q: np.ndarray,
    ridge: float,
):
    S_meas = np.asarray(S_meas, float)
    for i in range(S_meas.shape[0]):
        S_meas[i][:40] = S_meas[i][41]
    ne, nq = S_meas.shape
    a = _ensure_2d(a, ne, nq, "a")
    b = _ensure_2d(b, ne, nq, "b")

    Sa, Sb = np.zeros(nq), np.zeros(nq)
    bad_cols = 0
    for j in range(nq):
        Aj, yj = np.stack([a[:, j], b[:, j]], axis=1), S_meas[:, j : j + 1]
        s = npl.svd(Aj, compute_uv=False)
        cond = (s[0] / s[-1]) if (s[-1] > 0) else np.inf
        anchor = 5.0 if (not np.isfinite(cond) or cond > 1e6) else 0.0
        if anchor > 0.0:
            bad_cols += 1
        Sa[j], Sb[j] = _solve_two_channel_column(Aj, yj, ridge, anchor)
    return Sa, Sb, {"bad_Q_columns": bad_cols}


def _S_to_G(
    Q: np.ndarray, S: np.ndarray, r: np.ndarray, window: str, kbeta: float
) -> np.ndarray:
    Q = _as_1d(Q)
    S = _as_1d(S)
    r = _as_1d(r)
    Qmax = float(Q.max())
    w = np.ones_like(Q)
    if window == "lorch":
        w = _lorch(Q, Qmax)
    elif window == "kaiser":
        w = _kaiser(Q, Qmax, kbeta)
    F = Q * (S - 1.0) * w
    G = np.zeros_like(r)
    for i, ri in enumerate(r):
        G[i] = (2.0 / np.pi) * _trapz(Q, F * np.sin(Q * ri))
    return G


# =============================================================================
# 4. REFINEMENT & PLOTTING
# =============================================================================


def save_initial_edge_plots(
    E_eV: np.ndarray, edge_model: EdgeModel, cfg: RefinementConfig
):
    fpp_init = edge_model.fpp(E_eV)
    fp_init = edge_model.fp(E_eV)
    fp_chantler = xraydb.f1_chantler(
        edge_model.params.element,
        edge_model.to_eV(E_eV) + edge_model.params.dE_eV,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(E_eV / 1000, fpp_init, label="f'' (Model/Chantler)")
    ax[0].set_xlabel("Energy (keV)")
    ax[0].set_title("Initial f''")
    ax[0].legend()

    ax[1].plot(E_eV / 1000, fp_init, label="f' (Model)")
    ax[1].plot(E_eV / 1000, fp_chantler, "--", label="f' (Chantler Tabulated)")
    ax[1].set_xlabel("Energy (keV)")
    ax[1].set_title("Initial f'")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.fig_dir, "fpp_fp_initial.png"), dpi=150)
    plt.close()


def _plot_frame(
    path: str,
    E_eV: np.ndarray,
    edge_model: Any,
    r: np.ndarray,
    out: Dict[str, np.ndarray],
    Q_full: np.ndarray,
    iter_idx: int,
    hist: List[RefinementHistory],
    fp_offset: float,
) -> None:
    q = np.asarray(out.get("Q", Q_full))
    Sa, Sb, St = (
        np.asarray(out["S_alpha"]),
        np.asarray(out["S_not"]),
        np.asarray(out["S_tot"]),
    )
    r_plot = np.asarray(r)
    Ga, Gn, Gt = (
        np.asarray(out["G_alpha"]),
        np.asarray(out["G_not"]),
        np.asarray(out["G_tot"]),
    )

    E = np.asarray(E_eV, float)
    fpp = np.maximum(edge_model.fpp(E), 0.0)
    fp = edge_model.fp(E, fpp=fpp) + fp_offset

    plt.figure(figsize=(12, 20))
    plt.suptitle(f"Refinement frame {iter_idx:03d}")

    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(E / 1e3, fpp, label="f'' (model)")
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(E / 1e3, fp, label="f' (KK from model) + fp_offset")
    ax2.legend(loc="upper left")

    ax = plt.subplot(4, 1, 2)
    ax.plot(r_plot, Ga, "b-", label="G_alpha")
    ax.plot(r_plot, Gn, "g-", label="G_not")
    ax.plot(r_plot, Gt, "r-", label="G_tot")
    ax.legend(loc="upper right")

    ax = plt.subplot(4, 1, 3)
    ax.plot(q, Sa, "b-", label="S_alpha", linewidth=0.5)
    ax.plot(q, Sb, "g-", label="S_not", linewidth=0.5)
    ax.plot(q, St, "r-", label="S_tot", linewidth=0.5)
    ax.plot(q, np.ones_like(q), "k-", linewidth=0.1)
    ax.legend(loc="upper left")

    ax = plt.subplot(4, 1, 4)
    its = np.arange(len(hist))
    vals = np.array([h.obj for h in hist], float)
    ax.plot(its, vals, "-o")
    ax.set_yscale("log")
    p = hist[-1].params
    txt = (
        f"dE_eV={p['dE_eV']:.3g}\nscale={p['scale']:.3g}\n"
        f"alpha_C={p['alpha_C']:.3g}, beta_C={p['beta_C']:.3g}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        va="top",
        ha="left",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.savefig(path, dpi=150)
    plt.close()


def save_results(out: Dict[str, np.ndarray], cfg: RefinementConfig):
    """Save the output PDFs to .cgr files and export a final summary plot."""
    r = out["r"]
    np.savetxt(
        os.path.join(cfg.out_dir, "G_alpha.cgr"),
        np.column_stack([r, out["G_alpha"]]),
        header="r G_alpha",
    )
    np.savetxt(
        os.path.join(cfg.out_dir, "G_not.cgr"),
        np.column_stack([r, out["G_not"]]),
        header="r G_not",
    )
    np.savetxt(
        os.path.join(cfg.out_dir, "G_tot.cgr"),
        np.column_stack([r, out["G_tot"]]),
        header="r G_tot",
    )

    # Plot the full r-range to preserve diagnostic low-r ripples
    plt.figure()
    plt.plot(r, out["G_alpha"], "r", label="G_alpha (resonant)")
    plt.plot(r, out["G_not"], "g", label="G_not (non-resonant)")
    plt.plot(r, out["G_tot"], "b", label="G_tot (total/mean)")

    plt.xlabel("r (Å)")
    plt.ylabel("G(r)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.fig_dir, "final_results.png"), dpi=150)
    plt.close()
    print("Final .cgr files and full-range comparison plot saved.")


def _objective(
    Sa: np.ndarray,
    Sb: np.ndarray,
    Q: np.ndarray,
    hiQ_frac: float,
    anchor_lambda: float,
) -> float:
    nq = len(Q)
    k0 = int(max(0, math.floor((1.0 - hiQ_frac) * nq)))
    mask = np.zeros(nq, bool)
    mask[k0:] = True
    tails = np.sum((Sa[mask] - 1.0) ** 2 + (Sb[mask] - 1.0) ** 2)
    d2a = np.diff(Sa, n=2)
    d2b = np.diff(Sb, n=2)
    smooth = np.sum(d2a**2 + d2b**2)
    physics = np.sum((Sa - Sb) ** 2)
    return float(anchor_lambda * tails + smooth + physics)


def reconstruct_partials_and_pdfs(
    I_raw,
    Q,
    E_eV,
    elements,
    comp,
    alpha,
    edge_model,
    alpha_e,
    beta_e,
    rpoly,
    ridge,
    r,
    Qmin,
    Qmax,
    window="lorch",
    kaiser_beta=6.0,
    fp_offset=0.0,
):
    Q = _as_1d(Q)
    E = _as_1d(E_eV)
    I_raw = np.asarray(I_raw, float)
    frw, f0_map, fmean_real, fmean_abs2, fself, f_not_sum = (
        _compute_scattering(
            elements, comp, alpha, Q, E, edge_model, fp_offset=fp_offset
        )
    )
    a, b, wb = _weights_alpha_not(comp, alpha, frw, fmean_real, f_not_sum)

    if Qmax is None:
        Qmax = float(Q.max())
    mask_Q = (Q >= Qmin) & (Q <= Qmax)
    Q_use = Q[mask_Q]

    S_meas_all, beta_o_all = _build_S_from_raw(
        I_raw, Q, E, alpha_e, beta_e, fmean_real, fself, rpoly
    )
    S_meas = S_meas_all[:, mask_Q]
    a = a[:, mask_Q]
    b = b[:, mask_Q]

    Sa, Sb, condinfo = _reconstruct_Salpha_Snot(S_meas, a, b, Q_use, ridge)
    S_tot = S_meas[0]

    r = _as_1d(r)
    G_alpha = _S_to_G(Q_use, Sa, r, window, kaiser_beta)
    G_not = _S_to_G(Q_use, Sb, r, window, kaiser_beta)
    G_tot = _S_to_G(Q_use, S_tot, r, window, kaiser_beta)

    out = {
        "Q": Q_use,
        "r": r,
        "S_alpha": Sa,
        "S_not": Sb,
        "S_tot": S_tot,
        "G_alpha": G_alpha,
        "G_not": G_not,
        "G_tot": G_tot,
    }
    return out, {"weights": wb, "condinfo": condinfo, "beta_o": beta_o_all}


def run_adpdf_with_refinement(
    I_raw,
    Q,
    E_eV,
    elements,
    comp,
    alpha,
    edge_model,
    alpha_C=1.0,
    alpha_D=0.0,
    beta_C=0.0,
    beta_D=0.0,
    fp_offset=0.0,
    cfg=None,
):
    if cfg is None:
        cfg = RefinementConfig()
    Q = _as_1d(Q)
    E = _as_1d(E_eV)
    I_raw = np.asarray(I_raw, float)
    rpoly = cfg.rpoly
    edge_model.params.scale = max(1e-4, float(edge_model.params.scale))
    edge_model.params.broaden_fwhm_eV = float(
        np.clip(
            edge_model.params.broaden_fwhm_eV,
            cfg.min_broaden_fwhm_eV,
            cfg.max_broaden_fwhm_eV,
        )
    )

    r = np.arange(cfg.rmin, cfg.rmax + cfg.dr / 2.0, cfg.dr)
    Qmax = cfg.Qmax if cfg.Qmax is not None else float(Q.max())

    def compute():
        fpp = np.maximum(edge_model.fpp(E), 0.0)
        alpha_e, beta_e = _alpha_beta_from_fpp(
            fpp, alpha_C, alpha_D, beta_C, beta_D
        )
        out, diag = reconstruct_partials_and_pdfs(
            I_raw,
            Q,
            E,
            elements,
            comp,
            alpha,
            edge_model,
            alpha_e,
            beta_e,
            rpoly=rpoly,
            ridge=cfg.ridge,
            r=r,
            Qmin=cfg.Qmin,
            Qmax=Qmax,
            window=cfg.window,
            kaiser_beta=cfg.kaiser_beta,
            fp_offset=fp_offset,
        )
        return (
            out,
            diag,
            _objective(
                out["S_alpha"],
                out["S_not"],
                out["Q"],
                cfg.hiQ_frac,
                cfg.hiQ_anchor_lambda,
            ),
        )

    out, diag, obj = compute()
    hist = [
        RefinementHistory(
            0,
            obj,
            {
                "scale": edge_model.params.scale,
                "dE_eV": edge_model.params.dE_eV,
                "alpha_C": alpha_C,
                "beta_C": beta_C,
            },
        )
    ]
    os.makedirs(cfg.frame_dir, exist_ok=True)
    _plot_frame(
        os.path.join(cfg.frame_dir, f"frame_{0:03d}.png"),
        E,
        edge_model,
        r,
        out,
        Q,
        0,
        hist,
        fp_offset,
    )

    steps = dict(
        rpoly=cfg.step_rpoly,
        dE_eV=cfg.step_dE_eV,
        scale=cfg.step_scale,
        fpp_offset=cfg.step_fpp_offset,
        fp_offset=cfg.step_fp_offset,
        broaden_fwhm_eV=cfg.step_broaden_fwhm,
        alpha_C=cfg.step_alphaC,
        alpha_D=cfg.step_alphaD,
        beta_C=cfg.step_betaC,
        beta_D=cfg.step_betaD,
    )

    def tweak(name: str, d: float):
        nonlocal alpha_C, alpha_D, beta_C, beta_D, fp_offset, rpoly
        p = edge_model.params
        if name == "rpoly":
            rpoly = float(np.clip(rpoly + d, 0.0, 2.0))
        elif name == "dE_eV":
            p.dE_eV += d
        elif name == "scale":
            p.scale = max(1e-4, p.scale + d)
        elif name == "fpp_offset":
            p.fpp_offset += d
        elif name == "broaden_fwhm_eV":
            p.broaden_fwhm_eV = float(
                np.clip(
                    p.broaden_fwhm_eV + d,
                    cfg.min_broaden_fwhm_eV,
                    cfg.max_broaden_fwhm_eV,
                )
            )
        elif name == "alpha_C":
            alpha_C = max(1e-6, alpha_C + d)
        elif name == "alpha_D":
            alpha_D += d
        elif name == "beta_C":
            beta_C += d
        elif name == "beta_D":
            beta_D += d
        elif name == "fp_offset":
            fp_offset += d

    best_obj = obj
    for it in range(1, cfg.n_outer + 1):
        cycle_improved = False
        start_obj = best_obj

        for name in [
            "rpoly",
            "dE_eV",
            "scale",
            "fpp_offset",
            "fp_offset",
            "broaden_fwhm_eV",
            "alpha_C",
            "alpha_D",
            "beta_C",
            "beta_D",
        ]:
            param_improved = False
            for sgn in (+1, -1):
                tweak(name, sgn * steps[name])
                out2, diag2, obj2 = compute()
                if obj2 < best_obj:
                    out, diag, best_obj = out2, diag2, obj2
                    param_improved = True
                    cycle_improved = True
                else:
                    tweak(name, -sgn * steps[name])
            if not param_improved:
                steps[name] *= 0.7

        hist.append(
            RefinementHistory(
                it,
                best_obj,
                {
                    "scale": edge_model.params.scale,
                    "dE_eV": edge_model.params.dE_eV,
                    "alpha_C": alpha_C,
                    "beta_C": beta_C,
                },
            )
        )
        if (it % cfg.plot_every) == 0 or it == cfg.n_outer:
            _plot_frame(
                os.path.join(cfg.frame_dir, f"frame_{it:03d}.png"),
                E,
                edge_model,
                r,
                out,
                Q,
                it,
                hist,
                fp_offset,
            )

        rel_change = (start_obj - best_obj) / max(start_obj, 1e-12)
        if rel_change < cfg.tol_obj:
            print(
                f"Converged! Relative change ({rel_change:.2e}) "
                "is below tolerance."
            )
            break
        if not cycle_improved and max(steps.values()) < 1e-3:
            print(f"Stopped at iteration {it}: step sizes shrank below 1e-3.")
            break

    return out, diag, hist


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================


def main():
    # 1. Initialize Configuration
    cfg = RefinementConfig()
    setup_directories(cfg)

    # 2. Load Data
    # Make sure this points to the correct local file path
    print("Loading data...")
    try:
        with h5py.File("data/ESRF.h5", "r") as f:
            data = f["data"]
            I_raw = np.array(data["iq"])
            E = np.array(data["energy"]) * 1e3
            Q = np.array(data["q"])
    except FileNotFoundError:
        print("Data file 'data/ESRF.h5' not found. Please verify the path.")
        return

    # 3. Setup Elements & Edge Model
    elements = ["Ba", "Bi", "O"]
    comp = {"Ba": 1 / 5, "Bi": 1 / 5, "O": 3 / 5}
    alpha = "Bi"

    params = EdgeModelParams(
        element=alpha,
        e0_keV=90.526,
        scale=0.35,
        fpp_offset=-0.2,
        broaden_fwhm_eV=15.0,
        peaks=[Peak(center_eV=90.526e3 + 180, fwhm_eV=3.0, height=0.3)],
    )
    edge_model = EdgeModel(params)

    # 4. Save Initial Plots
    save_initial_edge_plots(E, edge_model, cfg)

    # 5. Run Refinement
    print("Starting refinement...")
    out, diag, hist = run_adpdf_with_refinement(
        I_raw=I_raw,
        Q=Q,
        E_eV=E,
        elements=elements,
        comp=comp,
        alpha=alpha,
        edge_model=edge_model,
        beta_D=0.1,
        cfg=cfg,
    )

    # 6. Output Final Results
    save_results(out, cfg)


if __name__ == "__main__":
    main()
