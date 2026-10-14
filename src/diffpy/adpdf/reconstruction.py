"""Build S(Q) from raw intensities, separate the partial structure functions
and transform to G(r)."""

import numpy as np
import numpy.linalg as npl

from diffpy.adpdf.scattering import compute_scattering, weights_alpha_not
from diffpy.adpdf.utils import as_1d, ensure_2d, kaiser, lorch, trapz


def build_S_from_raw(
    I_raw: np.ndarray,
    Q: np.ndarray,
    E_eV: np.ndarray,
    alpha_e: np.ndarray,
    beta_f_e: np.ndarray,
    w_tot: np.ndarray,
    fself: np.ndarray,
    rpoly: float,
    bkg_poly_type: str = "wls",
):
    I_raw = np.asarray(I_raw, float)
    E = as_1d(E_eV)
    Q = as_1d(Q)
    ne, _ = len(E), len(Q)

    # Apply energy-dependent absorption and fluorescence corrections
    I_corr = I_raw * alpha_e[:, None] + beta_f_e[:, None]

    # Normalize using the complex absolute squared average scattering factor
    S0 = (I_corr - fself) / (w_tot + 1e-12) + 1.0

    # Determine polynomial degree
    xspan = float(Q[-1] - Q[0])
    porder = max(1.0, (rpoly * xspan) / np.pi)
    porderlo, porderhi = int(np.floor(porder)), int(np.ceil(porder))
    poly_weights = (
        [0.5, 0.5]
        if porderlo == porderhi
        else [porderhi - porder, porder - porderlo]
    )

    beta_o = np.zeros_like(S0)

    # ---------------------------------------------------------
    # OPTION A: getx style F(Q) polynomial fitting
    # ---------------------------------------------------------
    if bkg_poly_type == "getx":
        # Build F(Q) matrix WITHOUT the constant Q^0 term (forces F(0)=0)
        Vlo_F = np.vander(Q, porderlo + 1)[:, :-1]
        Vhi_F = np.vander(Q, porderhi + 1)[:, :-1]

        # Build the exact S(Q) matrix to apply coefficients cleanly
        Vlo_S = np.vander(Q, porderlo)
        Vhi_S = np.vander(Q, porderhi)

        for i in range(ne):
            target = Q * (S0[i] - 1.0)

            # Fit fractional low-order to F(Q)
            coeffs_lo = npl.lstsq(Vlo_F, target, rcond=None)[0]
            P_lo_S = np.dot(Vlo_S, coeffs_lo)

            if porderlo == porderhi:
                P_eval_S = P_lo_S
            else:
                # Fit fractional high-order and blend
                coeffs_hi = npl.lstsq(Vhi_F, target, rcond=None)[0]
                P_hi_S = np.dot(Vhi_S, coeffs_hi)
                P_eval_S = poly_weights[0] * P_lo_S + poly_weights[1] * P_hi_S

            beta_o[i] = -P_eval_S

    # ---------------------------------------------------------
    # OPTION B: Weighted Least Squares on S(Q)
    # ---------------------------------------------------------
    elif bkg_poly_type == "wls":
        weight_power = 4.0
        W_sqrt = np.sqrt((Q / Q[-1]) ** weight_power)

        Xlo = np.vstack([Q**k for k in range(porderlo + 1)]).T
        Xhi = np.vstack([Q**k for k in range(porderhi + 1)]).T
        Xlo_w, Xhi_w = Xlo * W_sqrt[:, None], Xhi * W_sqrt[:, None]
        XtXlo = Xlo_w.T @ Xlo_w + 1e-12 * np.eye(Xlo_w.shape[1])
        XtXhi = Xhi_w.T @ Xhi_w + 1e-12 * np.eye(Xhi_w.shape[1])

        for i in range(ne):
            y = (1.0 - S0[i]) * w_tot[i]
            y_w = y * W_sqrt
            coeff_lo = npl.solve(XtXlo, Xlo_w.T @ y_w)
            beta_lo = (Xlo @ coeff_lo) / (w_tot[i] + 1e-12)

            if porderlo == porderhi:
                beta_o[i] = beta_lo
            else:
                coeff_hi = npl.solve(XtXhi, Xhi_w.T @ y_w)
                beta_hi = (Xhi @ coeff_hi) / (w_tot[i] + 1e-12)
                beta_o[i] = (
                    poly_weights[0] * beta_lo + poly_weights[1] * beta_hi
                )

    else:
        raise ValueError(f"Unknown bkg_poly_type: {bkg_poly_type}")

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


def reconstruct_Salpha_Snot(
    S_meas: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    Q: np.ndarray,
    ridge: float,
):
    S_meas = np.asarray(S_meas, float)
    # TODO(physics): hardcoded low-Q flattening (first 40 points set to
    # point 41, skipping 40). This modifies the caller's array in place, so
    # S_tot (= S_meas[0] in reconstruct_partials_and_pdfs) is flattened too.
    for i in range(S_meas.shape[0]):
        S_meas[i][:40] = S_meas[i][41]
    ne, nq = S_meas.shape
    a = ensure_2d(a, ne, nq, "a")
    b = ensure_2d(b, ne, nq, "b")

    Sa, Sb = np.zeros(nq), np.zeros(nq)
    bad_cols = 0
    for j in range(nq):
        Aj, yj = np.stack([a[:, j], b[:, j]], axis=1), S_meas[:, j : j + 1]
        s = npl.svd(Aj, compute_uv=False)
        cond = (s[0] / s[-1]) if (s[-1] > 0) else np.inf
        # TODO(physics): heuristic anchor strength and condition threshold
        anchor = 5.0 if (not np.isfinite(cond) or cond > 1e6) else 0.0
        if anchor > 0.0:
            bad_cols += 1
        Sa[j], Sb[j] = _solve_two_channel_column(Aj, yj, ridge, anchor)
    return Sa, Sb, {"bad_Q_columns": bad_cols}


def S_to_G(
    Q: np.ndarray, S: np.ndarray, r: np.ndarray, window: str, kbeta: float
) -> np.ndarray:
    Q = as_1d(Q)
    S = as_1d(S)
    r = as_1d(r)
    Qmax = float(Q.max())
    w = np.ones_like(Q)
    if window == "lorch":
        w = lorch(Q, Qmax)
    elif window == "kaiser":
        w = kaiser(Q, Qmax, kbeta)
    F = Q * (S - 1.0) * w
    G = np.zeros_like(r)
    for i, ri in enumerate(r):
        G[i] = (2.0 / np.pi) * trapz(Q, F * np.sin(Q * ri))
    return G


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
    bkg_poly_type="wls",
):
    Q = as_1d(Q)
    E = as_1d(E_eV)
    I_raw = np.asarray(I_raw, float)
    frw, fi_alpha, fmean_real, fmean_imag, w_tot, fself = compute_scattering(
        elements, comp, alpha, Q, E, edge_model, fp_offset=fp_offset
    )

    a, b, wb = weights_alpha_not(
        comp, alpha, frw, fi_alpha, fmean_real, fmean_imag, w_tot
    )

    if Qmax is None:
        Qmax = float(Q.max())
    mask_Q = (Q >= Qmin) & (Q <= Qmax)
    Q_use = Q[mask_Q]

    # Pass w_tot here instead of fmean_real
    S_meas_all, beta_o_all = build_S_from_raw(
        I_raw, Q, E, alpha_e, beta_e, w_tot, fself, rpoly, bkg_poly_type
    )
    S_meas = S_meas_all[:, mask_Q]
    a = a[:, mask_Q]
    b = b[:, mask_Q]

    Sa, Sb, condinfo = reconstruct_Salpha_Snot(S_meas, a, b, Q_use, ridge)
    S_tot = S_meas[0]

    r = as_1d(r)
    G_alpha = S_to_G(Q_use, Sa, r, window, kaiser_beta)
    G_not = S_to_G(Q_use, Sb, r, window, kaiser_beta)
    G_tot = S_to_G(Q_use, S_tot, r, window, kaiser_beta)

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
