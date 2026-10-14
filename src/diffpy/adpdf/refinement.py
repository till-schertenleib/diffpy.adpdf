"""Coordinate-descent refinement of the edge and correction parameters."""

import math
import os

import numpy as np

from diffpy.adpdf.config import RefinementConfig, RefinementHistory
from diffpy.adpdf.plotting import plot_frame
from diffpy.adpdf.reconstruction import reconstruct_partials_and_pdfs
from diffpy.adpdf.scattering import alpha_beta_from_fpp
from diffpy.adpdf.utils import as_1d


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


def run_adpdf_with_refinement(
    I_raw,
    Q,
    E_eV,
    elements,
    comp,
    alpha,
    edge_model,
    cfg=None,
):
    if cfg is None:
        cfg = RefinementConfig()

    alpha_C = cfg.init_alpha_C
    alpha_D = cfg.init_alpha_D
    beta_C = cfg.init_beta_C
    beta_D = cfg.init_beta_D
    fp_offset = cfg.init_fp_offset

    Q = as_1d(Q)
    E = as_1d(E_eV)
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
        alpha_e, beta_e = alpha_beta_from_fpp(
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
            bkg_poly_type=cfg.bkg_poly_type,
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
    plot_frame(
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
    # active_params = ["rpoly", "dE_eV", "scale", "alpha_C", "alpha_D", "beta_C", "beta_D"]
    active_params = [
        "rpoly",
        "dE_eV",
        "scale",
        "alpha_D",
        "beta_C",
        "beta_D",
    ]  # alpha_C (and possibly alpha_D) should be excluded for very dilute resonant scatterer

    # If using theoretical Chantler tables, allow tuning the edge shape
    if getattr(edge_model, "exp_energy", None) is None:
        active_params.extend(["fpp_offset", "fp_offset", "broaden_fwhm_eV"])

    for it in range(1, cfg.n_outer + 1):
        cycle_improved = False
        start_obj = best_obj

        for name in active_params:
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
            plot_frame(
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
