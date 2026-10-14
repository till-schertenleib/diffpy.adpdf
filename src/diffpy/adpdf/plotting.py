"""Diagnostic and result plots."""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xraydb

from diffpy.adpdf.config import RefinementConfig, RefinementHistory
from diffpy.adpdf.edge import EdgeModel


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


def plot_frame(
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


def plot_final_results(out: Dict[str, np.ndarray], cfg: RefinementConfig):
    r = out["r"]
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
