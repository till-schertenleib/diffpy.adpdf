"""Configuration dataclasses for the AD-PDF pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RefinementConfig:
    # S(Q) construction
    bkg_poly_type: str = "wls"  # Options: 'wls' or 'getx'
    rpoly: float = 1.0
    hiQ_frac: float = 0.85
    ridge: float = 8e-4

    # G(r)
    window: str = "lorch"
    kaiser_beta: float = 6.0
    rmin: float = 0.0
    rmax: float = 40.0
    dr: float = 0.01
    Qmin: float = 0.7
    Qmax: Optional[float] = 19.0

    # Outer refinement and early stopping
    n_outer: int = 24
    plot_every: int = 2
    tol_obj: float = 1e-4

    # Output directories
    out_dir: str = (
        "C:/Users/tills/PDF/danmax_maxiv/ts-15-3-1200_resonant/adpdf_ts-15-3_sn_edge_out"
    )
    fig_dir: str = os.path.join(
        "C:/Users/tills/PDF/danmax_maxiv/ts-15-3-1200_resonant/adpdf_ts-15-3_sn_edge_out",
        "figures",
    )
    frame_dir: str = os.path.join(
        "C:/Users/tills/PDF/danmax_maxiv/ts-15-3-1200_resonant/adpdf_ts-15-3_sn_edge_out",
        "figures",
        "refine_frames",
    )

    # Initial Parameter Values
    init_alpha_C: float = 1.0
    init_alpha_D: float = 0.0
    init_beta_C: float = 0.0
    init_beta_D: float = 0.0
    init_fp_offset: float = 0.0

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
    experimental_xas_file: Optional[str] = None
    experimental_xas_shift_eV: float = 0.0
