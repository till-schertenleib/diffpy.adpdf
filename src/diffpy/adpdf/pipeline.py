"""Run the AD-PDF pipeline end to end."""

import os

from diffpy.adpdf.config import EdgeModelParams, RefinementConfig
from diffpy.adpdf.edge import EdgeModel
from diffpy.adpdf.fileio import (
    load_background,
    load_experimental_data,
    save_results,
    setup_directories,
)
from diffpy.adpdf.plotting import plot_final_results, save_initial_edge_plots
from diffpy.adpdf.refinement import run_adpdf_with_refinement


def main():
    # 1. Initialize Configuration
    cfg = RefinementConfig()
    cfg.bkg_poly_type = "wls"
    cfg.rpoly = 1.0
    cfg.init_alpha_C = 0.2
    cfg.init_alpha_D = 0.05
    cfg.init_beta_C = 0.05
    cfg.init_beta_D = 0.1  # Set to 0.1 as you had previously
    cfg.init_fp_offset = 0.0
    cfg.ridge = 8e-3
    setup_directories(cfg)

    # 2. Load Experimental Scattering Data
    data_path = "C:/Users/tills/PDF/danmax_maxiv/ts-15-3-1200_resonant/ts-15-3-1200_sn_fine_spikes_removed"
    energies_path = "C:/Users/tills/PDF/danmax_maxiv/ts-15-3-1200_resonant/ts-15-3-1200_sn_energies.txt"

    print(f"Loading data from '{data_path}'...")
    try:
        I_raw, Q, E = load_experimental_data(data_path, energies_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # BACKGROUND SUBTRACTION ---
    bkg_file = None  # e.g., "data/empty_capillary.xy"
    bkg_scale = 1.0  # Tweak this to perfectly strip the container/background

    if bkg_file is not None:
        if os.path.isfile(bkg_file):
            print(
                f"Subtracting background from '{bkg_file}' (scale={bkg_scale})..."
            )
            try:
                I_bkg = load_background(bkg_file, Q)
                # Subtract the scaled background from every single energy row
                I_raw = I_raw - (bkg_scale * I_bkg[None, :])
            except Exception as e:
                print(f"Warning: Failed to process background file: {e}")
        else:
            print(
                f"Warning: Background file '{bkg_file}' not found. Skipping subtraction."
            )
    # -----------------------------------

    # Setup Elements & Edge Model
    # elements = ["Ba", "Bi", "O"]
    # comp = {"Ba": 1 / 5, "Bi": 1 / 5, "O": 3 / 5}
    # alpha = "Bi"

    elements = ["Sr", "Ru", "Sn", "O"]
    comp = {"Sr": 1 / 5, "Ru": 0.04 / 5, "Sn": 0.96 / 5, "O": 3 / 5}
    alpha = "Sn"

    # for Bi ESRF
    # params = EdgeModelParams(
    #     element=alpha,
    #     e0_keV=90.526,
    #     scale=0.35,
    #     fpp_offset=-0.2,
    #     broaden_fwhm_eV=15.0,
    #     peaks=[Peak(center_eV=90.526e3 + 180, fwhm_eV=3.0, height=0.3)],
    # )
    # edge_model = EdgeModel(params)

    params = EdgeModelParams(
        element=alpha,
        e0_keV=90.526,
        scale=0.35,
        experimental_xas_file="C:/Users/tills/PDF/danmax_maxiv/ts-15-3-1200_resonant/xas/ts-15-3-1200_Sn_exafs_T_merge.dat",  # "data/measured_xas.txt"
        experimental_xas_shift_eV=760.4,
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
        cfg=cfg,
    )

    # 6. Output Final Results
    save_results(out, cfg)
    plot_final_results(out, cfg)
    print("Final .cgr files, .csq files, and comparison plot saved.")


if __name__ == "__main__":
    main()
