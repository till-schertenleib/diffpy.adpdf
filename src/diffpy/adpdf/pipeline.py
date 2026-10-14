"""Run the AD-PDF pipeline end to end.

Usage::

    adpdf path/to/config.toml

See ``examples/*.toml`` for the configuration format.
"""

import argparse
import sys

from diffpy.adpdf.config import PipelineConfig, load_config
from diffpy.adpdf.edge import EdgeModel
from diffpy.adpdf.fileio import (
    load_background,
    load_experimental_data,
    save_refined_params,
    save_results,
    setup_directories,
)
from diffpy.adpdf.plotting import plot_final_results, save_initial_edge_plots
from diffpy.adpdf.refinement import run_adpdf_with_refinement


def run_pipeline(pc: PipelineConfig):
    """Load data, refine, and write G(r)/S(Q) files and figures.

    Returns ``(out, diag, hist)`` from the refinement.
    """
    cfg = pc.refinement
    setup_directories(cfg)

    print(f"Loading data from '{pc.data.path}'...")
    I_raw, Q, E = load_experimental_data(pc.data.path, pc.data.energies)

    if pc.data.background is not None:
        scale = pc.data.background_scale
        print(
            f"Subtracting background from '{pc.data.background}' "
            f"(scale={scale})..."
        )
        I_bkg = load_background(pc.data.background, Q)
        # Subtract the scaled background from every energy row
        I_raw = I_raw - (scale * I_bkg[None, :])

    edge_model = EdgeModel(pc.edge)
    save_initial_edge_plots(E, edge_model, cfg)

    print("Starting refinement...")
    out, diag, hist = run_adpdf_with_refinement(
        I_raw=I_raw,
        Q=Q,
        E_eV=E,
        elements=pc.sample.elements,
        comp=pc.sample.composition,
        alpha=pc.sample.resonant,
        edge_model=edge_model,
        cfg=cfg,
    )

    save_results(out, cfg)
    save_refined_params(hist[-1].params, hist[-1].obj, cfg)
    plot_final_results(out, cfg)
    print(f"Results written to '{cfg.out_dir}'.")
    return out, diag, hist


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="adpdf",
        description="Separate resonant and non-resonant partial PDFs "
        "from energy-dependent X-ray scattering data.",
    )
    parser.add_argument("config", help="path to a TOML run configuration")
    args = parser.parse_args(argv)

    try:
        pc = load_config(args.config)
    except (OSError, ValueError) as e:
        parser.exit(1, f"adpdf: config error: {e}\n")

    try:
        run_pipeline(pc)
    except FileNotFoundError as e:
        print(f"adpdf: error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
