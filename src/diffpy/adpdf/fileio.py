"""Reading input data and writing pipeline results."""

import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from diffpy.adpdf.config import RefinementConfig


def load_experimental_data(
    data_path: str, energies_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.isfile(data_path) and data_path.lower().endswith(
        (".h5", ".hdf5")
    ):
        with h5py.File(data_path, "r") as f:
            data = f["data"]
            I_raw = np.array(data["iq"])
            E_raw = np.array(data["energy"])
            Q = np.array(data["q"])
        E = E_raw * 1e3 if np.nanmax(E_raw) < 1e3 else E_raw
        return I_raw, Q, E

    elif os.path.isdir(data_path):
        if energies_path is None or not os.path.isfile(energies_path):
            raise FileNotFoundError(
                f"Could not find an energies text file at '{energies_path}'. "
                "When loading a directory, you must provide a 1D text file "
                "of energies."
            )

        E_raw = np.loadtxt(energies_path)
        E = E_raw * 1e3 if np.nanmax(E_raw) < 1e3 else E_raw

        valid_exts = (".iq", ".xy", ".dat", ".xye")
        # NOTE: patterns are paired with energies in plain string order of
        # their filenames, so e.g. "9999.0.xy" sorts after "10000.0.xy".
        files = sorted(
            [
                f
                for f in os.listdir(data_path)
                if f.lower().endswith(valid_exts)
                and os.path.isfile(os.path.join(data_path, f))
            ]
        )

        if not files:
            raise ValueError(f"No diffraction files found in '{data_path}'.")
        if len(files) != len(E):
            raise ValueError(
                f"Found {len(files)} patterns but {len(E)} energies."
            )

        I_list = []
        Q = None
        for f in files:
            d = np.loadtxt(os.path.join(data_path, f))
            if Q is None:
                Q = d[:, 0]
            I_list.append(d[:, 1])

        return np.vstack(I_list), Q, E
    else:
        raise FileNotFoundError(f"Path '{data_path}' is invalid.")


def load_background(bkg_path: str, target_Q: np.ndarray) -> np.ndarray:
    """Loads a 1D background pattern and interpolates it onto the experimental
    Q grid.

    Expects a standard 2-column text file: Q, Intensity.
    """
    if not os.path.isfile(bkg_path):
        raise FileNotFoundError(f"Background file '{bkg_path}' not found.")

    bkg_data = np.loadtxt(bkg_path)
    Q_bkg = bkg_data[:, 0]
    I_bkg = bkg_data[:, 1]

    # Interpolate to match the exact Q grid of the sample scans
    I_bkg_interp = np.interp(target_Q, Q_bkg, I_bkg)
    return I_bkg_interp


def setup_directories(cfg: RefinementConfig):
    for d in [cfg.out_dir, cfg.fig_dir, cfg.frame_dir]:
        if not os.path.exists(d):
            os.makedirs(d)


def save_results(out: Dict[str, np.ndarray], cfg: RefinementConfig):
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

    q = out["Q"]
    np.savetxt(
        os.path.join(cfg.out_dir, "S_alpha.csq"),
        np.column_stack([q, out["S_alpha"]]),
        header="Q S_alpha",
    )
    np.savetxt(
        os.path.join(cfg.out_dir, "S_not.csq"),
        np.column_stack([q, out["S_not"]]),
        header="Q S_not",
    )
    np.savetxt(
        os.path.join(cfg.out_dir, "S_tot.csq"),
        np.column_stack([q, out["S_tot"]]),
        header="Q S_tot",
    )


def save_refined_params(
    params: Dict[str, float], obj: float, cfg: RefinementConfig
):
    """Write the final refined parameters and objective value."""
    path = os.path.join(cfg.out_dir, "refined_params.txt")
    with open(path, "w") as f:
        f.write(f"objective {obj:.10g}\n")
        for name, value in params.items():
            f.write(f"{name} {value:.10g}\n")
