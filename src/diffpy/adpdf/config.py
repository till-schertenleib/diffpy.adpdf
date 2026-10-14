"""Configuration dataclasses for the AD-PDF pipeline."""

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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

    # Output directory (figures go to out_dir/figures)
    out_dir: str = "adpdf_out"

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

    @property
    def fig_dir(self) -> str:
        return os.path.join(self.out_dir, "figures")

    @property
    def frame_dir(self) -> str:
        return os.path.join(self.out_dir, "figures", "refine_frames")


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


@dataclass
class DataConfig:
    path: str
    energies: Optional[str] = None
    background: Optional[str] = None
    background_scale: float = 1.0


@dataclass
class SampleConfig:
    elements: List[str]
    composition: Dict[str, float]
    resonant: str


@dataclass
class PipelineConfig:
    data: DataConfig
    sample: SampleConfig
    edge: EdgeModelParams
    refinement: RefinementConfig = field(default_factory=RefinementConfig)


def _build(cls, section: Optional[Dict[str, Any]], name: str):
    if section is None:
        raise ValueError(f"Missing required [{name}] section.")
    allowed = {f.name for f in fields(cls)}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) in [{name}]: {', '.join(sorted(unknown))}"
        )
    try:
        return cls(**section)
    except TypeError as e:
        raise ValueError(f"Invalid [{name}] section: {e}") from None


def _resolve(base: Path, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = Path(p).expanduser()
    return str(p if p.is_absolute() else base / p)


def load_config(path: Union[str, Path]) -> PipelineConfig:
    """Read a TOML run configuration.

    Relative paths in the file are resolved against the folder that
    contains the TOML file.
    """
    path = Path(path)
    base = path.resolve().parent
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    unknown = set(raw) - {"data", "sample", "edge", "refinement", "output"}
    if unknown:
        raise ValueError(f"Unknown section(s): {', '.join(sorted(unknown))}")

    data = _build(DataConfig, raw.get("data"), "data")
    data.path = _resolve(base, data.path)
    data.energies = _resolve(base, data.energies)
    data.background = _resolve(base, data.background)

    sample = _build(SampleConfig, raw.get("sample"), "sample")
    if set(sample.composition) != set(sample.elements):
        raise ValueError(
            "[sample] composition keys must match elements: "
            f"{sorted(sample.composition)} vs {sorted(sample.elements)}"
        )
    if sample.resonant not in sample.elements:
        raise ValueError(
            f"[sample] resonant element '{sample.resonant}' "
            "is not in elements."
        )

    edge_raw = dict(raw.get("edge", {}))
    edge_raw.setdefault("element", sample.resonant)
    peaks = edge_raw.pop("peaks", [])
    edge = _build(EdgeModelParams, edge_raw, "edge")
    edge.peaks = [_build(Peak, pk, "edge.peaks") for pk in peaks]
    if edge.element != sample.resonant:
        raise ValueError(
            f"[edge] element '{edge.element}' differs from [sample] "
            f"resonant element '{sample.resonant}'."
        )
    edge.experimental_xas_file = _resolve(base, edge.experimental_xas_file)

    refinement = _build(
        RefinementConfig, raw.get("refinement", {}), "refinement"
    )
    output = raw.get("output", {})
    unknown = set(output) - {"out_dir"}
    if unknown:
        raise ValueError(
            f"Unknown key(s) in [output]: {', '.join(sorted(unknown))}"
        )
    refinement.out_dir = _resolve(
        base, output.get("out_dir", refinement.out_dir)
    )

    return PipelineConfig(data, sample, edge, refinement)
