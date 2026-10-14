"""Unit tests for TOML run configuration loading."""

from pathlib import Path

import pytest

from diffpy.adpdf.config import RefinementConfig, load_config

VALID = """
[data]
path = "patterns/"
energies = "energies.txt"

[sample]
elements = ["Ba", "Bi", "O"]
composition = { Ba = 0.2, Bi = 0.2, O = 0.6 }
resonant = "Bi"

[edge]
scale = 0.35
peaks = [{ center_eV = 90706.0, fwhm_eV = 3.0, height = 0.3 }]

[refinement]
ridge = 8e-3
n_outer = 3

[output]
out_dir = "results"
"""


def write(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_load_valid_config(tmp_path):
    pc = load_config(write(tmp_path, VALID))
    base = tmp_path.resolve()
    assert Path(pc.data.path) == base / "patterns"
    assert Path(pc.data.energies) == base / "energies.txt"
    assert pc.data.background is None
    assert pc.sample.resonant == "Bi"
    # edge element defaults to the resonant element
    assert pc.edge.element == "Bi"
    assert pc.edge.scale == 0.35
    assert len(pc.edge.peaks) == 1
    assert pc.edge.peaks[0].center_eV == 90706.0
    assert pc.refinement.ridge == 8e-3
    assert pc.refinement.n_outer == 3
    # untouched fields keep their defaults
    assert pc.refinement.rpoly == RefinementConfig().rpoly
    assert Path(pc.refinement.out_dir) == base / "results"


def test_absolute_paths_are_kept(tmp_path):
    abs_path = (tmp_path / "elsewhere" / "data.h5").resolve()
    text = VALID.replace('"patterns/"', f'"{abs_path.as_posix()}"')
    pc = load_config(write(tmp_path, text))
    assert Path(pc.data.path) == abs_path


def test_fig_dirs_follow_out_dir():
    cfg = RefinementConfig(out_dir="some/out")
    assert Path(cfg.fig_dir) == Path("some/out/figures")
    assert Path(cfg.frame_dir) == Path("some/out/figures/refine_frames")


@pytest.mark.parametrize(
    "old, new, match",
    [
        (
            "ridge = 8e-3",
            "rigde = 8e-3",
            r"Unknown key\(s\) in \[refinement\]",
        ),
        ("[output]", "[outptu]", r"Unknown section"),
        ("O = 0.6 }", "O = 0.6, Sr = 0.1 }", "composition keys"),
        ('resonant = "Bi"', 'resonant = "Sn"', "not in elements"),
        ("scale = 0.35", 'element = "Ba"', "differs from"),
        ("fwhm_eV = 3.0, ", "", r"Invalid \[edge.peaks\]"),
    ],
)
def test_invalid_config(tmp_path, old, new, match):
    assert old in VALID
    with pytest.raises(ValueError, match=match):
        load_config(write(tmp_path, VALID.replace(old, new)))


def test_missing_required_section(tmp_path):
    text = VALID.split("[sample]")[0]
    with pytest.raises(ValueError, match=r"Missing required \[sample\]"):
        load_config(write(tmp_path, text))
