"""Unit tests for __version__.py."""

import diffpy.adpdf


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(diffpy.adpdf, "__version__")
    assert diffpy.adpdf.__version__ != "0.0.0"
