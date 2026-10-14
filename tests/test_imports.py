"""Check that the public API is importable from the package."""

import diffpy.adpdf


def test_public_api():
    for name in diffpy.adpdf.__all__:
        assert hasattr(diffpy.adpdf, name), name
    assert callable(diffpy.adpdf.run_pipeline)
