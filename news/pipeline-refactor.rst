**Added:**

* ``adpdf <config.toml>`` command-line entry point and TOML run configuration (``load_config``), with Sn- and Bi-edge examples in ``examples/``.
* Refined parameter values are written to ``refined_params.txt`` and every refined parameter is recorded in the refinement history.

**Changed:**

* Split ``pipeline.py`` into ``config``, ``utils``, ``edge``, ``scattering``, ``reconstruction``, ``refinement``, ``fileio`` and ``plotting`` modules.
* Output figures are written under ``out_dir/figures``, derived from ``out_dir``.
* A missing background or experimental XAS file is now an error instead of being skipped silently.
* Requirements now list the packages actually used: numpy>=2, matplotlib, h5py, kkcalc>=0.8.4, xraydb.

**Deprecated:**

* <news item>

**Removed:**

* Pipeline output files and sample data are no longer part of the installed package; sample data and notebooks moved to ``examples/``.

**Fixed:**

* A rejected refinement step on a clipped parameter (e.g. ``rpoly`` at its bound) did not restore the previous value.
* ``reconstruct_partials_and_pdfs`` defaulted to ``bkg_poly_type="getx"`` while ``RefinementConfig`` defaulted to ``"wls"``.

**Security:**

* <news item>
