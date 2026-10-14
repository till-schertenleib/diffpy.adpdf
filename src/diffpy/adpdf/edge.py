"""Anomalous scattering factor (f', f'') model near an absorption edge."""

import os
from typing import Optional, Tuple

import kkcalc
import numpy as np
import xraydb

from diffpy.adpdf.config import EdgeModelParams


class _BaseEdgeModel:
    def __init__(self, params: EdgeModelParams):
        self.params = params

    @staticmethod
    def to_eV(E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=float)
        scale = 1.0 if np.nanmax(E) >= 1e3 else 1e3
        return E * scale

    @staticmethod
    def lorentzian_height(
        E: np.ndarray, center: float, fwhm: float, height: float
    ) -> np.ndarray:
        gamma = 0.5 * max(float(fwhm), 1e-18)
        return height * (gamma**2) / ((E - center) ** 2 + gamma**2)

    @staticmethod
    def broadening(
        E_eV: np.ndarray, y: np.ndarray, fwhm_eV: float
    ) -> np.ndarray:
        if fwhm_eV <= 0.0:
            return y
        Emin, Emax = float(E_eV[0]), float(E_eV[-1])
        dmins = float(np.min(np.diff(E_eV)))
        d_helper = max(min(dmins / 4.0, fwhm_eV / 25.0), 1e-3)
        pad = max(5.0 * fwhm_eV, 10.0 * dmins)
        Xu = np.arange(Emin - pad, Emax + pad + d_helper, d_helper)
        yu = np.interp(Xu, E_eV, y)

        gamma = 0.5 * float(fwhm_eV)
        Lx = np.arange(-10.0 * fwhm_eV, 10.0 * fwhm_eV + d_helper, d_helper)
        Lk = (gamma / np.pi) / (Lx**2 + gamma**2)
        Lk /= Lk.sum()
        conv = np.convolve(yu, Lk, mode="same")
        return np.interp(E_eV, Xu, conv)


class KKCalcAdapter:
    def __init__(self, element: str):
        self.stoich = kkcalc.data.ParseChemicalFormula(element)
        self.rc = kkcalc.calc_relativistic_correction(self.stoich)
        self.full_E, self.full_imag_coeffs = kkcalc.data.calculate_asf(
            self.stoich
        )

    def f1_from_f2(
        self,
        energy_eval_eV: np.ndarray,
        near_edge_f2_vals: Optional[np.ndarray],
        merge_points: Optional[Tuple[float, float]] = None,
        add_background: bool = False,
        fix_distortions: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        E = np.asarray(energy_eval_eV, float)
        if near_edge_f2_vals is not None:
            NearEdge_Data = np.column_stack(
                [E, np.asarray(near_edge_f2_vals, float)]
            )
            mp = (
                (float(E[0]), float(E[-1]))
                if merge_points is None
                else merge_points
            )
            Full_E, Imaginary_Spectrum = kkcalc.data.merge_spectra(
                NearEdge_Data,
                self.full_E,
                self.full_imag_coeffs,
                merge_points=mp,
                add_background=add_background,
                fix_distortions=fix_distortions,
            )
        else:
            Full_E, Imaginary_Spectrum = self.full_E, self.full_imag_coeffs

        f1_vals = kkcalc.KK_PP(E, Full_E, Imaginary_Spectrum, self.rc)
        i1 = max(np.searchsorted(Full_E, E[0], side="right") - 1, 0)
        i2 = max(np.searchsorted(Full_E, E[-1], side="right") - 1, i1 + 1)
        f2_vals = kkcalc.data.coeffs_to_ASF(
            E, np.vstack((Imaginary_Spectrum[i1:i2], Imaginary_Spectrum[-1]))
        )
        return f1_vals, f2_vals, float(self.rc)


class EdgeModel(_BaseEdgeModel):
    def __init__(self, params: EdgeModelParams):
        super().__init__(params)
        self.kk = KKCalcAdapter(params.element)
        self.exp_energy = None
        self.exp_fpp = None

        # Properly placed loading logic
        if self.params.experimental_xas_file:
            if os.path.isfile(self.params.experimental_xas_file):
                try:
                    data = np.loadtxt(self.params.experimental_xas_file)
                    raw_E = data[:, 0]
                    if np.nanmax(raw_E) < 1e3:
                        raw_E *= 1e3
                    raw_E += float(self.params.experimental_xas_shift_eV)

                    sort_idx = np.argsort(raw_E)
                    self.exp_energy = raw_E[sort_idx]
                    self.exp_fpp = data[sort_idx, 1]
                    print(
                        f"Loaded experimental XAS from {self.params.experimental_xas_file}"
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to load XAS file '{self.params.experimental_xas_file}': {e}"
                    )
            else:
                print(
                    f"Warning: XAS file '{self.params.experimental_xas_file}' not found. Falling back to Chantler."
                )

    def fpp(self, E: np.ndarray) -> np.ndarray:
        E_eV = self.to_eV(E)
        E_eval = E_eV + float(self.params.dE_eV)

        # Check if experimental XAS was successfully loaded
        if (
            getattr(self, "exp_energy", None) is not None
            and getattr(self, "exp_fpp", None) is not None
        ):
            f2 = np.interp(E_eval, self.exp_energy, self.exp_fpp)
            return float(self.params.scale) * f2

        # Fallback to Chantler baseline
        if self.params.use_chantler_f2_baseline:
            f2 = np.asarray(
                xraydb.f2_chantler(self.params.element, E_eval), float
            )
        else:
            f2 = np.zeros_like(E_eval)

        f2 = float(self.params.scale) * f2 + float(self.params.fpp_offset)
        f2 += float(self.params.fpp_slope_per_eV) * (
            E_eV - float(self.params.e0_keV) * 1e3
        )

        if self.params.peaks:
            add = np.zeros_like(f2)
            for pk in self.params.peaks:
                add += self.lorentzian_height(
                    E_eval, pk.center_eV, pk.fwhm_eV, pk.height
                )
            f2 = f2 + add

        return self.broadening(E_eV, f2, self.params.broaden_fwhm_eV)

    def fp(
        self,
        E: np.ndarray,
        fpp: Optional[np.ndarray] = None,
        return_f1: bool = False,
        merge_points: Optional[Tuple[float, float]] = None,
        add_background: bool = False,
        fix_distortions: bool = False,
    ) -> np.ndarray:
        E_eV = self.to_eV(E)
        E_eval = E_eV + float(self.params.dE_eV)
        if fpp is None:
            fpp = self.fpp(E_eV)
        f1_vals, f2_vals, f1_inf = self.kk.f1_from_f2(
            energy_eval_eV=E_eval,
            near_edge_f2_vals=fpp,
            merge_points=merge_points,
            add_background=add_background,
            fix_distortions=fix_distortions,
        )
        return f1_vals if return_f1 else (f1_vals - f1_inf)
