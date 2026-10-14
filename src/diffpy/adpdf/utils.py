"""Small numerical helpers shared across the pipeline."""

import numpy as np


def as_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).ravel()


def trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def safe(a: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(a)):
        raise ValueError(
            f"{name} has non-finite entries at {np.where(~np.isfinite(a))}"
        )
    return a


def ensure_2d(x: np.ndarray, ne: int, nq: int, name: str) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim == 0:
        return np.full((ne, nq), float(x))
    if x.ndim == 1:
        if x.size == ne:
            return x[:, None] * np.ones((1, nq))
        if x.size == nq:
            return np.ones((ne, 1)) * x[None, :]
        raise ValueError(f"{name} 1D mismatch")
    if x.shape != (ne, nq):
        raise ValueError(f"{name} shape mismatch")
    return x


def lorch(Q: np.ndarray, Qmax: float) -> np.ndarray:
    x = Q / Qmax
    w = np.ones_like(Q)
    m = (x > 0) & (x <= 1.0)
    w[m] = np.sin(np.pi * x[m]) / (np.pi * x[m])
    w[x > 1.0] = 0.0
    return w


def kaiser(Q: np.ndarray, Qmax: float, beta: float = 6.0) -> np.ndarray:
    from numpy import i0

    x = Q / Qmax
    w = np.zeros_like(Q)
    m = (x >= 0) & (x <= 1.0)
    w[m] = i0(beta * np.sqrt(1.0 - x[m] ** 2)) / i0(beta)
    return w
