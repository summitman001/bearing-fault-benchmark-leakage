"""Zaman ve frekans oznitelikleri. Tum oznitelikler pencere basina hesaplanir;
hicbiri egitim/test bolmesine bagli degildir (olceklemeyi model yapar)."""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

EPS = 1e-12


def time_features(X: np.ndarray) -> np.ndarray:
    """(n, L) -> (n, 8): RMS, basiklik, carpiklik, tepe, tepe faktoru,
    sekil faktoru, darbe faktoru, marj faktoru."""
    x = np.asarray(X, dtype=np.float64)
    mu = x.mean(1, keepdims=True)
    xc = x - mu
    rms = np.sqrt((xc ** 2).mean(1)) + EPS
    sd = xc.std(1) + EPS
    kurt = ((xc / sd[:, None]) ** 4).mean(1)
    skew = ((xc / sd[:, None]) ** 3).mean(1)
    peak = np.abs(xc).max(1)
    mean_abs = np.abs(xc).mean(1) + EPS
    sra = (np.sqrt(np.abs(xc)).mean(1)) ** 2 + EPS
    return np.column_stack([rms, kurt, skew, peak, peak / rms,
                            rms / mean_abs, peak / mean_abs, peak / sra])


TIME_FEATURE_NAMES = ["rms", "kurtosis", "skewness", "peak", "crest",
                      "shape", "impulse", "margin"]


def trivial_features(X: np.ndarray) -> np.ndarray:
    """Onemsiz taban cizgisi: yalnizca RMS ve basiklik."""
    f = time_features(X)
    return f[:, [0, 1]]


def band_energies(X: np.ndarray, n_bands: int = 16) -> np.ndarray:
    """Hizli Fourier buyuklugunun esit genislikli bant enerjileri (log)."""
    mag = np.abs(np.fft.rfft(np.asarray(X, dtype=np.float64), axis=1))
    edges = np.linspace(0, mag.shape[1], n_bands + 1).astype(int)
    out = np.empty((mag.shape[0], n_bands))
    for b in range(n_bands):
        out[:, b] = (mag[:, edges[b]:edges[b + 1]] ** 2).sum(1)
    return np.log(out + EPS)


def envelope_spectrum(X: np.ndarray, n_out: int = 64) -> np.ndarray:
    """Zarf spektrumu (Hilbert), n_out bantta toplanmis ve normalize."""
    x = np.asarray(X, dtype=np.float64)
    env = np.abs(hilbert(x, axis=1))
    env = env - env.mean(1, keepdims=True)
    mag = np.abs(np.fft.rfft(env, axis=1))
    edges = np.linspace(0, mag.shape[1], n_out + 1).astype(int)
    out = np.empty((mag.shape[0], n_out))
    for b in range(n_out):
        out[:, b] = mag[:, edges[b]:edges[b + 1]].mean(1)
    return out / (out.sum(1, keepdims=True) + EPS)


def full_features(X: np.ndarray) -> np.ndarray:
    """M2 (rastgele orman) icin: zaman + bant enerjileri + zarf spektrumu."""
    return np.hstack([time_features(X), band_energies(X), envelope_spectrum(X, 32)])
