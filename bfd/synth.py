"""Sentetik veri: boru hattini GERCEK VERI OLMADAN dogrulamak icindir.
Hicbir sayisi makaleye girmez. Bilerek kolaydir.

Kayda ozgu sabit bir ton ('parmak izi') eklenir; rastgele bolmenin neden
yanlis yuksek sonuc verdigini sentetik olarak da gosterebilmek icin.
"""
from __future__ import annotations

import numpy as np

from .data import WindowSet

SINIFLAR = ["Normal", "IR007", "B007", "OR007", "IR014", "B014",
            "OR014", "IR021", "B021", "OR021"]
KOSULLAR = ["0hp", "1hp", "2hp", "3hp"]
NUMUNELER = ["N1", "N2"]


def _kayit(ci, kosul_i, numune_i, n, rng, fs=12000.0, fingerprint=True):
    t = np.arange(n) / fs
    f_fault = 80.0 + 25.0 * ci + 4.0 * numune_i
    rpm_f = 29.0 - 0.6 * kosul_i
    sig = 0.05 * rng.standard_normal(n)
    sig += (0.3 + 0.05 * kosul_i) * np.sin(2 * np.pi * rpm_f * t)
    if ci > 0:
        imp = np.zeros(n)
        period = max(int(fs / f_fault), 2)
        imp[np.arange(0, n, period)] = 1.0
        decay = np.exp(-np.arange(period) / (period / 6))
        carrier = np.sin(2 * np.pi * (2500 + 300 * ci) * np.arange(period) / fs)
        sig += (0.4 + 0.08 * kosul_i) * np.convolve(imp, decay * carrier)[:n]
    if fingerprint:
        key = (ci * 10 + kosul_i) * 3 + numune_i
        sig += 0.12 * np.sin(2 * np.pi * (411.0 + 7.3 * key) * t + 0.7)
    return sig


def make_windowset(n_per_rec=40, window=2048, hop=None, seed=0,
                   n_classes=None, n_specimens=1, fingerprint=True) -> WindowSet:
    hop = hop or window
    siniflar = SINIFLAR[:(n_classes or len(SINIFLAR))]
    rng = np.random.default_rng(seed)
    n_samples = window + hop * (n_per_rec - 1)
    Xs, ys, conds, specs, recs, starts, fracs = [], [], [], [], [], [], []
    rec = 0
    for ci, _ in enumerate(siniflar):
        for ki, kosul in enumerate(KOSULLAR):
            for ni in range(n_specimens):
                sig = _kayit(ci, ki, ni, n_samples, rng, fingerprint=fingerprint)
                st = np.arange(n_per_rec) * hop
                Xs.append(np.stack([sig[s:s + window] for s in st]))
                ys.append(np.full(n_per_rec, ci))
                conds.append(np.full(n_per_rec, kosul, dtype=object))
                specs.append(np.full(n_per_rec, NUMUNELER[ni % len(NUMUNELER)], dtype=object))
                recs.append(np.full(n_per_rec, rec))
                starts.append(st)
                fracs.append(st / max(n_samples - window, 1))
                rec += 1
    return WindowSet(np.concatenate(Xs), np.concatenate(ys), np.concatenate(conds),
                     np.concatenate(specs), np.concatenate(recs),
                     np.concatenate(starts), np.concatenate(fracs), siniflar)
