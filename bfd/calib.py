"""Sicaklik olcekleme.

Sinir aglari icin sicaklik logit uzerinde uygulanir. Agac tabanli modellerde
logit yoktur; standart pratik log-olasiligi vekil logit olarak kullanmaktir.
Bu tercih makalede acikca belirtilmelidir.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-12


def probs_to_logits(probs):
    return np.log(np.clip(probs, EPS, 1.0))


def apply_temperature(logits, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y, bounds=(0.05, 20.0)) -> float:
    """NLL'yi en kucuk yapan T. Tek degiskenli, kapali aralikta arama."""
    def obj(logT):
        p = apply_temperature(logits, float(np.exp(logT)))
        return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean()

    res = minimize_scalar(obj, bounds=(np.log(bounds[0]), np.log(bounds[1])),
                          method="bounded", options=dict(xatol=1e-4))
    return float(np.exp(res.x))


def temperature_curve(probs_target, y_target, k_values, n_draws, seed,
                      n_classes, metric_fn):
    """Hedef kosuldan k etiketli ornekle T kestirir, KALAN orneklerde olcer.

    Onemli: T'yi kestirmekte kullanilan k ornek degerlendirmeden cikarilir.
    Dondurur: liste[dict(k, cekilis, sicaklik, <metrikler>)]
    """
    rng = np.random.default_rng(seed)
    logits = probs_to_logits(probs_target)
    rows = []
    for k in k_values:
        for d in range(n_draws):
            idx = np.arange(len(y_target))
            # siniflar arasinda katmanli secim
            pick = []
            per = max(k // n_classes, 1)
            for c in np.unique(y_target):
                ci = idx[y_target == c]
                if len(ci) == 0:
                    continue
                pick.append(rng.choice(ci, size=min(per, len(ci)), replace=False))
            pick = np.unique(np.concatenate(pick))[:k] if pick else np.array([], int)
            if len(pick) < 2:
                continue
            rest = np.setdiff1d(idx, pick)
            if len(rest) == 0:
                continue
            T = fit_temperature(logits[pick], y_target[pick])
            p_cal = apply_temperature(logits[rest], T)
            row = dict(k=int(k), cekilis=int(d), sicaklik=float(T),
                       n_kalibrasyon=int(len(pick)), n_degerlendirme=int(len(rest)))
            row.update(metric_fn(p_cal, y_target[rest]))
            rows.append(row)
    return rows
