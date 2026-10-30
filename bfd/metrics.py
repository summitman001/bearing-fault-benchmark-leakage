"""Siralama ve kalibrasyon olcutleri.

Kalibrasyon olcutleri makalenin asil katkisi oldugu icin burada acik
formullerle yazilmistir; ECE kutu sayisi config.N_ECE_BINS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

EPS = 1e-12


def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def brier(probs, y, n_classes=None):
    n_classes = n_classes or probs.shape[1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def ece(probs, y, n_bins=15):
    """Beklenen kalibrasyon hatasi (esit genislikli kutular, maks guven)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability_table(probs, y, n_bins=15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append(dict(kutu=k, kutu_alt=edges[k], kutu_ust=edges[k + 1],
                         n=int(m.sum()),
                         ort_guven=float(conf[m].mean()) if m.any() else np.nan,
                         gozlenen_dogruluk=float(correct[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def macro_auroc(probs, y, n_classes):
    """Tum siniflar testte yoksa np.nan doner."""
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    try:
        if len(present) == n_classes:
            return float(roc_auc_score(y, probs, multi_class="ovr",
                                       average="macro", labels=np.arange(n_classes)))
        sub = probs[:, present]
        s = sub.sum(1, keepdims=True)
        sub = np.divide(sub, s, out=np.full_like(sub, 1.0 / len(present)), where=s > 0)
        remap = {c: i for i, c in enumerate(present)}
        return float(roc_auc_score([remap[v] for v in y], sub,
                                   multi_class="ovr", average="macro"))
    except Exception:  # noqa: BLE001
        return float("nan")


def all_metrics(probs, y, n_classes, n_bins=15) -> dict:
    pred = probs.argmax(1)
    conf = probs.max(1)
    acc = float(accuracy_score(y, pred))
    return dict(
        dogruluk=acc,
        makro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        auroc=macro_auroc(probs, y, n_classes),
        nll=nll(probs, y),
        brier=brier(probs, y, n_classes),
        ece=ece(probs, y, n_bins),
        ort_guven=float(conf.mean()),
        guven_eksi_dogruluk=float(conf.mean() - acc),
        n_test=int(len(y)),
    )
