"""Bolme protokolleri (merdivenin kademeleri) ve sizinti kontrolleri.

P1 random_split(ortusmeli)    : literaturun yaygin pratigi
P2 random_split(ortusmesiz)   : literal ornek ortusmesi yok
P3 temporal_block_split       : kayit ici zaman ayrimi
P4 leave_one_cond_out         : calisma kosulu ayrimi
P5 leave_one_specimen_out     : numune ayrimi (en zorlu; numune bilgisi varsa)
"""
from __future__ import annotations

import numpy as np

from .data import WindowSet


def random_split(ws: WindowSet, seed: int, test_frac: float = 0.30):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(ws.y):
        idx = np.flatnonzero(ws.y == c)
        rng.shuffle(idx)
        k = int(round(len(idx) * test_frac))
        te.append(idx[:k])
        tr.append(idx[k:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


def temporal_block_split(ws: WindowSet, test_frac: float = 0.30, guard: int = 1):
    """Her kaydin ilk (1-test_frac) kismi egitim, son kismi test.
    guard: sinirda birakilan pencere sayisi."""
    tr, te = [], []
    for r in np.unique(ws.rec_id):
        idx = np.flatnonzero(ws.rec_id == r)
        order = idx[np.argsort(ws.start[idx])]
        cut = int(round(len(order) * (1 - test_frac)))
        tr.append(order[:max(cut - guard, 0)])
        te.append(order[cut:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


def leave_one_cond_out(ws: WindowSet, test_cond):
    tr = np.flatnonzero(ws.cond != test_cond)
    te = np.flatnonzero(ws.cond == test_cond)
    return tr, te


def leave_one_specimen_out(ws: WindowSet, test_specimen):
    tr = np.flatnonzero(ws.specimen != test_specimen)
    te = np.flatnonzero(ws.specimen == test_specimen)
    return tr, te


def gecerli_kat(ws: WindowSet, tr, te, min_sinif=2) -> bool:
    """Katin kullanilabilir olmasi icin: egitimde ve testte yeterli sinif,
    ve testteki her sinif egitimde de bulunmali."""
    if len(tr) == 0 or len(te) == 0:
        return False
    s_tr, s_te = set(ws.y[tr].tolist()), set(ws.y[te].tolist())
    return len(s_tr) >= min_sinif and len(s_te) >= min_sinif and s_te <= s_tr


# ------------------------------------------------------------- kontroller
def overlap_fraction(ws: WindowSet, tr, te, window: int = 2048) -> float:
    """Ayni kayitta bir egitim penceresiyle ornek duzeyinde kesisen test
    pencerelerinin orani. P1'de >0, P2-P5'te tam 0 olmalidir."""
    tr, te = np.asarray(tr), np.asarray(te)
    hit = 0
    for r in np.unique(ws.rec_id[te]):
        s_tr = np.sort(ws.start[tr[ws.rec_id[tr] == r]])
        s_te = ws.start[te[ws.rec_id[te] == r]]
        if len(s_tr) == 0:
            continue
        pos = np.searchsorted(s_tr, s_te)
        for i, p in enumerate(pos):
            cands = s_tr[max(p - 1, 0):min(p + 1, len(s_tr))]
            if len(cands) and np.any(np.abs(cands - s_te[i]) < window):
                hit += 1
    return hit / max(len(te), 1)


def shared_recording_fraction(ws: WindowSet, tr, te) -> float:
    tr_set = set(np.unique(ws.rec_id[np.asarray(tr)]).tolist())
    return float(np.mean([r in tr_set for r in ws.rec_id[np.asarray(te)]]))


def shared_specimen_fraction(ws: WindowSet, tr, te) -> float:
    tr_set = set(np.unique(ws.specimen[np.asarray(tr)]).tolist())
    return float(np.mean([s in tr_set for s in ws.specimen[np.asarray(te)]]))
