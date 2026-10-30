"""Boru hattinin dogrulugunu sinayan testler. `python -m pytest tests -q`
veya dogrudan `python tests/test_pipeline.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfd import calib, metrics, splits, synth  # noqa: E402
from bfd import config as C  # noqa: E402


def test_bolme_sizintisi():
    """P2/P3/P4 ornek duzeyinde ortusme icermemeli; P4 kayit paylasmamali."""
    ws = synth.make_windowset(n_per_rec=20, hop=C.WINDOW, seed=0, n_specimens=2)
    tr, te = splits.random_split(ws, 0)
    assert splits.overlap_fraction(ws, tr, te, C.WINDOW) == 0.0
    assert splits.shared_recording_fraction(ws, tr, te) == 1.0

    tr, te = splits.temporal_block_split(ws, 0.3)
    assert splits.overlap_fraction(ws, tr, te, C.WINDOW) == 0.0
    assert len(set(tr) & set(te)) == 0

    tr, te = splits.leave_one_cond_out(ws, "3hp")
    assert splits.shared_recording_fraction(ws, tr, te) == 0.0
    assert set(ws.cond[te]) == {"3hp"} and "3hp" not in set(ws.cond[tr])
    assert splits.gecerli_kat(ws, tr, te)

    tr, te = splits.leave_one_specimen_out(ws, "N2")
    assert splits.shared_specimen_fraction(ws, tr, te) == 0.0
    assert set(ws.specimen[te]) == {"N2"}


def test_ortusme_tespiti():
    """Ortusmeli pencerelerde rastgele bolme ortusme URETMELI."""
    ws = synth.make_windowset(n_per_rec=40, hop=C.WINDOW // 4, seed=0)
    tr, te = splits.random_split(ws, 0)
    assert splits.overlap_fraction(ws, tr, te, C.WINDOW) > 0.5


def test_ece_mukemmel_kalibrasyon():
    """Olasiliklar gercek uretici ise ECE kucuk olmali."""
    rng = np.random.default_rng(0)
    n, k = 20000, 4
    p = rng.dirichlet(np.ones(k) * 0.7, size=n)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    assert metrics.ece(p, y, 15) < 0.03


def test_ece_asiri_guven():
    """Keskinlestirilmis olasiliklar ECE'yi buyutmeli."""
    rng = np.random.default_rng(0)
    n, k = 20000, 4
    p = rng.dirichlet(np.ones(k) * 0.7, size=n)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    q = p ** 4
    q /= q.sum(1, keepdims=True)
    assert metrics.ece(q, y, 15) > metrics.ece(p, y, 15) + 0.1


def test_sicaklik_asiri_guveni_duzeltir():
    rng = np.random.default_rng(1)
    n, k = 8000, 5
    p = rng.dirichlet(np.ones(k) * 0.6, size=n)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    q = p ** 5
    q /= q.sum(1, keepdims=True)
    logits = calib.probs_to_logits(q)
    T = calib.fit_temperature(logits, y)
    assert T > 1.2, f"asiri guvende T>1 beklenir, T={T}"
    p_cal = calib.apply_temperature(logits, T)
    assert metrics.ece(p_cal, y, 15) < metrics.ece(q, y, 15)
    assert metrics.nll(p_cal, y) < metrics.nll(q, y)


def test_sicaklik_degerlendirmeden_cikarir():
    """T'yi kestiren ornekler degerlendirmeye girmemeli."""
    rng = np.random.default_rng(2)
    n, k = 2000, 10
    p = rng.dirichlet(np.ones(k), size=n)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    rows = calib.temperature_curve(p, y, [50], 3, 0, k,
                                   lambda pp, yy: metrics.all_metrics(pp, yy, k))
    for r in rows:
        assert r["n_kalibrasyon"] + r["n_degerlendirme"] == n


def test_metrikler_tutarli():
    rng = np.random.default_rng(3)
    n, k = 500, 10
    p = rng.dirichlet(np.ones(k), size=n)
    y = rng.integers(0, k, n)
    m = metrics.all_metrics(p, y, k)
    assert 0 <= m["dogruluk"] <= 1 and 0 <= m["ece"] <= 1
    assert m["nll"] > 0 and 0 <= m["brier"] <= 2
    assert abs(m["guven_eksi_dogruluk"] - (m["ort_guven"] - m["dogruluk"])) < 1e-9


if __name__ == "__main__":
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_")]
    for f in fns:
        f()
        print(f"  OK  {f.__name__}")
    print(f"\n{len(fns)} test gecti.")
