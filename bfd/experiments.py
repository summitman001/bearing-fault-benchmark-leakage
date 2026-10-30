"""Deneyler: bolme merdiveni, mekanizma sondalari, kalibrasyon ve recete."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import calib, config as C, splits
from .data import WindowSet
from .metrics import all_metrics, reliability_table


def _fit_eval(model_cls, ws: WindowSet, tr, te, seed, n_classes=None, **kw):
    n_classes = n_classes or ws.n_classes
    t0 = time.time()
    m = model_cls(n_classes=n_classes, seed=seed, **kw)
    m.fit(ws.X[tr], ws.y[tr])
    p = m.predict_proba(ws.X[te])
    out = all_metrics(p, ws.y[te], n_classes)
    out["sure_s"] = round(time.time() - t0, 2)
    return out, p, m


def _tani(ws, tr, te, window):
    return dict(ortusme_orani=splits.overlap_fraction(ws, tr, te, window),
                paylasilan_kayit_orani=splits.shared_recording_fraction(ws, tr, te),
                paylasilan_numune_orani=splits.shared_specimen_fraction(ws, tr, te),
                n_egitim=len(tr))


# ------------------------------------------------------------- 1) merdiven
def run_ladder(ws_ov: WindowSet, ws_no: WindowSet, models: dict, seeds=None,
               test_frac=None, window=None, dataset="") -> pd.DataFrame:
    seeds = seeds or C.SEEDS
    test_frac = test_frac or C.TEST_FRAC
    window = window or C.WINDOW
    conds = sorted(set(ws_no.cond.tolist()))
    specs = sorted(set(ws_no.specimen.tolist()))
    rows = []
    for name, cls in models.items():
        for seed in seeds:
            for prot, ws in (("P1_rastgele_ortusmeli", ws_ov),
                             ("P2_rastgele_ortusmesiz", ws_no)):
                tr, te = splits.random_split(ws, seed, test_frac)
                met, _, _ = _fit_eval(cls, ws, tr, te, seed)
                rows.append(dict(dataset=dataset, protokol=prot, model=name,
                                 tohum=seed, kat="-", **_tani(ws, tr, te, window), **met))

            tr, te = splits.temporal_block_split(ws_no, test_frac)
            met, _, _ = _fit_eval(cls, ws_no, tr, te, seed)
            rows.append(dict(dataset=dataset, protokol="P3_zamansal_blok", model=name,
                             tohum=seed, kat="-", **_tani(ws_no, tr, te, window), **met))

            for c in conds:
                tr, te = splits.leave_one_cond_out(ws_no, c)
                if not splits.gecerli_kat(ws_no, tr, te):
                    continue
                met, _, _ = _fit_eval(cls, ws_no, tr, te, seed)
                rows.append(dict(dataset=dataset, protokol="P4_kosul_ayrik", model=name,
                                 tohum=seed, kat=f"test_kosul_{c}",
                                 **_tani(ws_no, tr, te, window), **met))

            if len(specs) > 1:
                for s in specs:
                    tr, te = splits.leave_one_specimen_out(ws_no, s)
                    if not splits.gecerli_kat(ws_no, tr, te):
                        continue
                    met, _, _ = _fit_eval(cls, ws_no, tr, te, seed)
                    rows.append(dict(dataset=dataset, protokol="P5_numune_ayrik",
                                     model=name, tohum=seed, kat=f"test_numune_{s}",
                                     **_tani(ws_no, tr, te, window), **met))
            print(f"  [merdiven:{dataset}] {name} tohum={seed} bitti", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 2) sondalar
def run_probes(ws_ov: WindowSet, models: dict, seeds=None, test_frac=None,
               dataset="") -> pd.DataFrame:
    """S1 kayit kimligi sondasi, S2/S2b kayit duzeyi etiket permutasyonu."""
    seeds = seeds or C.SEEDS
    test_frac = test_frac or C.TEST_FRAC
    n_rec = int(ws_ov.rec_id.max()) + 1
    n_cls = ws_ov.n_classes
    conds = sorted(set(ws_ov.cond.tolist()))
    rows = []
    for name, cls in models.items():
        for seed in seeds:
            # S1: pencere hangi kayittan geldi
            ws_rec = WindowSet(ws_ov.X, ws_ov.rec_id.copy(), ws_ov.cond, ws_ov.specimen,
                               ws_ov.rec_id, ws_ov.start, ws_ov.pos_frac,
                               [str(i) for i in range(n_rec)])
            tr, te = splits.random_split(ws_rec, seed, test_frac)
            met, _, _ = _fit_eval(cls, ws_rec, tr, te, seed, n_classes=n_rec)
            rows.append(dict(dataset=dataset, sonda="S1_kayit_kimligi", model=name,
                             tohum=seed, sans_duzeyi=1.0 / n_rec,
                             etiketi_degisen_oran=np.nan, **met))

            # S2: KAYIT DUZEYINDE etiket permutasyonu.
            # NOT: kayit ICI permutasyon anlamsizdir, cunku bir kayit tek sinif
            # icerir. Her kayda rastgele (kayit icinde sabit) bir sinif atanir.
            rng = np.random.default_rng(1000 + seed)
            recs = np.unique(ws_ov.rec_id)
            yeni = rng.permutation(np.array([ws_ov.y[ws_ov.rec_id == r][0] for r in recs]))
            esl = dict(zip(recs.tolist(), yeni.tolist()))
            y_perm = np.array([esl[int(r)] for r in ws_ov.rec_id])
            ws_p = WindowSet(ws_ov.X, y_perm, ws_ov.cond, ws_ov.specimen, ws_ov.rec_id,
                             ws_ov.start, ws_ov.pos_frac, ws_ov.classes)
            degisen = float(np.mean(y_perm != ws_ov.y))

            tr, te = splits.random_split(ws_p, seed, test_frac)
            met, _, _ = _fit_eval(cls, ws_p, tr, te, seed)
            rows.append(dict(dataset=dataset, sonda="S2_permutasyon_rastgele_bolme",
                             model=name, tohum=seed, sans_duzeyi=1.0 / n_cls,
                             etiketi_degisen_oran=degisen, **met))

            # S2b: ayni permutasyon, kosul-ayrik bolme. Sans duzeyine inmeli.
            tr, te = splits.leave_one_cond_out(ws_p, conds[-1])
            if splits.gecerli_kat(ws_p, tr, te):
                met, _, _ = _fit_eval(cls, ws_p, tr, te, seed)
                rows.append(dict(dataset=dataset, sonda="S2b_permutasyon_kosul_ayrik",
                                 model=name, tohum=seed, sans_duzeyi=1.0 / n_cls,
                                 etiketi_degisen_oran=degisen, **met))
            print(f"  [sonda:{dataset}] {name} tohum={seed} bitti", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------- 3) kalibrasyon
def run_calibration(ws_no: WindowSet, models: dict, seeds=None,
                    k_values=(10, 20, 50, 100, 200, 500), n_draws=20, dataset=""):
    """P4'te guvenilirlik egrileri ve sicaklik-k egrisi."""
    seeds = seeds or C.SEEDS
    conds = sorted(set(ws_no.cond.tolist()))
    n_cls = ws_no.n_classes
    rel_rows, temp_rows = [], []

    def mfn(p, y):
        return all_metrics(p, y, n_cls)

    for name, cls in models.items():
        for seed in seeds:
            for c in conds:
                tr, te = splits.leave_one_cond_out(ws_no, c)
                if not splits.gecerli_kat(ws_no, tr, te):
                    continue
                _, p_te, _ = _fit_eval(cls, ws_no, tr, te, seed)

                ws_tr = ws_no.subset(tr)
                tr2, va2 = splits.random_split(ws_tr, seed, 0.2)
                _, p_va, _ = _fit_eval(cls, ws_tr, tr2, va2, seed)

                for kosul_ad, p, yy in (("ic_dogrulama", p_va, ws_tr.y[va2]),
                                        ("hedef_kosul", p_te, ws_no.y[te])):
                    t = reliability_table(p, yy, C.N_ECE_BINS)
                    for j, (k_, v_) in enumerate([("dataset", dataset), ("model", name),
                                                  ("tohum", seed), ("test_kosul", c),
                                                  ("kosul", kosul_ad)]):
                        t.insert(j, k_, v_)
                    rel_rows.append(t)

                base = mfn(p_te, ws_no.y[te])
                temp_rows.append(dict(dataset=dataset, model=name, tohum=seed,
                                      test_kosul=c, k=0, cekilis=-1, sicaklik=1.0,
                                      n_kalibrasyon=0, n_degerlendirme=len(te), **base))
                for r in calib.temperature_curve(p_te, ws_no.y[te], k_values,
                                                 n_draws, seed, n_cls, mfn):
                    temp_rows.append(dict(dataset=dataset, model=name, tohum=seed,
                                          test_kosul=c, **r))
                print(f"  [kalibrasyon:{dataset}] {name} tohum={seed} kosul={c} bitti",
                      flush=True)
    rel = pd.concat(rel_rows, ignore_index=True) if rel_rows else pd.DataFrame()
    return rel, pd.DataFrame(temp_rows)


def onkayit_hukmu(temp_df: pd.DataFrame, esik_oran=0.90) -> pd.DataFrame:
    """ONCEDEN KAYITLI OLCUT: sicaklik olcekleme, cekilislerin en az %90'inda
    duzeltmesiz ECE'nin altina inerse o k degerinde ONERILIR."""
    if temp_df.empty:
        return pd.DataFrame()
    key = ["dataset", "model", "tohum", "test_kosul"]
    base = temp_df[temp_df.k == 0].groupby(key)["ece"].mean().rename("ece0")
    d = temp_df[temp_df.k > 0].merge(base, on=key)
    d["iyilesti"] = d.ece < d.ece0
    rows = []
    for (ds, model, k), g in d.groupby(["dataset", "model", "k"]):
        oran = float(g.iyilesti.mean())
        rows.append(dict(dataset=ds, model=model, k=int(k), n_cekilis=len(g),
                         iyilesme_orani=oran, ece_medyan=float(g.ece.median()),
                         ece0_medyan=float(g.ece0.median()),
                         hukum="ONERILIR" if oran >= esik_oran else "ONERILMEZ"))
    return pd.DataFrame(rows).sort_values(["dataset", "model", "k"]).reset_index(drop=True)
