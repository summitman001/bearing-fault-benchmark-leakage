#!/usr/bin/env python3
"""Deney kosucusu (cok veri kumeli).

Tipik akis
----------
# 0) Boru hattini gercek veri olmadan dogrula (~1 dk)
python run.py --asama hepsi --veri sentetik --hizli --tohum 0

# 1) Ham veri klasorunu tara ve manifest taslagi uret
python run.py --asama tara --ham /yol/veri --cikti sonuc

# 2) manifest_taslak.csv'yi ELLE doldur (sinif/kosul/numune/kullan),
#    manifest.csv olarak kaydet, sonra:
python run.py --asama hepsi --veri manifest --manifest sonuc/manifest.csv --ham /yol/veri

# CWRU'yu indirip yerlesik manifestle kosmak icin:
python run.py --asama indir
python run.py --asama hepsi --veri cwru --ham data
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bfd import config as C          # noqa: E402
from bfd import data as D            # noqa: E402
from bfd import experiments as E     # noqa: E402
from bfd import manifest as MF       # noqa: E402
from bfd import models as M          # noqa: E402
from bfd import synth                # noqa: E402


def ortam_bilgisi():
    import scipy
    import sklearn
    info = dict(python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        info["torch"] = "YOK"
    return info


def cwru_indir(ham_kok: Path):
    hedef = ham_kok / "cwru_mat"
    hedef.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, per_load in C.CWRU_FILE_MAP.items():
        for load, num in per_load.items():
            dst = hedef / f"{num}.mat"
            durum = "mevcut"
            if not dst.exists():
                try:
                    with urllib.request.urlopen(C.CWRU_URL.format(num=num),
                                                timeout=60) as r:
                        dst.write_bytes(r.read())
                    durum = "indirildi"
                except Exception as e:  # noqa: BLE001
                    durum = f"BASARISIZ: {type(e).__name__}: {e}"
            rows.append(dict(sinif=cls, kosul=f"{load}hp", dosya_no=num, durum=durum,
                             boyut=dst.stat().st_size if dst.exists() else 0,
                             sha256=D.sha256(dst) if dst.exists() else ""))
    return pd.DataFrame(rows)


def pencere_kumeleri(man: pd.DataFrame, ham: Path, pencere_basi=None, verbose=True):
    if pencere_basi is None:
        pencere_basi = D.min_windows_per_recording(man, ham, C.WINDOW, C.HOP_NONOVERLAP)
        print(f"  kayit basina pencere (dengeleme): {pencere_basi}")
    ws_no, inv = D.build_from_manifest(man, ham, C.WINDOW, C.HOP_NONOVERLAP,
                                       pencere_basi, verbose=verbose)
    ws_ov, _ = D.build_from_manifest(man, ham, C.WINDOW, C.HOP_OVERLAP,
                                     pencere_basi * C.HOP_NONOVERLAP // C.HOP_OVERLAP)
    return ws_ov, ws_no, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", default="hepsi",
                    choices=["tara", "indir", "merdiven", "sonda", "kalibrasyon", "hepsi"])
    ap.add_argument("--veri", default="sentetik", choices=["sentetik", "cwru", "manifest"])
    ap.add_argument("--ham", default="data", help="ham veri kok klasoru")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--hizli", action="store_true", help="CNN'i atla")
    ap.add_argument("--tohum", type=int, nargs="*", default=None)
    ap.add_argument("--pencere-basi", type=int, default=None)
    ap.add_argument("--cikti", default=None)
    args = ap.parse_args()

    out = Path(args.cikti) if args.cikti else C.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ham = Path(args.ham)
    seeds = args.tohum if args.tohum else C.SEEDS
    models = M.FAST_MODELS if args.hizli else M.ALL_MODELS
    (out / "ortam.json").write_text(json.dumps(ortam_bilgisi(), indent=1))

    if args.asama == "tara":
        df = MF.tara(ham)
        df.to_csv(out / "manifest_taslak.csv", index=False)
        rap = MF.uygunluk_raporu(df)
        rap.to_csv(out / "uygunluk_raporu.csv", index=False)
        print(rap.to_string(index=False))
        print(f"\nTaslak: {out / 'manifest_taslak.csv'}  ({len(df)} satir)")
        print("sinif/kosul/numune kolonlarini DOGRULAYIN, kullan=1 olanlar alinir.")
        return

    if args.asama == "indir":
        inv = cwru_indir(ham)
        inv.to_csv(out / "indirme_envanteri.csv", index=False)
        print(inv[["sinif", "kosul", "dosya_no", "durum", "boyut"]].to_string(index=False))
        print(f"\nBasarisiz: {int(inv.durum.str.startswith('BASARISIZ').sum())}/{len(inv)}")
        return

    t0 = time.time()
    # --- veri kumelerini hazirla -------------------------------------------
    kumeler = {}
    if args.veri == "sentetik":
        print("UYARI: SENTETIK veri. Hicbir sayi makaleye girmez.")
        kumeler["SENTETIK"] = (synth.make_windowset(60, hop=C.HOP_OVERLAP, n_specimens=2),
                               synth.make_windowset(30, hop=C.HOP_NONOVERLAP, n_specimens=2))
    elif args.veri == "cwru":
        man = C.cwru_manifest()
        ws_ov, ws_no, inv = pencere_kumeleri(man, ham, args.pencere_basi)
        inv.to_csv(out / "veri_envanteri_CWRU.csv", index=False)
        kumeler["CWRU"] = (ws_ov, ws_no)
    else:
        if not args.manifest:
            sys.exit("--manifest gerekli")
        man_all = pd.read_csv(args.manifest)
        if "kullan" in man_all.columns:
            man_all = man_all[man_all.kullan == 1]
        for ds, man in man_all.groupby("dataset"):
            print(f"\n--- {ds}: {len(man)} kayit ---")
            try:
                ws_ov, ws_no, inv = pencere_kumeleri(man.reset_index(drop=True), ham,
                                                     args.pencere_basi)
                inv.to_csv(out / f"veri_envanteri_{ds}.csv", index=False)
                kumeler[str(ds)] = (ws_ov, ws_no)
            except Exception as e:  # noqa: BLE001
                print(f"  ATLANDI: {type(e).__name__}: {e}")

    for ds, (ws_ov, ws_no) in kumeler.items():
        print(f"\n### {ds}: pencere ortusmeli={len(ws_ov)} ortusmesiz={len(ws_no)} "
              f"sinif={ws_no.n_classes} kosul={len(set(ws_no.cond))} "
              f"numune={len(set(ws_no.specimen))}")
        ws_no.ozet().to_csv(out / f"ozet_{ds}.csv", index=False)

    def birlestir_yaz(parcalar, ad):
        if parcalar:
            df = pd.concat(parcalar, ignore_index=True)
            df.to_csv(out / ad, index=False)
            return df
        return pd.DataFrame()

    if args.asama in ("merdiven", "hepsi"):
        p = [E.run_ladder(a, b, models, seeds=seeds, dataset=ds)
             for ds, (a, b) in kumeler.items()]
        df = birlestir_yaz(p, "merdiven.csv")
        print("\n=== MERDIVEN ===")
        print(df.groupby(["dataset", "protokol", "model"])
              [["dogruluk", "ece", "ort_guven", "ortusme_orani"]].mean().round(3).to_string())

    if args.asama in ("sonda", "hepsi"):
        p = [E.run_probes(a, models, seeds=seeds, dataset=ds)
             for ds, (a, b) in kumeler.items()]
        df = birlestir_yaz(p, "sonda.csv")
        print("\n=== SONDALAR ===")
        print(df.groupby(["dataset", "sonda", "model"])
              [["dogruluk", "sans_duzeyi"]].mean().round(3).to_string())

    if args.asama in ("kalibrasyon", "hepsi"):
        rels, temps = [], []
        for ds, (a, b) in kumeler.items():
            r, t = E.run_calibration(b, models, seeds=seeds, dataset=ds)
            rels.append(r)
            temps.append(t)
        birlestir_yaz(rels, "guvenilirlik.csv")
        temp = birlestir_yaz(temps, "sicaklik_k.csv")
        hukum = E.onkayit_hukmu(temp)
        hukum.to_csv(out / "onkayit_hukmu.csv", index=False)
        print("\n=== ONCEDEN KAYITLI OLCUT ===")
        print(hukum.to_string(index=False))

    print(f"\nBitti. Sure: {time.time() - t0:.1f} s. Cikti: {out}")


if __name__ == "__main__":
    main()
