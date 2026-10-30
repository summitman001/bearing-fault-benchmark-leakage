"""Bilinmeyen bir veri klasorunu tarar ve doldurulacak manifest taslagi uretir.

Kullanim:
    python run.py --asama tara --ham <klasor> --cikti sonuc

Cikti: manifest_taslak.csv. sinif / kosul / numune kolonlari yol
parcalarindan TAHMIN edilir; elle DOGRULANMALIDIR.

UYGUNLUK KAPISI: her veri kumesi icin su uc sorunun cevabi evet olmali.
  1. Ham surekli sinyal var mi (onceden pencerelenmemis)?
  2. Kayit kimligi (hangi dosyadan geldigi) biliniyor mu?
  3. Calisma kosulu etiketi var mi?
Numune kimligi varsa bonus: numune-ayrik kademe kurulabilir.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

UZANTILAR = {".mat", ".csv", ".txt", ".npy"}

SINIF_IPUCU = [
    (r"(?i)\b(normal|healthy|health|good|baseline|saglam|none)\b", "Normal"),
    (r"(?i)(inner|\bir\b|ic_?bilezik)", "IR"),
    (r"(?i)(outer|\bor\b|dis_?bilezik)", "OR"),
    (r"(?i)(ball|\bb\d|bilya|roller)", "Ball"),
    (r"(?i)(cage|kafes)", "Cage"),
    (r"(?i)(comb|mixed|karma)", "Combined"),
    (r"(?i)(tooth|gear|chipped|miss|root|surface|disli)", "Gear"),
]


def _tahmin_sinif(yol: str):
    for pat, ad in SINIF_IPUCU:
        if re.search(pat, yol):
            return ad
    return ""


def _tahmin_kosul(yol: str):
    m = re.search(r"(?i)(\d+)\s*(hp|hz|rpm|rps|n|nm|v)\b", yol)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    m = re.search(r"(?i)\bload[_\- ]?(\d+)", yol)
    return f"load{m.group(1)}" if m else ""


def _kanallar(path: Path, limit=6):
    """Dosyadaki aday sinyal kanallarini (ad, uzunluk) olarak dondurur."""
    p = str(path)
    try:
        if p.endswith(".mat"):
            from scipy.io import loadmat
            m = loadmat(p)
            out = [(k, int(np.asarray(v).size)) for k, v in m.items()
                   if not k.startswith("__") and isinstance(v, np.ndarray)
                   and np.asarray(v).size > 100]
        elif p.endswith(".npy"):
            a = np.load(p, mmap_mode="r")
            out = ([("0", int(a.shape[0]))] if a.ndim == 1
                   else [(str(i), int(a.shape[0])) for i in range(min(a.shape[1], limit))])
        else:
            df = pd.read_csv(p, sep=None, engine="python", nrows=5000)
            num = df.select_dtypes("number")
            n = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
            out = [(str(c), n) for c in num.columns[:limit]]
        return sorted(out, key=lambda t: -t[1])[:limit]
    except Exception as e:  # noqa: BLE001
        return [(f"OKUNAMADI: {type(e).__name__}", 0)]


def tara(ham_kok, max_dosya=2000) -> pd.DataFrame:
    """Klasoru tarar, her (dosya, kanal) icin bir manifest satiri onerir."""
    kok = Path(ham_kok)
    rows = []
    dosyalar = [p for p in sorted(kok.rglob("*")) if p.suffix.lower() in UZANTILAR]
    for p in dosyalar[:max_dosya]:
        rel = p.relative_to(kok).as_posix()
        kanallar = _kanallar(p)
        for ad, n in kanallar:
            rows.append(dict(
                dataset=rel.split("/")[0] if "/" in rel else kok.name,
                file=rel, channel=ad, n_ornek=n, fs="",
                sinif=_tahmin_sinif(rel), kosul=_tahmin_kosul(rel),
                numune="", boyut_mb=round(p.stat().st_size / 1e6, 2),
                kullan=1 if ad == kanallar[0][0] else 0))
    df = pd.DataFrame(rows)
    if len(dosyalar) > max_dosya:
        print(f"UYARI: {len(dosyalar)} dosya var, ilk {max_dosya} tarandi.")
    return df


def uygunluk_raporu(df: pd.DataFrame) -> pd.DataFrame:
    """Veri kumesi basina uygunluk kapisi ozeti."""
    rows = []
    for ds, g in df.groupby("dataset"):
        g1 = g[g.kullan == 1]
        n_kisa = int((g1.n_ornek < 20000).sum())
        rows.append(dict(
            dataset=ds, n_dosya=g1.file.nunique(), n_kanal=len(g),
            medyan_uzunluk=int(g1.n_ornek.median()) if len(g1) else 0,
            kisa_kayit_sayisi=n_kisa,
            sinif_tahmini=g1.sinif.replace("", np.nan).nunique(),
            kosul_tahmini=g1.kosul.replace("", np.nan).nunique(),
            ham_sinyal_mi="SUPHELI" if n_kisa > len(g1) / 2 else "EVET",
            not_=("Kayitlar kisa: onceden pencerelenmis olabilir, "
                  "merdiven kurulamaz" if n_kisa > len(g1) / 2 else ""),
        ))
    return pd.DataFrame(rows)
