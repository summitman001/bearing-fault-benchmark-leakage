"""Pencere kumesi ve manifest tabanli veri yukleme.

Her pencere su alanlarla etiketlenir:
    y        : sinif indeksi
    cond     : calisma kosulu (yuk/hiz), veri kumesine gore
    specimen : numune kimligi (rulman/disli birimi). Ayni numuneden gelen
               kayitlar birbirine bagimlidir.
    rec_id   : kayit indeksi (bir dosya/kanal = bir kayit)
    start    : pencerenin kayit icindeki baslangic ornegi
    pos_frac : start / kayit_uzunlugu

Analizimiz HAM SUREKLI SINYAL ve KAYIT KIMLIGI gerektirir. Onceden
pencerelenmis ve kayit kimligi kaybolmus bir veri kumesinde P2-P5
kademeleri kurulamaz; boyle bir veri kumesi disarida birakilmalidir.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ------------------------------------------------------------- pencere kumesi
@dataclass
class WindowSet:
    X: np.ndarray
    y: np.ndarray
    cond: np.ndarray
    specimen: np.ndarray
    rec_id: np.ndarray
    start: np.ndarray
    pos_frac: np.ndarray
    classes: list | None = None

    def __len__(self):
        return len(self.y)

    def subset(self, idx) -> "WindowSet":
        i = np.asarray(idx)
        return WindowSet(self.X[i], self.y[i], self.cond[i], self.specimen[i],
                         self.rec_id[i], self.start[i], self.pos_frac[i], self.classes)

    @property
    def n_classes(self):
        return len(self.classes) if self.classes else int(self.y.max()) + 1

    def ozet(self) -> pd.DataFrame:
        return (pd.DataFrame(dict(sinif=self.y, kosul=self.cond,
                                  numune=self.specimen, kayit=self.rec_id))
                .groupby(["sinif", "kosul", "numune"]).agg(
                    n_pencere=("kayit", "size"), n_kayit=("kayit", "nunique"))
                .reset_index())


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------- sinyal okuma
def read_signal(path, channel=None):
    """Tek boyutlu sinyal dondurur. .mat, .csv, .txt, .npy destekli.

    channel: .mat icin degisken adi, .csv/.txt icin kolon adi veya indeksi,
             .npy icin kolon indeksi. None ise en uzun sayisal dizi secilir.
    """
    p = str(path)
    if p.endswith(".mat"):
        from scipy.io import loadmat
        m = loadmat(p)
        keys = [k for k in m if not k.startswith("__")
                and isinstance(m[k], np.ndarray) and m[k].size > 100]
        if channel:
            cand = [k for k in keys if k == channel] or [k for k in keys if channel in k]
            if not cand:
                raise KeyError(f"{p}: '{channel}' kanali yok. Mevcut: {keys}")
            key = cand[0]
        else:
            key = max(keys, key=lambda k: m[k].size)
        return np.asarray(m[key]).ravel().astype(np.float64)

    if p.endswith(".npy"):
        a = np.load(p)
        if a.ndim == 1:
            return a.astype(np.float64)
        col = int(channel) if channel not in (None, "") else 0
        return np.asarray(a[:, col]).ravel().astype(np.float64)

    # csv / txt
    df = pd.read_csv(p, sep=None, engine="python",
                     header=0 if channel and not str(channel).isdigit() else None)
    num = df.select_dtypes("number")
    if num.shape[1] == 0:
        raise ValueError(f"{p}: sayisal kolon yok")
    if channel in (None, ""):
        col = num.columns[int(np.argmax(num.notna().sum().values))]
    elif str(channel).isdigit():
        col = num.columns[int(channel)]
    else:
        col = channel
    return num[col].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- pencereleme
def _slice(sig, window, hop):
    n = 1 + (len(sig) - window) // hop if len(sig) >= window else 0
    starts = np.arange(n) * hop
    out = np.empty((n, window), dtype=np.float64)
    for i, s in enumerate(starts):
        out[i] = sig[s:s + window]
    return out, starts


REQUIRED_COLS = ["file", "sinif", "kosul", "numune"]


def build_from_manifest(manifest: pd.DataFrame, raw_root, window=2048, hop=2048,
                        max_per_recording=None, verbose=False):
    """Manifest -> (WindowSet, envanter DataFrame).

    Manifest kolonlari: file, sinif, kosul, numune [, channel, fs, dataset]
    Her satir bir KAYITTIR.
    """
    from pathlib import Path
    eksik = [c for c in REQUIRED_COLS if c not in manifest.columns]
    if eksik:
        raise ValueError(f"Manifest'te eksik kolon: {eksik}")
    raw_root = Path(raw_root)
    classes = sorted(manifest["sinif"].astype(str).unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}

    Xs, ys, conds, specs, recs, starts, fracs, inv = [], [], [], [], [], [], [], []
    for rec_id, (_, r) in enumerate(manifest.iterrows()):
        path = raw_root / str(r["file"])
        sig = read_signal(path, r.get("channel"))
        W, st = _slice(sig, window, hop)
        if max_per_recording is not None and len(W) > max_per_recording:
            W, st = W[:max_per_recording], st[:max_per_recording]
        if len(W) == 0:
            if verbose:
                print(f"  ATLANDI (kisa): {path}")
            continue
        Xs.append(W)
        ys.append(np.full(len(W), c2i[str(r["sinif"])]))
        conds.append(np.full(len(W), str(r["kosul"]), dtype=object))
        specs.append(np.full(len(W), str(r["numune"]), dtype=object))
        recs.append(np.full(len(W), rec_id))
        starts.append(st)
        fracs.append(st / max(len(sig) - window, 1))
        inv.append(dict(rec_id=rec_id, dataset=r.get("dataset", ""), dosya=str(r["file"]),
                        sinif=r["sinif"], kosul=r["kosul"], numune=r["numune"],
                        kanal=r.get("channel", ""), fs=r.get("fs", ""),
                        n_ornek=len(sig), n_pencere=len(W)))
        if verbose:
            print(f"  rec {rec_id:3d} {str(r['sinif']):12s} kosul={r['kosul']} "
                  f"numune={r['numune']} n={len(sig):8d} pencere={len(W)}")
    if not Xs:
        raise RuntimeError("Hicbir kayit okunamadi.")
    ws = WindowSet(np.concatenate(Xs), np.concatenate(ys), np.concatenate(conds),
                   np.concatenate(specs), np.concatenate(recs),
                   np.concatenate(starts), np.concatenate(fracs), classes)
    return ws, pd.DataFrame(inv)


def min_windows_per_recording(manifest, raw_root, window=2048, hop=2048) -> int:
    from pathlib import Path
    raw_root = Path(raw_root)
    n = []
    for _, r in manifest.iterrows():
        sig = read_signal(raw_root / str(r["file"]), r.get("channel"))
        n.append(max(1 + (len(sig) - window) // hop, 0))
    return int(min(n))
