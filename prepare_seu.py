"""SEU bearingset CSV dosyalarindan data/seu_clean/*.npy uretir.

16 satirlik cihaz basligini atlar, Channel2'yi (8 kanalin 1. indeksi) float64
olarak kaydeder. Ayrac dosyaya gore sekme ya da virguldur.

python prepare_seu.py --ham "/yol/SEU bearingset" --cikti data/seu_clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASLIK_SATIRI = 16
KANAL = 1


def oku(yol: Path) -> np.ndarray:
    with open(yol, "rb") as fh:
        ayrac = "\t" if b"\t" in fh.readline() else ","
    return pd.read_csv(yol, sep=ayrac, skiprows=BASLIK_SATIRI, header=None,
                       usecols=[KANAL], dtype=float).to_numpy().ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham", type=Path, required=True)
    ap.add_argument("--cikti", type=Path, default=Path("data/seu_clean"))
    a = ap.parse_args()
    a.cikti.mkdir(parents=True, exist_ok=True)
    for yol in sorted(a.ham.glob("*.csv")):
        x = oku(yol)
        np.save(a.cikti / f"{yol.stem}.npy", x)
        print(f"{yol.stem}: {len(x)} ornek")


if __name__ == "__main__":
    main()
