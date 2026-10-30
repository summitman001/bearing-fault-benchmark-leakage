"""Yapilandirma ve yerlesik CWRU manifesti."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "sonuc"

WINDOW = 2048
HOP_OVERLAP = 512        # P1: %75 ortusme (yaygin pratik)
HOP_NONOVERLAP = 2048    # P2-P5: ortusme yok

SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.30
N_ECE_BINS = 15

# ------------------------------------------------------------------- CWRU
# !!! DOGRULA: asagidaki dosya numaralari literaturdeki standart 10 sinifli
# 12k Drive End kurulumudur. Resmi indirme sayfasiyla TEK TEK karsilastirin;
# yanlis numara sessiz bir etiket hatasi uretir.
CWRU_FILE_MAP = {
    "Normal":   {0: 97,  1: 98,  2: 99,  3: 100},
    "IR007":    {0: 105, 1: 106, 2: 107, 3: 108},
    "B007":     {0: 118, 1: 119, 2: 120, 3: 121},
    "OR007@6":  {0: 130, 1: 131, 2: 132, 3: 133},
    "IR014":    {0: 169, 1: 170, 2: 171, 3: 172},
    "B014":     {0: 185, 1: 186, 2: 187, 3: 188},
    "OR014@6":  {0: 197, 1: 198, 2: 199, 3: 200},
    "IR021":    {0: 209, 1: 210, 2: 211, 3: 212},
    "B021":     {0: 222, 1: 223, 2: 224, 3: 225},
    "OR021@6":  {0: 234, 1: 235, 2: 236, 3: 237},
}
CWRU_URL = "https://engineering.case.edu/sites/default/files/{num}.mat"


def cwru_manifest() -> pd.DataFrame:
    """CWRU icin manifest. Numune tektir (ayni test duzenegi), bu yuzden
    P5 (numune-ayrik) CWRU'da KURULAMAZ; bu bir sinirlamadir, raporlanmalidir."""
    rows = []
    for cls, per_load in CWRU_FILE_MAP.items():
        for load, num in per_load.items():
            # Kanal adi dosya numarasina GORE sabitlenir (orn. "X099_DE_time").
            # Bazi resmi CWRU .mat dosyalari (orn. 99.mat) birden fazla kayda
            # ait degisken barindirir (X098_DE_time VE X099_DE_time bir arada);
            # genel bir "_DE_time" alt-dizge eslesmesi ilk bulunani (yanlis
            # kaydi) sessizce secebilir. Tam degisken adi bu riski kaldirir.
            rows.append(dict(dataset="CWRU", file=f"cwru_mat/{num}.mat",
                             channel=f"X{num:03d}_DE_time", fs=12000, sinif=cls,
                             kosul=f"{load}hp", numune="CWRU_rig"))
    return pd.DataFrame(rows)
