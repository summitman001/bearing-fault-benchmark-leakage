"""Merdiven sonuclarindan dagilim ozeti (YENI KOSU YOK, yalnizca mevcut CSV'ler).

Girdi:  sonuc_cwru_tam/, sonuc_hust_tam/, sonuc_seu_tam/ icindeki merdiven.csv
Cikti:  sekiller/dagilim_ozeti.csv, sekiller/dagilim_notu.md

Standart sapmalar sekiller.py ile ayni tanimla (ddof=1) hesaplanir.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sekiller"

DATASET_DIRS = {"CWRU": ROOT / "sonuc_cwru_tam",
                "HUST": ROOT / "sonuc_hust_tam",
                "SEU": ROOT / "sonuc_seu_tam"}
METRICS = ["dogruluk", "makro_f1", "ece"]
MODELS = ["M1_CNN", "M2_RF", "M3_TRIVIAL", "M4_KNN"]
COK_KATLI = ("P4_kosul_ayrik", "P5_numune_ayrik")
TEK_KATLI = ("P1_rastgele_ortusmeli", "P2_rastgele_ortusmesiz", "P3_zamansal_blok")
ORAN_ESIGI = 2.0
SIFIR_TOL = 1e-12  # bunun altindaki sd kayan nokta gurultusu sayilir


def uclar(kat_ort, fn):
    """En kucuk/buyuk kat ortalamasi; esitlik varsa tum katlar ';' ile yazilir."""
    deger = fn(kat_ort.values)
    katlar = kat_ort.index[np.isclose(kat_ort.values, deger, rtol=0, atol=SIFIR_TOL)]
    return deger, ";".join(katlar)


def oran(kat_sd, tohum_sd):
    if tohum_sd < SIFIR_TOL:
        return np.nan
    return kat_sd / tohum_sd


def ozet_satiri(g, dataset, model, protokol, olcut):
    tohum_ort = g.groupby("tohum")[olcut].mean()
    satir = dict(dataset=dataset, model=model, protokol=protokol, olcut=olcut,
                 kat_sayisi=g["kat"].nunique(), tohum_sayisi=g["tohum"].nunique(),
                 tohumlar_arasi_sd=tohum_ort.std(ddof=1))
    if protokol in TEK_KATLI:
        satir["tek_katli"] = True
        return satir

    kat_ort = g.groupby("kat")[olcut].mean()
    kat_sd = kat_ort.std(ddof=1)
    tohum_sd = satir["tohumlar_arasi_sd"]
    kat_min, kat_min_adi = uclar(kat_ort, np.min)
    kat_maks, kat_maks_adi = uclar(kat_ort, np.max)
    satir.update(tek_katli=False, ortalama=g[olcut].mean(), katlar_arasi_sd=kat_sd,
                 kat_min=kat_min, kat_min_adi=kat_min_adi,
                 kat_maks=kat_maks, kat_maks_adi=kat_maks_adi,
                 oran_kat_tohum=oran(kat_sd, tohum_sd),
                 oran_kat_tohum_ddof0=oran(kat_ort.std(ddof=0), tohum_ort.std(ddof=0)))
    return satir


def ozet_tablosu(merdiven):
    satirlar = []
    for dataset, m in merdiven.items():
        for (protokol, model), g in m.groupby(["protokol", "model"]):
            # kat ortalamalari ile tohum ortalamalari ancak denge varsa karsilastirilabilir
            assert len(g) == g["tohum"].nunique() * g["kat"].nunique()
            for olcut in METRICS:
                satirlar.append(ozet_satiri(g, dataset, model, protokol, olcut))
    kolonlar = ["dataset", "model", "protokol", "olcut", "tek_katli", "kat_sayisi", "tohum_sayisi",
                "ortalama", "katlar_arasi_sd", "tohumlar_arasi_sd",
                "kat_min", "kat_min_adi", "kat_maks", "kat_maks_adi",
                "oran_kat_tohum", "oran_kat_tohum_ddof0"]
    return pd.DataFrame(satirlar)[kolonlar]


def f(x, n=4):
    return f"{x:.{n}f}"


def oran_yazi(x):
    return "tanımsız" if pd.isna(x) else f"{x:.1f}"


def buyuk_tablo(satirlar):
    d = ["| dataset | model | protokol | ölçüt | ortalama | kat min–maks | katlar arası sd | tohumlar arası sd | oran |",
         "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in satirlar.iterrows():
        d.append(f"| {r.dataset} | {r.model} | {r.protokol} | {r.olcut} | {f(r.ortalama)} | "
                 f"{f(r.kat_min)}–{f(r.kat_maks)} | {f(r.katlar_arasi_sd)} | "
                 f"{f(r.tohumlar_arasi_sd)} | {oran_yazi(r.oran_kat_tohum)} |")
    return d


def birlesim_tablosu(cok):
    d = ["| dataset | model | protokol | kat sayısı | dogruluk | makro_f1 | ece |",
         "|---|---|---|---:|---:|---:|---:|"]
    for (dataset, model, protokol), g in cok.groupby(["dataset", "model", "protokol"]):
        o = g.set_index("olcut").oran_kat_tohum
        d.append(f"| {dataset} | {model} | {protokol} | {g.kat_sayisi.iloc[0]} | "
                 + " | ".join(oran_yazi(o[m]) for m in METRICS) + " |")
    return d


def not_yaz(oz, merdiven):
    cok = oz[~oz.tek_katli]
    sifir = cok[cok.oran_kat_tohum.isna()]
    tanimli = cok[cok.oran_kat_tohum.notna()]
    buyuk = tanimli[tanimli.oran_kat_tohum > ORAN_ESIGI]
    kucuk = tanimli[tanimli.oran_kat_tohum <= ORAN_ESIGI]
    sifir_poz = sifir[sifir.katlar_arasi_sd >= SIFIR_TOL]
    sifir_iki = sifir[sifir.katlar_arasi_sd < SIFIR_TOL]

    # ddof duyarliligi: ddof=0 ile esigin farkli tarafinda kalanlar
    dv = tanimli[(tanimli.oran_kat_tohum > ORAN_ESIGI) != (tanimli.oran_kat_tohum_ddof0 > ORAN_ESIGI)]

    d = ["# Dağılım notu — katlar arası ve tohumlar arası standart sapma", ""]
    d += ["Yeni koşu yapılmadı. Tüm değerler `dagilim.py` ile `sonuc_cwru_tam/`, `sonuc_hust_tam/`, "
          "`sonuc_seu_tam/` içindeki `merdiven.csv` dosyalarından hesaplandı; tam tablo "
          "`dagilim_ozeti.csv`. Standart sapmalar `sekiller.py` ile aynı tanımla (`ddof=1`).", ""]
    d += ["- `katlar_arasi_sd`: önce her kat için tohumlar üzerinden ortalama, sonra bu kat ortalamalarının sd'si.",
          "- `tohumlar_arasi_sd`: önce her tohum için katlar üzerinden ortalama, sonra bu tohum ortalamalarının sd'si.",
          "- `oran` = katlar arası sd / tohumlar arası sd.",
          "- P1, P2, P3 tek katlıdır (`kat` = `-`, `tek_katli=True`); yalnızca tohumlar arası sd hesaplandı, "
          "`ortalama` ve kat kolonları bu satırlarda boştur.",
          f"- P4/P5'te {len(cok)} satır var ({cok.groupby(['dataset', 'model', 'protokol']).ngroups} "
          "dataset×model×protokol birleşimi × 3 ölçüt). Her hücre dengeli: 5 tohum × tüm katlar, eksik veya yinelenen satır yok.",
          ""]

    birlesim = buyuk.groupby(["dataset", "model", "protokol"]).size()
    kat_sayilari = ", ".join(f"{ds} {p[:2]}={n}" for (ds, p), n in
                             cok.groupby(["dataset", "protokol"]).kat_sayisi.first().items())
    d += ["## 1) Katlar arası sd'nin tohumlar arası sd'nin 2 katından büyük olduğu birleşimler", ""]
    d += [f"Oran {len(tanimli)} satırda tanımlı; bunların {len(buyuk)} tanesinde 2'den büyük, "
          f"{len(kucuk)} tanesinde değil. {len(sifir)} satırda tohumlar arası sd sıfır olduğu için oran tanımsız "
          "(aşağıda ayrı listelendi).", ""]
    d += [f"Oranı 2'yi aşan satırlar {birlesim.size} birleşime, oranı tanımsız satırlar "
          f"{sifir.groupby(['dataset', 'model', 'protokol']).ngroups} birleşime yayılıyor. "
          "Birleşim × ölçüt özeti (\"tanımsız\" = tohumlar arası sd sıfır):", ""]
    d += birlesim_tablosu(cok) + [""]
    d += ["Okurken dikkat edilecekler:", "",
          f"- Katlar arası sd kat sayısı kadar değerden hesaplanır: {kat_sayilari}. Özellikle SEU'da iki değerin sd'sidir.",
          "- Oran, mutlak büyüklüğü göstermez; bu yüzden aşağıdaki tablolarda kat aralığı da verildi. "
          "Örneğin SEU M2_RF P4 dogruluk satırında oran 2'yi aşıyor ama iki sd de 0.002'nin altında.", ""]
    d += ["### Oranı 2'yi aşan satırlar (orana göre azalan)", ""]
    d += buyuk_tablo(buyuk.sort_values("oran_kat_tohum", ascending=False)) + [""]

    d += ["### Tohumlar arası sd'nin sıfır olduğu satırlar (oran tanımsız)", ""]
    if len(sifir_poz):
        d += [f"Bu satırların tamamı {', '.join(sorted(sifir.model.unique()))} modellerine ait. Katlar arası sd pozitif olduğu için "
              "bütün dağılım katlardan geliyor; 2 eşiği sayısal olarak test edilemese de bu birleşimler metinde anılmalı.", "",
              "Sıfır bir kararlılık bulgusu değil, tasarımın sonucu: P4 ve P5 bölmeleri tohuma bağlı değil "
              "(`splits.leave_one_cond_out`, `splits.leave_one_specimen_out`), tohum yalnızca modele geçiyor "
              "(`experiments.run_ladder`). `EnvelopeKNNModel` tohum kullanmıyor; `TrivialModel` kullanıyor ama "
              "tohumlar arasında aynı değerleri üretti. Bu yüzden tablolarda bu hücrelerde görülecek ± 0.000, "
              "\"belirsizlik yok\" olarak okunmamalı.", ""]
        d += buyuk_tablo(sifir_poz) + [""]
    if len(sifir_iki):
        d += ["Her iki sd de sıfır (dağılım yok):", ""]
        d += buyuk_tablo(sifir_iki) + [""]
    if not len(sifir):
        d += ["Yok.", ""]

    d += ["### Oranı 2'yi aşmayan satırlar", ""]
    d += buyuk_tablo(kucuk.sort_values("oran_kat_tohum", ascending=False)) if len(kucuk) else ["Yok."]
    d += [""]

    d += ["### `ddof` duyarlılığı", ""]
    if len(dv):
        d += ["Popülasyon sd'si (`ddof=0`) kullanılsaydı aşağıdaki satırlar eşiğin diğer tarafına geçerdi "
              "(tanım değişimi, veri değişimi değil):", ""]
        d += ["| dataset | model | protokol | ölçüt | oran (ddof=1) | oran (ddof=0) |", "|---|---|---|---|---:|---:|"]
        for _, r in dv.iterrows():
            d.append(f"| {r.dataset} | {r.model} | {r.protokol} | {r.olcut} | "
                     f"{r.oran_kat_tohum:.2f} | {r.oran_kat_tohum_ddof0:.2f} |")
    else:
        d += ["`ddof=0` kullanılsaydı da hiçbir satır eşiğin diğer tarafına geçmezdi."]
    d += [""]

    d += ["## 2) HUST P5: model ve kat (rulman modeli) bazında doğruluk", ""]
    p5 = merdiven["HUST"]
    p5 = p5[p5.protokol == "P5_numune_ayrik"]
    katlar = sorted(p5.kat.unique())
    n_test = p5.groupby("kat").n_test.first()
    d += ["Kaynak: `sonuc_hust_tam/merdiven.csv`, `protokol==P5_numune_ayrik`, `dogruluk`. "
          "Her hücre 5 tohum üzerinden ortalama ± tohumlar arası sd. Kat adı, test edilen (eğitimden çıkarılan) rulman modelidir.", ""]
    d += ["| model | " + " | ".join(k.replace("test_numune_", "") for k in katlar) + " |",
          "|---|" + "---:|" * len(katlar)]
    for model in MODELS:
        h = []
        for k in katlar:
            s = p5[(p5.model == model) & (p5.kat == k)].dogruluk
            h.append(f"{s.mean():.4f} ± {s.std(ddof=1):.4f}")
        d.append(f"| {model} | " + " | ".join(h) + " |")
    kat_ort = p5.groupby(["kat", "model"]).dogruluk.mean().unstack()
    d += ["| en yüksek | " + " | ".join(kat_ort.loc[k].idxmax() for k in katlar) + " |"]
    d += ["| n_test | " + " | ".join(str(n_test[k]) for k in katlar) + " |", ""]
    d += ["- \"En yüksek\" satırı kat ortalamalarına göredir; modellerin P5 sıralaması kattan kata değişiyor, "
          "protokol ortalamasına dayalı sıra bunu göstermez.",
          "- 6204 katı doğrudan karşılaştırılmamalı: bu modelde Ball ve Inner+Ball kaydı yok (`doldurma_kaydi.md`), "
          "test kümesi diğer katlardan küçük (n_test satırı) ve daha az sınıf içeriyor.",
          "- `ortalama` kolonu istendiği gibi düz ortalamadır, kat büyüklüğüne göre ağırlıklı değildir.", ""]

    d += ["Tohum bazında ham değerler:", ""]
    d += ["| model | kat | tohum 0 | tohum 1 | tohum 2 | tohum 3 | tohum 4 |", "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for k in katlar:
            s = p5[(p5.model == model) & (p5.kat == k)].set_index("tohum").dogruluk.sort_index()
            d.append(f"| {model} | {k.replace('test_numune_', '')} | " + " | ".join(f"{v:.4f}" for v in s) + " |")
    d += [""]

    d += ["## 3) Şekil 1 ile tablolar arasındaki tanım farkı (doğruluk)", ""]
    d += ["Şekil 1'in P4/P5 hata çubukları (`sekil1_veri.csv`, `dogruluk_std`) tohum ve kat satırlarının birlikte alındığı "
          "havuzlanmış sd'dir (`sekil_kaydi.md`). Bu nedenle değerler tohumlar arası sd'ye değil, katlar arası sd'ye yakındır. "
          "Aynı hücre için şekildeki çubuk ile tablodaki ± farklı nicelikler gösterir; şekil altyazısı bunu belirtmeli.", ""]
    d += ["| dataset | model | protokol | Şekil 1 (havuzlanmış) | katlar arası sd | tohumlar arası sd |",
          "|---|---|---|---:|---:|---:|"]
    for _, r in cok[cok.olcut == "dogruluk"].iterrows():
        m = merdiven[r.dataset]
        havuz = m[(m.model == r.model) & (m.protokol == r.protokol)].dogruluk.std(ddof=1)
        d.append(f"| {r.dataset} | {r.model} | {r.protokol} | {f(havuz)} | {f(r.katlar_arasi_sd)} | {f(r.tohumlar_arasi_sd)} |")
    d += [""]

    tek_sifir = oz[oz.tek_katli & (oz.tohumlar_arasi_sd < SIFIR_TOL)]
    d += ["## 4) Tek katlı protokoller (P1, P2, P3)", ""]
    d += ["Bu protokollerde kat yok; katlar arası sd hesaplanmadı. Tohumlar arası sd `dagilim_ozeti.csv` içinde. "
          f"{len(oz[oz.tek_katli])} satırın {len(tek_sifir)} tanesinde tohumlar arası sd sıfır:", ""]
    d += ["| dataset | model | protokol | ölçütler |", "|---|---|---|---|"]
    for (dataset, model, protokol), g in tek_sifir.groupby(["dataset", "model", "protokol"]):
        d.append(f"| {dataset} | {model} | {protokol} | {', '.join(g.olcut)} |")
    d += [""]

    (OUT / "dagilim_notu.md").write_text("\n".join(d), encoding="utf-8")


def main():
    merdiven = {ds: pd.read_csv(p / "merdiven.csv", keep_default_na=False, na_values=[""])
                for ds, p in DATASET_DIRS.items()}
    oz = ozet_tablosu(merdiven)
    oz.to_csv(OUT / "dagilim_ozeti.csv", index=False)
    not_yaz(oz, merdiven)
    print(f"{len(oz)} satir yazildi -> {OUT}")


if __name__ == "__main__":
    main()
