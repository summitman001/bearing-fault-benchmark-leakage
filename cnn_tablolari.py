"""M1_CNN icin makale tablolarindaki eksik degerler (YENI KOSU YOK).

Girdi:  sonuc_{cwru,hust,seu}_tam/{merdiven,sonda,sicaklik_k,onkayit_hukmu}.csv
        sonuc/manifest_seu_run.csv (yalnizca SEU S2b etiket permutasyonunu yeniden kurmak icin)
Cikti:  sekiller/cnn_tablo_degerleri.csv, sekiller/cnn_tablo_degerleri.md

Standart sapmalar tohumlar arasidir (once her tohum icin katlar uzerinden
ortalama), ddof=1; sekiller.py ve dagilim.py ile ayni tanim. CWRU, tanimlari tablodaki dolu
hucrelerle karsilastirabilmek icin referans olarak dahildir.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sekiller"

DATASET_DIRS = {"CWRU": ROOT / "sonuc_cwru_tam",
                "HUST": ROOT / "sonuc_hust_tam",
                "SEU": ROOT / "sonuc_seu_tam"}
MODEL = "M1_CNN"
PROTOKOLLER = ["P1_rastgele_ortusmeli", "P2_rastgele_ortusmesiz", "P3_zamansal_blok",
               "P4_kosul_ayrik", "P5_numune_ayrik"]
GUVEN_OLCUTLERI = ["ort_guven", "guven_eksi_dogruluk", "nll", "brier"]
S2B = "S2b_permutasyon_kosul_ayrik"
ESIK_ORAN = 0.90  # experiments.onkayit_hukmu ile ayni
T_TEKRAR_ESIGI = 10  # ayni T (4 ondalik) en az bu kadar cekiliste cikiyorsa kestirim degil, arama izi sayilir


def ort_sd(g, olcut):
    tohum_ort = g.groupby("tohum")[olcut].mean()
    return g[olcut].mean(), tohum_ort.std(ddof=1), tohum_ort.size


def merdiven_satirlari(ds, m, tablo, olcutler):
    satirlar = []
    for protokol in PROTOKOLLER:
        g = m[(m.model == MODEL) & (m.protokol == protokol)]
        if g.empty:
            continue
        for olcut in olcutler:
            deger, sd, n = ort_sd(g, olcut)
            satirlar.append(dict(tablo=tablo, dataset=ds, model=MODEL, kademe=protokol, k="",
                                 nicelik=olcut, deger=deger, tohumlar_arasi_sd=sd, n_tohum=n))
    return satirlar


def kalibrasyon_tablosu(ds, t, h):
    """Tablo VI satirlari; onkayit_hukmu.csv sicaklik_k.csv'den yeniden uretilerek dogrulanir."""
    t = t[t.model == MODEL]
    h = h[h.model == MODEL].set_index("k")
    key = ["tohum", "test_kosul"]
    taban = t[t.k == 0].set_index(key).ece.rename("ece0")
    d = t[t.k > 0].join(taban, on=key)
    d["iyilesti"] = d.ece < d.ece0
    sayac = d.sicaklik.round(4).value_counts()
    satirlar = []
    for k, g in d.groupby("k"):
        oran = g.iyilesti.mean()
        assert np.isclose(oran, h.loc[k, "iyilesme_orani"], rtol=0, atol=1e-12)
        assert np.isclose(g.ece.median(), h.loc[k, "ece_medyan"], rtol=0, atol=1e-12)
        assert (oran >= ESIK_ORAN) == (h.loc[k, "hukum"] == "ONERILIR")
        birim = g.groupby(key).iyilesti.mean()
        tekrar = g.sicaklik.round(4).map(sayac)
        satirlar.append(dict(dataset=ds, k=int(k), n_kalibrasyon=int(g.n_kalibrasyon.median()),
                             T_tekrar_orani=(tekrar >= T_TEKRAR_ESIGI).mean(),
                             n_cekilis=len(g), ece0_medyan=g.ece0.median(), ece_medyan=g.ece.median(),
                             iyilesme_orani=oran, medyan_T=g.sicaklik.median(),
                             T_q25=g.sicaklik.quantile(.25), T_q75=g.sicaklik.quantile(.75),
                             hukum=h.loc[k, "hukum"], n_birim=birim.size,
                             hic_iyilesmeyen_birim=int((birim == 0).sum()),
                             hep_iyilesen_birim=int((birim == 1).sum())))
    return pd.DataFrame(satirlar)


def kosul_tablosu(ds, t):
    """Havuzlanmis medyanin gizledigi hedef kosul farklari (tum k > 0 cekilisleri)."""
    t = t[t.model == MODEL]
    key = ["tohum", "test_kosul"]
    taban = t[t.k == 0].set_index(key).ece.rename("ece0")
    d = t[t.k > 0].join(taban, on=key)
    d["iyilesti"] = d.ece < d.ece0
    g = d.groupby("test_kosul").agg(ece0_medyan=("ece0", "median"), ece_medyan=("ece", "median"),
                                    iyilesme_orani=("iyilesti", "mean"), medyan_T=("sicaklik", "median"))
    return g.reset_index().assign(dataset=ds)


def seu_permutasyonlari(sonda):
    """run_probes'taki S2 permutasyonunu model egitmeden yeniden kurar."""
    man = pd.read_csv(ROOT / "sonuc" / "manifest_seu_run.csv")
    man = man[man.kullan == 1].reset_index(drop=True)
    siniflar = sorted(man.sinif.astype(str).unique())
    y = man.sinif.astype(str).map({c: i for i, c in enumerate(siniflar)}).values
    kosul = man.kosul.astype(str).values
    test = kosul == sorted(set(kosul))[-1]
    kayitli = sonda[(sonda.model == MODEL) & (sonda.sonda == "S2_permutasyon_rastgele_bolme")].set_index("tohum")
    gecerli_csv = set(sonda[(sonda.model == MODEL) & (sonda.sonda == S2B)].tohum)
    satirlar = []
    for tohum in range(5):
        yeni = np.random.default_rng(1000 + tohum).permutation(y)
        degisen = float(np.mean(yeni != y))
        s_tr, s_te = set(yeni[~test].tolist()), set(yeni[test].tolist())
        gecerli = s_te <= s_tr
        # yeniden kurulum, CSV'deki etiketi_degisen_oran ve gecerli tohumla birebir uymali
        assert np.isclose(degisen, kayitli.loc[tohum, "etiketi_degisen_oran"], rtol=0, atol=1e-9)
        assert gecerli == (tohum in gecerli_csv)
        egitim = dict(zip(y[~test].tolist(), yeni[~test].tolist()))
        uyum = sum(egitim[a] == b for a, b in zip(y[test].tolist(), yeni[test].tolist()))
        satirlar.append(dict(tohum=tohum, degisen=degisen, gecerli=gecerli,
                             uyum=uyum, n_test_kayit=int(test.sum()),
                             test_kosulu=sorted(set(kosul))[-1]))
    return pd.DataFrame(satirlar)


def f(x, n=4):
    return f"{x:.{n}f}"


def not_yaz(uzun, kal, kos, sonda, perm, seu_m):
    d = ["# M1_CNN — eksik tablo değerleri", ""]
    d += ["Yeni koşu yapılmadı. Tüm değerler `cnn_tablolari.py` ile `sonuc_*_tam/` CSV'lerinden hesaplandı; tam duyarlıklı "
          "değerler `cnn_tablo_degerleri.csv`. Ortalama ± sd: sd tohumlar arasıdır (`ddof=1`), P4/P5'te önce katlar üzerinden "
          "ortalama alınır. CWRU, tanımları tablodaki dolu hücrelerle karşılaştırmak için referans olarak eklendi.", ""]

    d += ["## Tablo III — ECE (ortalama ± sd)", "",
          "| dataset | P1 | P2 | P3 | P4 | P5 |", "|---|---:|---:|---:|---:|---:|"]
    u = uzun[uzun.tablo == "III"]
    for ds in DATASET_DIRS:
        h = []
        for p in PROTOKOLLER:
            r = u[(u.dataset == ds) & (u.kademe == p)]
            h.append(f"{f(r.deger.iloc[0])} ± {f(r.tohumlar_arasi_sd.iloc[0])}" if len(r) else "yok")
        d.append(f"| {ds} | " + " | ".join(h) + " |")
    d += [""]

    d += ["## Tablo V — ortalama güven, güven farkı, NLL, Brier (ortalama ± sd)", "",
          "Güven farkı = `guven_eksi_dogruluk` (ortalama güven − doğruluk); pozitif değer aşırı güven demektir. "
          "P4 satırları `sicaklik_k.csv`'nin k=0 satırlarıyla birebir aynıdır.", "",
          "| dataset | protokol | ort. güven | güven farkı | NLL | Brier |", "|---|---|---:|---:|---:|---:|"]
    u = uzun[uzun.tablo == "V"]
    for ds in DATASET_DIRS:
        for p in PROTOKOLLER:
            r = u[(u.dataset == ds) & (u.kademe == p)].set_index("nicelik")
            if r.empty:
                continue
            d.append(f"| {ds} | {p} | " + " | ".join(
                f"{f(r.deger[o])} ± {f(r.tohumlar_arasi_sd[o])}" for o in GUVEN_OLCUTLERI) + " |")
    d += [""]

    d += ["## Tablo VI — k başına ECE, iyileşme oranı, medyan T", "",
          "Kaynak: `onkayit_hukmu.csv` (ECE ve iyileşme oranı; `sicaklik_k.csv`'den yeniden üretilip eşitliği doğrulandı) ve "
          "`sicaklik_k.csv` (`sicaklik` medyanı). Havuz: tüm tohum × hedef koşul × 20 çekiliş. "
          "Eşik: iyileşme oranı ≥ 0.90.", "",
          "| dataset | k | gerçek n_kal | n_çekiliş | ECE₀ medyan | ECE medyan | iyileşme oranı | medyan T | T çeyrekleri | tekrar eden T oranı | hüküm |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in kal.iterrows():
        d.append(f"| {r.dataset} | {r.k} | {r.n_kalibrasyon} | {r.n_cekilis} | {f(r.ece0_medyan)} | {f(r.ece_medyan)} | "
                 f"{f(r.iyilesme_orani, 3)} | {f(r.medyan_T, 3)} | {f(r.T_q25, 3)}–{f(r.T_q75, 3)} | "
                 f"{f(r.T_tekrar_orani, 3)} | {r.hukum} |")
    d += [""]
    d += ["- **Gerçek kalibrasyon örneği k'dan az olabilir:** `calib.temperature_curve` sınıf başına `k // sınıf_sayısı` örnek seçiyor. "
          "`gerçek n_kal` sütunu fiili sayıdır; HUST'ta 7 sınıf olduğu için nominal k'dan küçüktür.",
          "- **Hüküm ile medyan ECE ayrı şeyler:** kural çekiliş düzeyinde iyileşme oranına bakar. "
          "Medyan ECE düşse de oran 0.90'ın altında kalabilir; \"kural sağlanmadı\" \"ECE düşmedi\" anlamına gelmez.",
          f"- **CWRU ve HUST'ta medyan T dikkatle yazılmalı:** `tekrar eden T oranı`, aynı T değerinin (4 ondalık) o veri kümesinde en az "
          f"{T_TEKRAR_ESIGI} çekilişte çıktığı çekilişlerin payıdır. Sürekli bir kestirimde tekrar beklenmez. En sık değer 0.2057, "
          "`fit_temperature` aramasının izidir: tamamı doğru ve çok güvenli yapay örneklerle (marj 1e-6 ve altı, n=10 ve 50) fonksiyon model eğitilmeden "
          "çağrıldığında tam olarak 0.2057 döndürüyor, çünkü NLL düzleşiyor ve arama T'yi belirleyemiyor. Bu çekilişlerde T bir kestirim değildir. "
          "Diğer tekrar eden değerler 0.1198 ve alt sınır 0.05'tir; 0.1198 birebir yeniden üretilmedi (marj 1e-3'te 0.1190). "
          "SEU'da tekrar yoktur ve T üst sınıra (20) ulaşmaz.", ""]
    d += ["Hedef koşula göre (tüm k > 0 çekilişleri birlikte):", "",
          "| dataset | hedef koşul | ECE₀ medyan | ECE medyan | iyileşme oranı | medyan T |", "|---|---|---:|---:|---:|---:|"]
    for _, r in kos.iterrows():
        d.append(f"| {r.dataset} | {r.test_kosul} | {f(r.ece0_medyan)} | {f(r.ece_medyan)} | {f(r.iyilesme_orani, 3)} | {f(r.medyan_T, 3)} |")
    d += ["", "Havuzlanmış medyan düşüşü bütün koşullar için geçerli değil; `kural sağlanmadı` hükmünün kaynağı belirli koşullardır.", ""]
    d += ["Tohum × hedef koşul birimlerine göre iyileşme (her birimde k başına 20 çekiliş):", "",
          "| dataset | k | birim | hiç iyileşmeyen | her çekilişte iyileşen |", "|---|---:|---:|---:|---:|"]
    for _, r in kal[kal.dataset != "CWRU"].iterrows():
        d.append(f"| {r.dataset} | {r.k} | {r.n_birim} | {r.hic_iyilesmeyen_birim} | {r.hep_iyilesen_birim} |")
    d += [""]

    s = sonda[sonda.sonda == S2B]
    c = s[s.model == MODEL].iloc[0]
    d += ["## Tablo IV — SEU S2b, M1_CNN", "",
          f"**Doğruluk {f(c.dogruluk)}** (makro F1 {f(c.makro_f1)}, ECE {f(c.ece)}), şans düzeyi {f(c.sans_duzeyi, 3)}, "
          f"n_test {c.n_test}. **n = 1 tohum** (tohum {c.tohum}); sd hesaplanamaz. Kaynak: `sonuc_seu_tam/sonda.csv`.", "",
          "| model | doğruluk | makro F1 | ECE |", "|---|---:|---:|---:|"]
    for _, r in s.iterrows():
        d.append(f"| {r.model} | {f(r.dogruluk)} | {f(r.makro_f1)} | {f(r.ece)} |")
    d += ["", "### Bu değer şans düzeyiyle karşılaştırılmamalı", "",
          "Etiket permütasyonu model eğitilmeden yeniden kuruldu (`etiketi_degisen_oran` ve geçerli tohum CSV ile birebir uyuşuyor; "
          "betikte `assert`). S2b testi tek kattır (test koşulu "
          f"{perm.test_kosulu.iloc[0]}, 5 kayıt).", "",
          "| tohum | etiketi değişen oran | geçerli kat mı | testte, aynı gerçek sınıfın eğitimdeki etiketini taşıyan kayıt |",
          "|---:|---:|---|---:|"]
    for _, r in perm.iterrows():
        d.append(f"| {r.tohum} | {f(r.degisen, 2)} | {'evet' if r.gecerli else 'hayır'} | {r.uyum}/{r.n_test_kayit} |")
    test_kat = f"test_kosul_{perm.test_kosulu.iloc[0]}"
    d += ["", "Geçerli olan tek tohumda (0), test kayıtlarının 3/5'i aynı gerçek sınıfın eğitimdeki etiketini koruyor. Yani bu tohumda "
          "permütasyon koşullar arasında sınıf bilgisini yok etmiyor. Gerçek arıza sınıfını koşullar arasında genelleyen bir model, "
          "bu 3 sınıfı tanıyorsa bu etiketlerle en çok 0.60 alır. Aynı test koşulunda gerçek etiketli P4 doğruluğu bu okumayla uyumlu:", "",
          "| model | P4 gerçek etiket, tohum 0 | S2b permütasyon, tohum 0 |", "|---|---:|---:|"]
    for _, r in s.iterrows():
        p4 = seu_m[(seu_m.model == r.model) & (seu_m.protokol == "P4_kosul_ayrik") & (seu_m.kat == test_kat) & (seu_m.tohum == 0)].dogruluk.iloc[0]
        d.append(f"| {r.model} | {f(p4)} | {f(r.dogruluk)} |")
    d += ["", "Gerçek etiketle şans düzeyinde kalan M2_RF ve M3_TRIVIAL S2b'de de şans düzeyinde; gerçek etiketle iyi olan M4_KNN ve M1_CNN "
          "0.60 sınırında. Model tahminleri `sonda.csv`'de bulunmadığı için bu doğrudan gösterilmiş değil, verilerle tutarlı bir açıklamadır; "
          "`RAPOR.md`'nin M4_KNN için önerdiği kayıt \"parmak izi\" açıklamasından daha basittir.", "",
          "Ayrıca geçerlilik filtresi (`gecerli_kat`: testteki her etiket eğitimde de olmalı) seçim etkisi yaratabilir: "
          "filtreden geçen tek tohum, etiket uyumu olan tek tohumdur. Bu nedenle değer, S2b'nin \"şans düzeyine iniş\" "
          "bulgusunu SEU için ne destekler ne de çürütür.", "",
          "Tablo IV için önerilen dipnot:", "",
          f"> SEU S2b: n = 1 seed ({f(c.dogruluk, 3)}); the other four permutations gave no valid fold. In this permutation 3 of 5 test "
          "recordings keep the label of the same true class in training, so the attainable accuracy for a model that generalises "
          "across conditions is 0.60, not the nominal chance level of 0.20.", ""]
    (OUT / "cnn_tablo_degerleri.md").write_text("\n".join(d), encoding="utf-8")


def main():
    uzun, kal, kos, tum_m = [], [], [], {}
    for ds, p in DATASET_DIRS.items():
        m = tum_m[ds] = pd.read_csv(p / "merdiven.csv", keep_default_na=False, na_values=[""])
        uzun += merdiven_satirlari(ds, m, "III", ["ece"])
        uzun += merdiven_satirlari(ds, m, "V", GUVEN_OLCUTLERI)
        t = pd.read_csv(p / "sicaklik_k.csv")
        kal.append(kalibrasyon_tablosu(ds, t, pd.read_csv(p / "onkayit_hukmu.csv")))
        kos.append(kosul_tablosu(ds, t))
    kal = pd.concat(kal, ignore_index=True)
    kos = pd.concat(kos, ignore_index=True)
    sonda = pd.read_csv(DATASET_DIRS["SEU"] / "sonda.csv")

    uzun = pd.DataFrame(uzun)
    vi = kal.melt(id_vars=["dataset", "k"], value_vars=["n_kalibrasyon", "n_cekilis", "ece0_medyan", "ece_medyan",
                                                        "iyilesme_orani", "medyan_T", "T_q25", "T_q75",
                                                        "T_tekrar_orani", "hukum"],
                  var_name="nicelik", value_name="deger").assign(tablo="VI", model=MODEL, kademe="P4_kosul_ayrik")
    c = sonda[(sonda.model == MODEL) & (sonda.sonda == S2B)].iloc[0]
    iv = pd.DataFrame([dict(tablo="IV", dataset="SEU", model=MODEL, kademe=S2B, k="", nicelik=o, deger=c[o], n_tohum=1)
                       for o in ["dogruluk", "makro_f1", "ece", "sans_duzeyi"]])
    pd.concat([uzun, iv, vi], ignore_index=True).to_csv(OUT / "cnn_tablo_degerleri.csv", index=False)

    not_yaz(uzun, kal, kos, sonda, seu_permutasyonlari(sonda), tum_m["SEU"])
    print(f"yazildi -> {OUT}")


if __name__ == "__main__":
    main()
