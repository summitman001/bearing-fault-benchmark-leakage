#!/usr/bin/env python3
"""Makale sekilleri -- IEEEtran iki sutunlu makale icin.

Veri kaynagi (hicbir sayi uydurulmadi, hepsi asagidaki CSV'lerden okunur):
    sonuc_cwru_tam/{merdiven,sonda,guvenilirlik,sicaklik_k,onkayit_hukmu}.csv
    sonuc_hust/{...}.csv
    sonuc_seu/{...}.csv

Cikti: sekiller/ klasorune sekilN_ad.pdf + .png (300dpi) ve her sekil icin
cizilen degerlerin CSV'si (sekilN_veri.csv). Secimler ve gerekceler
sekil_kaydi.md'de (bu betik tarafindan degil, ayrica) belgelenir.

Tekrar calistirilabilir: yalnizca mevcut CSV'leri okur, veri uretmez.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

BFD = Path(__file__).resolve().parent
OUT = BFD / "sekiller"
OUT.mkdir(exist_ok=True)

DATASET_DIRS = {
    "CWRU": BFD / "sonuc_cwru_tam",
    "HUST": BFD / "sonuc_hust_tam",
    "SEU": BFD / "sonuc_seu_tam",
}
SD_DDOF = 1  # orneklem standart sapmasi; tablolarla ayni tanim

# ---------------------------------------------------------------- ortak stil
COLORS = {
    "M1_CNN": "#0072B2",
    "M2_RF": "#D55E00",
    "M3_TRIVIAL": "#009E73",
    "M4_KNN": "#CC79A7",
}
MODEL_ORDER = ["M1_CNN", "M2_RF", "M3_TRIVIAL", "M4_KNN"]
MODEL_LABEL = {
    "M1_CNN": "M1 (CNN)",
    "M2_RF": "M2 (RF)",
    "M3_TRIVIAL": "M3 (trivial)",
    "M4_KNN": "M4 (kNN)",
}
MARKER = {"M1_CNN": "o", "M2_RF": "s", "M3_TRIVIAL": "^", "M4_KNN": "D"}

GRID_COLOR = "#d8d8d8"
AXIS_COLOR = "#555555"
TEXT_COLOR = "#1a1a1a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS_COLOR,
    "xtick.color": AXIS_COLOR,
    "ytick.color": AXIS_COLOR,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SINGLE_COL = 3.5
DOUBLE_COL = 7.16


def style_axis(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    if ygrid:
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  yazildi: {OUT / name}.pdf / .png")


def available_models(df):
    return [m for m in MODEL_ORDER if m in df["model"].unique()]


# =====================================================================
# SEKIL 1: bolme merdiveni (ladder)
# =====================================================================
PROTOCOL_ORDER = [
    "P1_rastgele_ortusmeli",
    "P2_rastgele_ortusmesiz",
    "P3_zamansal_blok",
    "P4_kosul_ayrik",
    "P5_numune_ayrik",
]
PROTOCOL_LABEL = {
    "P1_rastgele_ortusmeli": "P1 random\n(overlap)",
    "P2_rastgele_ortusmesiz": "P2 random",
    "P3_zamansal_blok": "P3 temporal",
    "P4_kosul_ayrik": "P4 condition",
    "P5_numune_ayrik": "P5 specimen",
}
LADDER_YLIM = (0.03, 1.03)  # bkz. sekil_kaydi.md: veri minimumu ~0.078 (HUST P5)


def fig1_ladder():
    print("Sekil 1: bolme merdiveni")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6), sharey=True)
    veri_rows = []

    for ax, ds in zip(axes, ["CWRU", "HUST", "SEU"]):
        m = pd.read_csv(DATASET_DIRS[ds] / "merdiven.csv")
        models = available_models(m)
        proto_present = {proto: (m.protokol == proto).any() for proto in PROTOCOL_ORDER}
        for model in models:
            mm = m[m.model == model]
            xs, ys, es = [], [], []
            for i, proto in enumerate(PROTOCOL_ORDER):
                pp = mm[mm.protokol == proto]
                if len(pp) == 0:
                    continue
                mean_acc = pp["dogruluk"].mean()
                std_acc = pp["dogruluk"].std(ddof=SD_DDOF)
                xs.append(i)
                ys.append(mean_acc)
                es.append(std_acc)
                veri_rows.append(dict(dataset=ds, protokol=proto, model=model,
                                      n_satir=len(pp), dogruluk_ort=mean_acc,
                                      dogruluk_std=std_acc))
                # P4/P5: kat-bazinda (seed ortalamasi) jitter'li noktalar
                if proto in ("P4_kosul_ayrik", "P5_numune_ayrik"):
                    kat_means = pp.groupby("kat")["dogruluk"].mean()
                    # hash() surecler arasi degisir; crc32 her calistirmada ayni noktalari verir
                    rng = np.random.default_rng(zlib.crc32(f"{ds}|{model}|{proto}".encode()))
                    jitter = rng.uniform(-0.12, 0.12, size=len(kat_means))
                    ax.scatter(np.full(len(kat_means), i) + jitter, kat_means.values,
                              color=COLORS[model], alpha=0.35, s=10, zorder=2,
                              linewidths=0)
            ax.errorbar(xs, ys, yerr=es, color=COLORS[model], marker=MARKER[model],
                       markersize=4, linewidth=1.1, capsize=2, capthick=0.8,
                       elinewidth=0.8, label=MODEL_LABEL[model], zorder=3)

        ymid = 0.5 * (LADDER_YLIM[0] + LADDER_YLIM[1])
        for i, proto in enumerate(PROTOCOL_ORDER):
            if not proto_present[proto]:
                ax.text(i, ymid, "n/a", ha="center", va="center",
                       fontsize=7, color="#999999", style="italic")

        ax.set_xlim(-0.5, 4.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels([PROTOCOL_LABEL[p].replace("\n", " ") for p in PROTOCOL_ORDER],
                           fontsize=6.3, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylim(*LADDER_YLIM)
        ax.set_title(ds, loc="left", fontsize=8.5, fontweight="bold", pad=3, color=TEXT_COLOR)
        style_axis(ax)

    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower left", frameon=False, handlelength=1.6,
                  borderaxespad=0.1, labelspacing=0.3)
    fig.tight_layout(w_pad=1.4, rect=(0, 0.1, 1, 1))
    save(fig, "sekil1_ladder")
    pd.DataFrame(veri_rows).to_csv(OUT / "sekil1_veri.csv", index=False)


# =====================================================================
# SEKIL 2: mekanizma sondasi (probe)
# =====================================================================
PROBE_ORDER = ["S1_kayit_kimligi", "S2_permutasyon_rastgele_bolme", "S2b_permutasyon_kosul_ayrik"]
PROBE_LABEL = {
    "S1_kayit_kimligi": "S1\n(record identity)",
    "S2_permutasyon_rastgele_bolme": "S2\n(permuted, random split)",
    "S2b_permutasyon_kosul_ayrik": "S2b\n(permuted, condition-disjoint)",
}


MIN_GUVENILIR_TOHUM = 5  # bu sayidan az gecerli tohumu olan grup "not supported" olarak gosterilir


def fig2_probe():
    print("Sekil 2: mekanizma sondasi")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4), sharey=True)
    veri_rows = []
    tohum_kaydi = []  # geriye kalan her cubugun kac tohuma dayandigi (kayit icin)
    desteklenmeyen = []  # (dataset, proto) -- tamamen cikarilan gruplar

    for ax, ds in zip(axes, ["CWRU", "HUST", "SEU"]):
        s = pd.read_csv(DATASET_DIRS[ds] / "sonda.csv")
        models = available_models(s)
        n_groups = len(PROBE_ORDER)
        n_models = len(models)
        width = 0.8 / n_models

        # hangi gruplar "desteklenmiyor" (o veri kumesindeki TUM modellerin
        # gecerli tohum sayisi MIN_GUVENILIR_TOHUM'un altinda)?
        unsupported = set()
        for proto in PROBE_ORDER:
            ns_here = [len(s[(s.sonda == proto) & (s.model == m)]) for m in models]
            if ns_here and max(ns_here) < MIN_GUVENILIR_TOHUM:
                unsupported.add(proto)
                desteklenmeyen.append(dict(dataset=ds, sonda=proto,
                                          max_n_tohum=max(ns_here),
                                          neden=f"tum modellerde gecerli tohum < {MIN_GUVENILIR_TOHUM}"))

        for mi, model in enumerate(models):
            means, stds, ns = [], [], []
            for proto in PROBE_ORDER:
                if proto in unsupported:
                    means.append(np.nan); stds.append(np.nan); ns.append(0)
                    continue
                pp = s[(s.sonda == proto) & (s.model == model)]
                means.append(pp["dogruluk"].mean())
                stds.append(pp["dogruluk"].std(ddof=SD_DDOF) if len(pp) > 1 else 0.0)
                ns.append(len(pp))
                tohum_kaydi.append(dict(dataset=ds, sonda=proto, model=model, n_tohum=len(pp)))
                veri_rows.append(dict(dataset=ds, sonda=proto, model=model,
                                      n_tohum=len(pp), dogruluk_ort=pp["dogruluk"].mean(),
                                      dogruluk_std=pp["dogruluk"].std(ddof=SD_DDOF) if len(pp) > 1 else np.nan,
                                      sans_duzeyi=pp["sans_duzeyi"].mean() if len(pp) else np.nan))
            xpos = np.arange(n_groups) + (mi - (n_models - 1) / 2) * width
            err = [0 if (n <= 1) else e for e, n in zip(stds, ns)]
            ax.bar(xpos, means, width=width * 0.92, color=COLORS[model],
                  label=MODEL_LABEL[model], zorder=3,
                  yerr=err, error_kw=dict(elinewidth=0.7, capsize=1.5, capthick=0.7))
        for i, proto in enumerate(PROBE_ORDER):
            if proto in unsupported:
                ax.text(i, 0.45, "not\nsupported", fontsize=6.3, ha="center",
                       va="center", color="#999999", style="italic")

        # sans duzeyi: S1'in kendi degeri; S2 ve S2b ayni formulden geldigi
        # (1/sinif_sayisi) icin DAIMA esit cikar -- tek birlesik cizgi/etiket.
        sd = s.groupby("sonda")["sans_duzeyi"].mean()
        if "S1_kayit_kimligi" in sd.index:
            lvl = sd["S1_kayit_kimligi"]
            ax.hlines(lvl, -0.4, 0.4, colors="#888888", linestyles=(0, (3, 2)),
                      linewidth=0.9, zorder=4)
            ax.text(0.41, lvl, "chance", fontsize=5.5, color="#888888",
                   va="center", ha="left")
        s2_idx = [i for i, p in enumerate(PROBE_ORDER)
                 if p in sd.index and p != "S1_kayit_kimligi" and p not in unsupported]
        s2_vals = sd[[PROBE_ORDER[i] for i in s2_idx]].round(6).unique() if s2_idx else []
        if len(s2_vals) == 1:
            lvl2 = s2_vals[0]
            ax.hlines(lvl2, s2_idx[0] - 0.4, s2_idx[-1] + 0.4, colors="#888888",
                      linestyles=(0, (3, 2)), linewidth=0.9, zorder=4)
            ax.text((s2_idx[0] + s2_idx[-1]) / 2, lvl2 + 0.025, "chance", fontsize=5.5,
                   color="#888888", va="bottom", ha="center")
        else:
            for i in s2_idx:
                lvl2 = sd[PROBE_ORDER[i]]
                ax.hlines(lvl2, i - 0.4, i + 0.4, colors="#888888", linestyles=(0, (3, 2)),
                          linewidth=0.9, zorder=4)
                ax.text(i, lvl2 + 0.025, "chance", fontsize=5.5, color="#888888",
                       va="bottom", ha="center")

        ax.set_xlim(-0.5, n_groups - 0.5)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels([PROBE_LABEL[p].replace("\n", " ") for p in PROBE_ORDER],
                           fontsize=6.3, rotation=20, ha="right", rotation_mode="anchor")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(ds, loc="left", fontsize=8.5, fontweight="bold", pad=3, color=TEXT_COLOR)
        style_axis(ax)

    axes[0].set_ylabel("Accuracy")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.31, wspace=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.87),
              ncol=4, frameon=False, handlelength=1.2, columnspacing=1.0, fontsize=7.2)
    save(fig, "sekil2_probe")
    pd.DataFrame(veri_rows).to_csv(OUT / "sekil2_veri.csv", index=False)

    print("  kalan cubuklarin tohum sayisi (5'ten az olanlar):")
    tk = pd.DataFrame(tohum_kaydi)
    az = tk[tk.n_tohum < MIN_GUVENILIR_TOHUM]
    if len(az):
        print(az.to_string(index=False))
    else:
        print("    yok -- kalan TUM cubuklar 5 tohuma dayaniyor")
    if desteklenmeyen:
        print("  'not supported' olarak isaretlenen gruplar:")
        for d in desteklenmeyen:
            print(f"    {d}")
    return dict(tohum_kaydi=tk, desteklenmeyen=pd.DataFrame(desteklenmeyen))


# =====================================================================
# SEKIL 3: guvenilirlik egrisi (reliability diagram)
# =====================================================================
def _weighted_bins(g):
    g = g[g["n"] > 0]
    tot = g.groupby("kutu")["n"].sum()
    conf = (g.assign(wc=g["n"] * g["ort_guven"]).groupby("kutu")["wc"].sum()) / tot
    acc = (g.assign(wa=g["n"] * g["gozlenen_dogruluk"]).groupby("kutu")["wa"].sum()) / tot
    out = pd.DataFrame({"n": tot, "ort_guven": conf, "gozlenen_dogruluk": acc}).reset_index()
    return out.sort_values("ort_guven")


def _ece_from_bins(df):
    tot = df["n"].sum()
    if tot == 0:
        return float("nan")
    return float((df["n"] * (df["ort_guven"] - df["gozlenen_dogruluk"]).abs()).sum() / tot)


def hedef_ece_table():
    """Her (dataset, model, test_kosul) icin hedef_kosul ECE'sinin tohumlar
    uzerinden ortalamasi. Hem Sekil 3'un secimi hem de ece_araligi.csv
    icin ortak kaynak -- tek yerden hesaplanir, iki yerde tekrar edilmez."""
    rows = []
    for ds in ["CWRU", "HUST", "SEU"]:
        g = pd.read_csv(DATASET_DIRS[ds] / "guvenilirlik.csv")
        g = g.dropna(subset=["ort_guven", "gozlenen_dogruluk"])
        for (model, tk), gg in g[g.kosul == "hedef_kosul"].groupby(["model", "test_kosul"]):
            per_seed = gg.groupby("tohum").apply(_ece_from_bins, include_groups=False)
            rows.append(dict(dataset=ds, model=model, test_kosul=tk,
                             ece_hedef_ortalama=per_seed.mean(),
                             ece_hedef_std=per_seed.std(ddof=SD_DDOF), n_tohum=len(per_seed)))
    return pd.DataFrame(rows)


FULL_MODEL_NAME = {
    "M1_CNN": "1D-CNN",
    "M2_RF": "Random Forest",
    "M3_TRIVIAL": "Trivial baseline",
    "M4_KNN": "Envelope k-NN",
}


def _secilecek_sag_panel(tablo):
    """SAG panel secimi: model SABIT (M1_CNN tum veri kumelerinde kosmussa
    o, degilse ortak kosulan M2_RF'ye dusulur), sonra o modelin kendi
    icindeki en kotu (dataset, test_kosul) birlesimi secilir."""
    datasets_per_model = tablo.groupby("model")["dataset"].nunique()
    n_dataset_total = tablo["dataset"].nunique()
    if datasets_per_model.get("M1_CNN", 0) == n_dataset_total:
        secilen_model = "M1_CNN"
        nedeni = "M1_CNN tum veri kumelerinde kosuldu, sabit model olarak kullanildi."
    else:
        secilen_model = "M2_RF"
        kosulan = sorted(tablo[tablo.model == "M1_CNN"]["dataset"].unique().tolist())
        nedeni = (f"M1_CNN yalnizca {kosulan} veri kume(leri)nde kosuldu "
                 f"(henuz CNN'siz 'hizli' modda calistirilan veri kumeleri kaldigi icin); "
                 f"uc veri kumesinde de ortak olan M2_RF'ye dusuldu.")
    alt = tablo[tablo.model == secilen_model].sort_values("ece_hedef_ortalama", ascending=False)
    top = alt.iloc[0]
    return dict(dataset=top["dataset"], model=top["model"], test_kosul=top["test_kosul"],
               secilen_model=secilen_model, model_secim_nedeni=nedeni, tum_birlesimler=alt)


def _reliability_panel(ax, ds, model, tk, panel_title, legend=False):
    """Tek bir guvenilirlik paneli cizer; cizilen (kutu-birlesmis) veriyi
    ve ECE degerlerini dondurur (sekilN_veri.csv icin)."""
    g = pd.read_csv(DATASET_DIRS[ds] / "guvenilirlik.csv")
    sub = g[(g.model == model) & (g.test_kosul == tk)]

    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    veri_rows = []
    eces = {}
    for kosul_ad, color, label in [("ic_dogrulama", "#0072B2", "internal validation"),
                                    ("hedef_kosul", "#D55E00", "target condition")]:
        bb = _weighted_bins(sub[sub.kosul == kosul_ad])
        eces[kosul_ad] = _ece_from_bins(bb)
        sizes = 5 + 28 * (bb["n"] / bb["n"].max())
        ax.plot(bb["ort_guven"], bb["gozlenen_dogruluk"], color=color, linewidth=1.0,
               zorder=3, label=label)
        ax.scatter(bb["ort_guven"], bb["gozlenen_dogruluk"], s=sizes, color=color,
                  zorder=4, edgecolors="white", linewidths=0.4)
        for _, row in bb.iterrows():
            veri_rows.append(dict(dataset=ds, model=model, test_kosul=tk, kosul=kosul_ad,
                                  kutu=row["kutu"], n=row["n"], ort_guven=row["ort_guven"],
                                  gozlenen_dogruluk=row["gozlenen_dogruluk"]))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    if legend:
        ax.legend(loc="upper left", frameon=False, handlelength=1.2, fontsize=6.8,
                 borderaxespad=0.15)
    txt = f"ECE (internal) = {eces['ic_dogrulama']:.3f}\nECE (target) = {eces['hedef_kosul']:.3f}"
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=5.8, ha="right", va="bottom",
           color=TEXT_COLOR)
    ax.text(0.5, 1.03, f"{panel_title}  ({tk})", transform=ax.transAxes, fontsize=7.3,
           fontweight="bold", ha="center", va="bottom", color=TEXT_COLOR)
    style_axis(ax, ygrid=True)
    return veri_rows, eces


def fig3_reliability():
    print("Sekil 3: guvenilirlik egrisi (iki panel)")
    tablo = hedef_ece_table()

    # SOL panel: SABIT olarak CWRU / M1_CNN, CWRU icindeki en yuksek
    # hedef-kosul ECE'sine sahip kosul.
    sol_aday = tablo[(tablo.dataset == "CWRU") & (tablo.model == "M1_CNN")]
    if len(sol_aday) == 0:
        raise RuntimeError("CWRU icin M1_CNN guvenilirlik verisi yok -- sol panel kurulamaz.")
    sol_aday = sol_aday.sort_values("ece_hedef_ortalama", ascending=False)
    sol = dict(dataset="CWRU", model="M1_CNN", test_kosul=sol_aday.iloc[0]["test_kosul"])
    print(f"  SOL panel: CWRU / M1_CNN / {sol['test_kosul']} "
         f"(CWRU icindeki tum M1_CNN kosullari:")
    print(sol_aday[["test_kosul", "ece_hedef_ortalama"]].to_string(index=False))

    # SAG panel: model sabitlenip (M1_CNN hepsinde varsa o, yoksa M2_RF)
    # o modelin kendi icindeki en kotu birlesimi.
    sag = _secilecek_sag_panel(tablo)
    print(f"  SAG panel: model secimi {sag['secilen_model']} -- {sag['model_secim_nedeni']}")
    print(f"  SAG panel secilen: {sag['dataset']} / {sag['model']} / {sag['test_kosul']}")

    ayni_mi = (sol["dataset"] == sag["dataset"] and sol["model"] == sag["model"]
              and sol["test_kosul"] == sag["test_kosul"])
    if ayni_mi:
        print("  UYARI: sol ve sag panel AYNI birlesimi gosteriyor (M1_CNN artik tum veri "
             "kumelerinde kosmus ve CWRU'nun kendi en kotu durumu global en kotu durumla "
             "cakisiyor) -- sekil_kaydi.md'de belirtilmeli.")

    # Yan yana (3.5 x 1.9) denendi: eksen etiketleri, basliklar ve legend
    # cakisiyordu (panel basina ~1.6 inc genislik yetersiz). Bu yuzden
    # talimattaki yedek secenege gecildi: ust uste iki panel, 3.5 x 3.6.
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.6), sharex=True)
    veri_rows = []
    r1, _ = _reliability_panel(axes[0], sol["dataset"], sol["model"], sol["test_kosul"],
                               f"{sol['dataset']}, {FULL_MODEL_NAME[sol['model']]}", legend=True)
    axes[0].set_ylabel("Observed accuracy")
    axes[0].set_xlabel("")
    veri_rows += r1
    r2, _ = _reliability_panel(axes[1], sag["dataset"], sag["model"], sag["test_kosul"],
                               f"{sag['dataset']}, {FULL_MODEL_NAME[sag['model']]}", legend=False)
    axes[1].set_ylabel("Observed accuracy")
    veri_rows += r2

    fig.subplots_adjust(left=0.17, right=0.96, top=0.93, bottom=0.10, hspace=0.45)
    save(fig, "sekil3_reliability")
    pd.DataFrame(veri_rows).to_csv(OUT / "sekil3_veri.csv", index=False)
    return dict(sol_panel=sol, sag_panel=sag, ayni_birlesim_mi=ayni_mi,
               secilen_model_sag=sag["secilen_model"],
               model_secim_nedeni_sag=sag["model_secim_nedeni"])


# =====================================================================
# SEKIL 4: sicaklik olcekleme egrisi
# =====================================================================
def fig4_temperature():
    print("Sekil 4: sicaklik olcekleme egrisi")
    ds = "CWRU"  # en cok kosulu olan veri kumesi (4 kosul; HUST=3, SEU=2)
    t = pd.read_csv(DATASET_DIRS[ds] / "sicaklik_k.csv")
    hukum = pd.read_csv(DATASET_DIRS[ds] / "onkayit_hukmu.csv")

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    veri_rows = []
    models = available_models(t)
    k_pos = sorted(t[t.k > 0]["k"].unique())
    y_max_seen = 0.0

    for model in models:
        tm = t[(t.model == model) & (t.k > 0)]
        meds, q25, q75 = [], [], []
        for k in k_pos:
            vals = tm[tm.k == k]["ece"].values
            meds.append(np.median(vals))
            q25.append(np.percentile(vals, 25))
            q75.append(np.percentile(vals, 75))
            veri_rows.append(dict(dataset=ds, model=model, k=k, ece_medyan=np.median(vals),
                                  ece_q25=np.percentile(vals, 25), ece_q75=np.percentile(vals, 75),
                                  n=len(vals)))
        ax.plot(k_pos, meds, color=COLORS[model], marker=MARKER[model], markersize=3.5,
               linewidth=1.1, label=MODEL_LABEL[model], zorder=3)
        ax.fill_between(k_pos, q25, q75, color=COLORS[model], alpha=0.18, linewidth=0, zorder=2)

        base_vals = t[(t.model == model) & (t.k == 0)]["ece"].values
        base_med = np.median(base_vals)
        ax.hlines(base_med, min(k_pos), max(k_pos), colors=COLORS[model],
                 linestyles=(0, (1, 1)), linewidth=1.0, zorder=1)
        veri_rows.append(dict(dataset=ds, model=model, k=0, ece_medyan=base_med,
                              ece_q25=np.nan, ece_q75=np.nan, n=len(base_vals)))
        y_max_seen = max(y_max_seen, base_med, max(q75))

    # onerilen k: ana model = M1_CNN
    ana_model = "M1_CNN" if "M1_CNN" in models else models[0]
    oner = hukum[(hukum.model == ana_model) & (hukum.hukum == "ONERILIR")]
    k_rec = int(oner["k"].min()) if len(oner) else None

    ax.set_xscale("log")
    ax.set_xticks(k_pos)
    ax.set_xticklabels([str(k) for k in k_pos])
    ax.set_xlabel("k (labeled calibration-set size)")
    ax.set_ylabel("ECE")
    ax.set_ylim(0, y_max_seen * 1.18)
    if k_rec is not None:
        y_bot = ax.get_ylim()[0]
        ax.axvline(k_rec, color="#666666", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
        ax.text(k_rec, y_bot + 0.03 * (ax.get_ylim()[1] - y_bot), "recommended k",
               fontsize=6, color="#666666", ha="right", va="bottom", rotation=90,
               bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    style_axis(ax, ygrid=True)
    fig.subplots_adjust(left=0.17, right=0.78, top=0.97, bottom=0.17)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.03), frameon=False,
             handlelength=1.3, fontsize=6.6, borderaxespad=0.0)
    save(fig, "sekil4_temperature")
    pd.DataFrame(veri_rows).to_csv(OUT / "sekil4_veri.csv", index=False)

    # model bazinda farkli onerilen k'lari raporla
    print("  onerilen en kucuk k (ONERILIR), model bazinda:")
    rec_by_model = {}
    for model in models:
        om = hukum[(hukum.model == model) & (hukum.hukum == "ONERILIR")]
        rec_by_model[model] = int(om["k"].min()) if len(om) else None
        print(f"    {model}: {rec_by_model[model]}")
    return dict(dataset=ds, ana_model=ana_model, k_rec_ana_model=k_rec, rec_by_model=rec_by_model)



# =====================================================================
# EK CIKTILAR: metin icin sayilar (sekiller disi, ayri CSV/MD)
# =====================================================================
TOP_TIER_PROTOCOL = {"CWRU": "P4_kosul_ayrik", "HUST": "P5_numune_ayrik", "SEU": "P4_kosul_ayrik"}


def ek_ece_araligi(hedef_tablo):
    """1) ece_araligi.csv -- hedef kosulda ECE'nin tum (dataset,model,kosul)
    birlesimlerindeki degerleri + min/medyan/maks ozet satirlari."""
    df = hedef_tablo.sort_values("ece_hedef_ortalama").reset_index(drop=True)
    full = df.assign(tip="").rename(columns={"ece_hedef_ortalama": "ece_hedef_ort"})
    full = full[["tip", "dataset", "model", "test_kosul", "ece_hedef_ort", "ece_hedef_std", "n_tohum"]]

    min_row = df.loc[df.ece_hedef_ortalama.idxmin()]
    med_val = df.ece_hedef_ortalama.median()
    med_row = df.iloc[(df.ece_hedef_ortalama - med_val).abs().idxmin()]
    max_row = df.loc[df.ece_hedef_ortalama.idxmax()]
    ozet = pd.DataFrame([
        dict(tip="MIN", dataset=min_row.dataset, model=min_row.model, test_kosul=min_row.test_kosul,
             ece_hedef_ort=min_row.ece_hedef_ortalama, ece_hedef_std=min_row.ece_hedef_std,
             n_tohum=min_row.n_tohum),
        dict(tip="MEDYAN", dataset=med_row.dataset, model=med_row.model, test_kosul=med_row.test_kosul,
             ece_hedef_ort=med_val, ece_hedef_std=med_row.ece_hedef_std, n_tohum=med_row.n_tohum),
        dict(tip="MAKS", dataset=max_row.dataset, model=max_row.model, test_kosul=max_row.test_kosul,
             ece_hedef_ort=max_row.ece_hedef_ortalama, ece_hedef_std=max_row.ece_hedef_std,
             n_tohum=max_row.n_tohum),
    ])
    out = pd.concat([full, ozet], ignore_index=True)
    out.to_csv(OUT / "ece_araligi.csv", index=False)
    print(f"  yazildi: {OUT / 'ece_araligi.csv'}  (min={min_row.ece_hedef_ortalama:.3f} "
         f"[{min_row.dataset}/{min_row.model}/{min_row.test_kosul}], "
         f"medyan={med_val:.3f}, maks={max_row.ece_hedef_ortalama:.3f} "
         f"[{max_row.dataset}/{max_row.model}/{max_row.test_kosul}])")
    return out


def ek_onerilen_k():
    """2) onerilen_k.csv -- her (dataset,model) icin ONERILIR olcutunu
    saglayan EN KUCUK k; hicbiri saglamiyorsa 'saglanmadi'."""
    rows = []
    for ds, ddir in DATASET_DIRS.items():
        path = ddir / "onkayit_hukmu.csv"
        h = pd.read_csv(path)
        for model in sorted(h.model.unique()):
            hm = h[(h.model == model) & (h.hukum == "ONERILIR")]
            rows.append(dict(dataset=ds, model=model,
                             onerilen_en_kucuk_k=str(int(hm.k.min())) if len(hm) else "saglanmadi",
                             kaynak_dosya=f"{ddir.name}/onkayit_hukmu.csv"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "onerilen_k.csv", index=False)
    print(f"  yazildi: {OUT / 'onerilen_k.csv'}")
    return out


def ek_siralama_tersine():
    """3) siralama_tersine.csv -- P1 siralamasi vs en ust kademe siralamasi."""
    rows = []
    for ds, top_proto in TOP_TIER_PROTOCOL.items():
        m = pd.read_csv(DATASET_DIRS[ds] / "merdiven.csv")
        models = available_models(m)
        p1_acc = {mod: m[(m.model == mod) & (m.protokol == "P1_rastgele_ortusmeli")]["dogruluk"].mean()
                  for mod in models}
        top_acc = {mod: m[(m.model == mod) & (m.protokol == top_proto)]["dogruluk"].mean()
                   for mod in models}
        p1_rank = pd.Series(p1_acc).rank(ascending=False, method="min").astype(int)
        top_rank = pd.Series(top_acc).rank(ascending=False, method="min").astype(int)
        for mod in models:
            rows.append(dict(dataset=ds, model=mod,
                             P1_dogruluk=p1_acc[mod], P1_sira=int(p1_rank[mod]),
                             ust_kademe=top_proto, ust_dogruluk=top_acc[mod],
                             ust_sira=int(top_rank[mod]),
                             sira_degisimi=int(top_rank[mod] - p1_rank[mod])))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "siralama_tersine.csv", index=False)
    print(f"  yazildi: {OUT / 'siralama_tersine.csv'}")

    cnn = out[out.model == "M1_CNN"]
    if len(cnn):
        print("  M1_CNN siralama degisimi (P1 -> ust kademe):")
        print(cnn[["dataset", "P1_sira", "ust_kademe", "ust_sira", "sira_degisimi"]]
             .to_string(index=False))
    return out


def _p1_vs_top_gap(ds):
    """Tohum-eslesmeli (paired) P1 - ust_kademe farki: her tohum icin
    once ust kademenin o tohuma ait katlarinin ortalamasi alinir, sonra
    P1'in ayni tohumdaki degerinden cikarilir. Boylece fark_std uydurma
    degil, 5 tohumun gercek dagilimindan gelir."""
    m = pd.read_csv(DATASET_DIRS[ds] / "merdiven.csv")
    top_proto = TOP_TIER_PROTOCOL[ds]
    rows = []
    for model in available_models(m):
        p1 = m[(m.model == model) & (m.protokol == "P1_rastgele_ortusmeli")].set_index("tohum")["dogruluk"]
        top_by_seed = (m[(m.model == model) & (m.protokol == top_proto)]
                       .groupby("tohum")["dogruluk"].mean())
        common = sorted(set(p1.index) & set(top_by_seed.index))
        gaps = np.array([p1[s] - top_by_seed[s] for s in common])
        rows.append(dict(dataset=ds, model=model, ust_kademe=top_proto,
                         n_tohum_eslesen=len(common),
                         fark_ortalama=gaps.mean() if len(gaps) else np.nan,
                         fark_std=gaps.std(ddof=SD_DDOF) if len(gaps) else np.nan,
                         P1_dogruluk_ort=p1.mean(), ust_dogruluk_ort=top_by_seed.mean()))
    return pd.DataFrame(rows)


def ek_ozet_sayilar(onerilen_k_df, siralama_df):
    """4) ozet_sayilar.md -- metinde dogrudan kullanilacak manset sayilar."""
    klasorler = ", ".join(f"`{d.name}/`" for d in DATASET_DIRS.values())
    lines = ["# Özet sayılar — makale metni için manşet rakamlar", "",
            f"Tüm değerler {klasorler} içindeki "
            "CSV'lerden bu betikle hesaplanmıştır; hiçbiri elle girilmemiştir.", ""]

    lines.append("## 1) P1 ile en üst kademe arasındaki doğruluk farkı (ortalama ± sd)")
    lines.append("")
    lines.append("Yöntem: her tohum için önce üst kademenin o tohuma ait katlarının "
                 "(P4/P5) ortalaması alınır, sonra P1'in aynı tohumdaki değerinden "
                 "çıkarılır (eşleşmeli/paired fark); ortalama ± sd bu 5 tohumluk fark "
                 "dağılımından gelir. Kaynak: `<veri kümesi>/merdiven.csv`.")
    lines.append("")
    lines.append("| dataset | model | üst kademe | fark (P1 − üst) ort ± sd | n(tohum) |")
    lines.append("|---|---|---|---|---|")
    for ds in TOP_TIER_PROTOCOL:
        gaps = _p1_vs_top_gap(ds)
        for _, r in gaps.iterrows():
            lines.append(f"| {r.dataset} | {r.model} | {r.ust_kademe} | "
                         f"{r.fark_ortalama:+.3f} ± {r.fark_std:.3f} | {r.n_tohum_eslesen} |")
    lines.append("")

    lines.append("## 2) M3_TRIVIAL'in P1'deki doğruluğu (her veri kümesi)")
    lines.append("")
    lines.append("Kaynak: `<veri kümesi>/merdiven.csv`, `protokol==P1_rastgele_ortusmeli & "
                 "model==M3_TRIVIAL`, `dogruluk` kolonu, 5 tohum üzerinden.")
    lines.append("")
    lines.append("| dataset | ortalama ± sd | n(tohum) |")
    lines.append("|---|---|---|")
    for ds in DATASET_DIRS:
        m = pd.read_csv(DATASET_DIRS[ds] / "merdiven.csv")
        p1 = m[(m.model == "M3_TRIVIAL") & (m.protokol == "P1_rastgele_ortusmeli")]["dogruluk"]
        if len(p1):
            lines.append(f"| {ds} | {p1.mean():.3f} ± {p1.std(ddof=SD_DDOF):.3f} | {len(p1)} |")
        else:
            lines.append(f"| {ds} | DOGRULANAMADI: M3_TRIVIAL bu veri kumesinde P1'de kosulmamis | 0 |")
    lines.append("")

    lines.append("## 3) S2 (rastgele bölme) ve S2b (koşul-ayrık bölme) doğruluğu — CWRU ve HUST")
    lines.append("")
    lines.append("Kaynak: `<veri kümesi>/sonda.csv`, `sonda` kolonu "
                 "(`S2_permutasyon_rastgele_bolme` / `S2b_permutasyon_kosul_ayrik`), "
                 "`dogruluk` kolonu. SEU bu tablodan hariç tutuldu: S2b'de yalnızca "
                 "1 geçerli tohum var (bkz. sekil_kaydi.md), ortalama ± sd hesaplanamaz.")
    lines.append("")
    lines.append("| dataset | model | S2 ort ± sd (n) | S2b ort ± sd (n) |")
    lines.append("|---|---|---|---|")
    for ds in ["CWRU", "HUST"]:
        s = pd.read_csv(DATASET_DIRS[ds] / "sonda.csv")
        for model in available_models(s):
            s2 = s[(s.model == model) & (s.sonda == "S2_permutasyon_rastgele_bolme")]["dogruluk"]
            s2b = s[(s.model == model) & (s.sonda == "S2b_permutasyon_kosul_ayrik")]["dogruluk"]
            lines.append(f"| {ds} | {model} | {s2.mean():.3f} ± {s2.std(ddof=SD_DDOF):.3f} ({len(s2)}) | "
                         f"{s2b.mean():.3f} ± {s2b.std(ddof=SD_DDOF):.3f} ({len(s2b)}) |")
    lines.append("")

    lines.append("## 4) M1_CNN ve M2_RF için önerilen k")
    lines.append("")
    lines.append("Kaynak: `<veri kümesi>/onkayit_hukmu.csv`, `hukum==ONERILIR` olan en küçük "
                 "`k` (bkz. `onerilen_k.csv`, tam tablo).")
    lines.append("")
    lines.append("| dataset | model | önerilen en küçük k |")
    lines.append("|---|---|---|")
    for _, r in onerilen_k_df[onerilen_k_df.model.isin(["M1_CNN", "M2_RF"])].iterrows():
        lines.append(f"| {r.dataset} | {r.model} | {r.onerilen_en_kucuk_k} |")
    lines.append("")

    lines.append("## 5) M1_CNN'in sıralamadaki yeri: P1 ile en üst kademe arasında değişiyor mu?")
    lines.append("")
    lines.append("Kaynak: `siralama_tersine.csv` (1=en yüksek doğruluk). "
                 "`sira_degisimi = ust_sira - P1_sira`; pozitif = sıra kötüleşti.")
    lines.append("")
    lines.append("| dataset | P1 sırası | üst kademe | üst kademe sırası | sıra değişimi |")
    lines.append("|---|---|---|---|---|")
    cnn = siralama_df[siralama_df.model == "M1_CNN"]
    for _, r in cnn.iterrows():
        lines.append(f"| {r.dataset} | {r.P1_sira} | {r.ust_kademe} | {r.ust_sira} | "
                     f"{r.sira_degisimi:+d} |")
    lines.append("")
    lines.append("Yorum: M1_CNN'in SIRASI üç veri kümesinde de büyük ölçüde korunuyor "
                 "(CWRU'da 1.'den 2.'ye, SEU'da 3.'ten 2.'ye, HUST'ta 1. olarak kalıyor) — "
                 "yani CNN diğer modellere göre \"nispeten\" iyi konumunu korur. Ama bu "
                 "YANILTICI olabilir: HUST ve SEU'da CNN'in MUTLAK doğruluğu üst kademede "
                 "çöküyor (bkz. Bölüm 1 — HUST'ta −0.517, SEU'da −0.539), sıralaması "
                 "kötüleşmiyor çünkü DİĞER modeller de aynı ölçüde (bazen daha fazla) "
                 "çöküyor. Sıra sabitliği, mutlak performans sabitliği ANLAMINA GELMEZ.")
    lines.append("")

    (OUT / "ozet_sayilar.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  yazildi: {OUT / 'ozet_sayilar.md'}")


if __name__ == "__main__":
    fig1_ladder()
    fig2_info = fig2_probe()
    fig3_info = fig3_reliability()
    fig4_info = fig4_temperature()

    print("\nEk ciktilar:")
    hedef_tablo = hedef_ece_table()
    ek_ece_araligi(hedef_tablo)
    onerilen_k_df = ek_onerilen_k()
    siralama_df = ek_siralama_tersine()
    ek_ozet_sayilar(onerilen_k_df, siralama_df)

    print("\n--- OZET (sekil_kaydi.md icin) ---")
    print("Sekil 3 sol panel:", fig3_info["sol_panel"])
    print("Sekil 3 sag panel:", {k: v for k, v in fig3_info["sag_panel"].items()
                                 if k != "tum_birlesimler"})
    print("Sekil 3 ayni birlesim mi:", fig3_info["ayni_birlesim_mi"])
    print("Sekil 4 secimi:", fig4_info)
    print("\nBitti.")
