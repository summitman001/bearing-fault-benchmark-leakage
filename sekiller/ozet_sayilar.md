# Özet sayılar — makale metni için manşet rakamlar

Tüm değerler `sonuc_cwru_tam/`, `sonuc_hust_tam/`, `sonuc_seu_tam/` içindeki CSV'lerden bu betikle hesaplanmıştır; hiçbiri elle girilmemiştir.

## 1) P1 ile en üst kademe arasındaki doğruluk farkı (ortalama ± sd)

Yöntem: her tohum için önce üst kademenin o tohuma ait katlarının (P4/P5) ortalaması alınır, sonra P1'in aynı tohumdaki değerinden çıkarılır (eşleşmeli/paired fark); ortalama ± sd bu 5 tohumluk fark dağılımından gelir. Kaynak: `<veri kümesi>/merdiven.csv`.

| dataset | model | üst kademe | fark (P1 − üst) ort ± sd | n(tohum) |
|---|---|---|---|---|
| CWRU | M1_CNN | P4_kosul_ayrik | +0.027 ± 0.019 | 5 |
| CWRU | M2_RF | P4_kosul_ayrik | +0.018 ± 0.003 | 5 |
| CWRU | M3_TRIVIAL | P4_kosul_ayrik | +0.014 ± 0.001 | 5 |
| CWRU | M4_KNN | P4_kosul_ayrik | +0.036 ± 0.001 | 5 |
| HUST | M1_CNN | P5_numune_ayrik | +0.517 ± 0.012 | 5 |
| HUST | M2_RF | P5_numune_ayrik | +0.573 ± 0.005 | 5 |
| HUST | M3_TRIVIAL | P5_numune_ayrik | +0.178 ± 0.004 | 5 |
| HUST | M4_KNN | P5_numune_ayrik | +0.733 ± 0.002 | 5 |
| SEU | M1_CNN | P4_kosul_ayrik | +0.539 ± 0.037 | 5 |
| SEU | M2_RF | P4_kosul_ayrik | +0.799 ± 0.000 | 5 |
| SEU | M3_TRIVIAL | P4_kosul_ayrik | +0.327 ± 0.003 | 5 |
| SEU | M4_KNN | P4_kosul_ayrik | +0.188 ± 0.001 | 5 |

## 2) M3_TRIVIAL'in P1'deki doğruluğu (her veri kümesi)

Kaynak: `<veri kümesi>/merdiven.csv`, `protokol==P1_rastgele_ortusmeli & model==M3_TRIVIAL`, `dogruluk` kolonu, 5 tohum üzerinden.

| dataset | ortalama ± sd | n(tohum) |
|---|---|---|
| CWRU | 0.697 ± 0.001 | 5 |
| HUST | 0.554 ± 0.004 | 5 |
| SEU | 0.557 ± 0.003 | 5 |

## 3) S2 (rastgele bölme) ve S2b (koşul-ayrık bölme) doğruluğu — CWRU ve HUST

Kaynak: `<veri kümesi>/sonda.csv`, `sonda` kolonu (`S2_permutasyon_rastgele_bolme` / `S2b_permutasyon_kosul_ayrik`), `dogruluk` kolonu. SEU bu tablodan hariç tutuldu: S2b'de yalnızca 1 geçerli tohum var (bkz. sekil_kaydi.md), ortalama ± sd hesaplanamaz.

| dataset | model | S2 ort ± sd (n) | S2b ort ± sd (n) |
|---|---|---|---|
| CWRU | M1_CNN | 0.987 ± 0.011 (5) | 0.080 ± 0.045 (5) |
| CWRU | M2_RF | 0.998 ± 0.001 (5) | 0.082 ± 0.080 (5) |
| CWRU | M3_TRIVIAL | 0.380 ± 0.071 (5) | 0.101 ± 0.122 (5) |
| CWRU | M4_KNN | 0.992 ± 0.002 (5) | 0.036 ± 0.048 (5) |
| HUST | M1_CNN | 0.798 ± 0.085 (5) | 0.135 ± 0.044 (5) |
| HUST | M2_RF | 0.960 ± 0.004 (5) | 0.124 ± 0.040 (5) |
| HUST | M3_TRIVIAL | 0.313 ± 0.016 (5) | 0.073 ± 0.052 (5) |
| HUST | M4_KNN | 0.878 ± 0.006 (5) | 0.129 ± 0.031 (5) |

## 4) M1_CNN ve M2_RF için önerilen k

Kaynak: `<veri kümesi>/onkayit_hukmu.csv`, `hukum==ONERILIR` olan en küçük `k` (bkz. `onerilen_k.csv`, tam tablo).

| dataset | model | önerilen en küçük k |
|---|---|---|
| CWRU | M1_CNN | 50 |
| CWRU | M2_RF | 10 |
| HUST | M1_CNN | saglanmadi |
| HUST | M2_RF | 10 |
| SEU | M1_CNN | saglanmadi |
| SEU | M2_RF | 10 |

## 5) M1_CNN'in sıralamadaki yeri: P1 ile en üst kademe arasında değişiyor mu?

Kaynak: `siralama_tersine.csv` (1=en yüksek doğruluk). `sira_degisimi = ust_sira - P1_sira`; pozitif = sıra kötüleşti.

| dataset | P1 sırası | üst kademe | üst kademe sırası | sıra değişimi |
|---|---|---|---|---|
| CWRU | 1 | P4_kosul_ayrik | 2 | +1 |
| HUST | 1 | P5_numune_ayrik | 1 | +0 |
| SEU | 3 | P4_kosul_ayrik | 2 | -1 |

Yorum: M1_CNN'in SIRASI üç veri kümesinde de büyük ölçüde korunuyor (CWRU'da 1.'den 2.'ye, SEU'da 3.'ten 2.'ye, HUST'ta 1. olarak kalıyor) — yani CNN diğer modellere göre "nispeten" iyi konumunu korur. Ama bu YANILTICI olabilir: HUST ve SEU'da CNN'in MUTLAK doğruluğu üst kademede çöküyor (bkz. Bölüm 1 — HUST'ta −0.517, SEU'da −0.539), sıralaması kötüleşmiyor çünkü DİĞER modeller de aynı ölçüde (bazen daha fazla) çöküyor. Sıra sabitliği, mutlak performans sabitliği ANLAMINA GELMEZ.
