# M1_CNN — eksik tablo değerleri

Yeni koşu yapılmadı. Tüm değerler `cnn_tablolari.py` ile `sonuc_*_tam/` CSV'lerinden hesaplandı; tam duyarlıklı değerler `cnn_tablo_degerleri.csv`. Ortalama ± sd: sd tohumlar arasıdır (`ddof=1`), P4/P5'te önce katlar üzerinden ortalama alınır. CWRU, tanımları tablodaki dolu hücrelerle karşılaştırmak için referans olarak eklendi.

## Tablo III — ECE (ortalama ± sd)

| dataset | P1 | P2 | P3 | P4 | P5 |
|---|---:|---:|---:|---:|---:|
| CWRU | 0.0003 ± 0.0000 | 0.0067 ± 0.0009 | 0.0073 ± 0.0007 | 0.0313 ± 0.0051 | yok |
| HUST | 0.0010 ± 0.0006 | 0.0167 ± 0.0054 | 0.0143 ± 0.0072 | 0.0244 ± 0.0069 | 0.3557 ± 0.0206 |
| SEU | 0.0019 ± 0.0004 | 0.0396 ± 0.0251 | 0.0243 ± 0.0109 | 0.4680 ± 0.0383 | yok |

## Tablo V — ortalama güven, güven farkı, NLL, Brier (ortalama ± sd)

Güven farkı = `guven_eksi_dogruluk` (ortalama güven − doğruluk); pozitif değer aşırı güven demektir. P4 satırları `sicaklik_k.csv`'nin k=0 satırlarıyla birebir aynıdır.

| dataset | protokol | ort. güven | güven farkı | NLL | Brier |
|---|---|---:|---:|---:|---:|
| CWRU | P1_rastgele_ortusmeli | 0.9997 ± 0.0000 | -0.0003 ± 0.0000 | 0.0003 ± 0.0000 | 0.0000 ± 0.0000 |
| CWRU | P2_rastgele_ortusmesiz | 0.9933 ± 0.0009 | -0.0067 ± 0.0009 | 0.0070 ± 0.0010 | 0.0005 ± 0.0003 |
| CWRU | P3_zamansal_blok | 0.9927 ± 0.0007 | -0.0073 ± 0.0007 | 0.0074 ± 0.0008 | 0.0002 ± 0.0001 |
| CWRU | P4_kosul_ayrik | 0.9578 ± 0.0071 | -0.0149 ± 0.0136 | 0.0729 ± 0.0263 | 0.0359 ± 0.0164 |
| HUST | P1_rastgele_ortusmeli | 0.9990 ± 0.0006 | -0.0010 ± 0.0006 | 0.0010 ± 0.0006 | 0.0001 ± 0.0000 |
| HUST | P2_rastgele_ortusmesiz | 0.9830 ± 0.0052 | -0.0159 ± 0.0056 | 0.0183 ± 0.0053 | 0.0031 ± 0.0008 |
| HUST | P3_zamansal_blok | 0.9856 ± 0.0071 | -0.0140 ± 0.0073 | 0.0152 ± 0.0075 | 0.0022 ± 0.0011 |
| HUST | P4_kosul_ayrik | 0.9541 ± 0.0081 | -0.0198 ± 0.0090 | 0.1032 ± 0.0163 | 0.0445 ± 0.0076 |
| HUST | P5_numune_ayrik | 0.8355 ± 0.0065 | 0.3524 ± 0.0174 | 2.1096 ± 0.1796 | 0.8168 ± 0.0249 |
| SEU | P1_rastgele_ortusmeli | 0.9968 ± 0.0005 | -0.0014 ± 0.0004 | 0.0044 ± 0.0008 | 0.0022 ± 0.0005 |
| SEU | P2_rastgele_ortusmesiz | 0.9426 ± 0.0427 | -0.0393 ± 0.0254 | 0.0823 ± 0.0690 | 0.0357 ± 0.0377 |
| SEU | P3_zamansal_blok | 0.9757 ± 0.0109 | -0.0243 ± 0.0109 | 0.0262 ± 0.0123 | 0.0049 ± 0.0037 |
| SEU | P4_kosul_ayrik | 0.8879 ± 0.0143 | 0.4286 ± 0.0410 | 3.0532 ± 0.2960 | 0.9852 ± 0.0332 |

## Tablo VI — k başına ECE, iyileşme oranı, medyan T

Kaynak: `onkayit_hukmu.csv` (ECE ve iyileşme oranı; `sicaklik_k.csv`'den yeniden üretilip eşitliği doğrulandı) ve `sicaklik_k.csv` (`sicaklik` medyanı). Havuz: tüm tohum × hedef koşul × 20 çekiliş. Eşik: iyileşme oranı ≥ 0.90.

| dataset | k | gerçek n_kal | n_çekiliş | ECE₀ medyan | ECE medyan | iyileşme oranı | medyan T | T çeyrekleri | tekrar eden T oranı | hüküm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CWRU | 10 | 10 | 400 | 0.0261 | 0.0041 | 0.855 | 0.206 | 0.117–0.206 | 0.500 | ONERILMEZ |
| CWRU | 20 | 20 | 400 | 0.0261 | 0.0038 | 0.892 | 0.206 | 0.115–0.314 | 0.532 | ONERILMEZ |
| CWRU | 50 | 50 | 400 | 0.0261 | 0.0021 | 0.915 | 0.206 | 0.120–0.380 | 0.502 | ONERILIR |
| CWRU | 100 | 100 | 400 | 0.0261 | 0.0027 | 0.917 | 0.206 | 0.120–0.403 | 0.405 | ONERILIR |
| CWRU | 200 | 200 | 400 | 0.0261 | 0.0032 | 0.917 | 0.206 | 0.120–0.447 | 0.350 | ONERILIR |
| CWRU | 500 | 500 | 400 | 0.0261 | 0.0015 | 0.875 | 0.206 | 0.120–0.409 | 0.360 | ONERILMEZ |
| HUST | 10 | 7 | 300 | 0.0231 | 0.0113 | 0.607 | 0.202 | 0.115–0.206 | 0.380 | ONERILMEZ |
| HUST | 20 | 14 | 300 | 0.0231 | 0.0110 | 0.643 | 0.120 | 0.085–0.381 | 0.283 | ONERILMEZ |
| HUST | 50 | 49 | 300 | 0.0231 | 0.0071 | 0.733 | 0.206 | 0.085–0.846 | 0.133 | ONERILMEZ |
| HUST | 100 | 98 | 300 | 0.0231 | 0.0067 | 0.760 | 0.344 | 0.068–0.877 | 0.120 | ONERILMEZ |
| HUST | 200 | 196 | 300 | 0.0231 | 0.0065 | 0.790 | 0.456 | 0.068–0.853 | 0.140 | ONERILMEZ |
| HUST | 500 | 497 | 300 | 0.0231 | 0.0061 | 0.803 | 0.531 | 0.060–0.870 | 0.163 | ONERILMEZ |
| SEU | 10 | 10 | 200 | 0.4363 | 0.2700 | 0.795 | 5.215 | 4.336–6.697 | 0.000 | ONERILMEZ |
| SEU | 20 | 20 | 200 | 0.4363 | 0.2709 | 0.795 | 5.275 | 4.316–6.586 | 0.000 | ONERILMEZ |
| SEU | 50 | 50 | 200 | 0.4363 | 0.2685 | 0.795 | 5.272 | 4.323–6.526 | 0.000 | ONERILMEZ |
| SEU | 100 | 100 | 200 | 0.4363 | 0.2691 | 0.800 | 5.305 | 4.331–6.474 | 0.000 | ONERILMEZ |
| SEU | 200 | 200 | 200 | 0.4363 | 0.2695 | 0.800 | 5.248 | 4.320–6.450 | 0.000 | ONERILMEZ |
| SEU | 500 | 500 | 200 | 0.4363 | 0.2703 | 0.800 | 5.273 | 4.313–6.414 | 0.000 | ONERILMEZ |

- **Gerçek kalibrasyon örneği k'dan az olabilir:** `calib.temperature_curve` sınıf başına `k // sınıf_sayısı` örnek seçiyor. `gerçek n_kal` sütunu fiili sayıdır; HUST'ta 7 sınıf olduğu için nominal k'dan küçüktür.
- **Hüküm ile medyan ECE ayrı şeyler:** kural çekiliş düzeyinde iyileşme oranına bakar. Medyan ECE düşse de oran 0.90'ın altında kalabilir; "kural sağlanmadı" "ECE düşmedi" anlamına gelmez.
- **CWRU ve HUST'ta medyan T dikkatle yazılmalı:** `tekrar eden T oranı`, aynı T değerinin (4 ondalık) o veri kümesinde en az 10 çekilişte çıktığı çekilişlerin payıdır. Sürekli bir kestirimde tekrar beklenmez. En sık değer 0.2057, `fit_temperature` aramasının izidir: tamamı doğru ve çok güvenli yapay örneklerle (marj 1e-6 ve altı, n=10 ve 50) fonksiyon model eğitilmeden çağrıldığında tam olarak 0.2057 döndürüyor, çünkü NLL düzleşiyor ve arama T'yi belirleyemiyor. Bu çekilişlerde T bir kestirim değildir. Diğer tekrar eden değerler 0.1198 ve alt sınır 0.05'tir; 0.1198 birebir yeniden üretilmedi (marj 1e-3'te 0.1190). SEU'da tekrar yoktur ve T üst sınıra (20) ulaşmaz.

Hedef koşula göre (tüm k > 0 çekilişleri birlikte):

| dataset | hedef koşul | ECE₀ medyan | ECE medyan | iyileşme oranı | medyan T |
|---|---|---:|---:|---:|---:|
| CWRU | 0hp | 0.0490 | 0.0151 | 0.947 | 0.352 |
| CWRU | 1hp | 0.0086 | 0.0000 | 1.000 | 0.120 |
| CWRU | 2hp | 0.0050 | 0.0000 | 1.000 | 0.206 |
| CWRU | 3hp | 0.0577 | 0.0392 | 0.635 | 0.806 |
| HUST | 0W | 0.0208 | 0.0249 | 0.320 | 0.892 |
| HUST | 200W | 0.0165 | 0.0002 | 1.000 | 0.115 |
| HUST | 400W | 0.0286 | 0.0071 | 0.848 | 0.206 |
| SEU | 20Hz-0V | 0.5865 | 0.1814 | 1.000 | 6.499 |
| SEU | 30Hz-2V | 0.3220 | 0.3018 | 0.595 | 4.323 |

Havuzlanmış medyan düşüşü bütün koşullar için geçerli değil; `kural sağlanmadı` hükmünün kaynağı belirli koşullardır.

Tohum × hedef koşul birimlerine göre iyileşme (her birimde k başına 20 çekiliş):

| dataset | k | birim | hiç iyileşmeyen | her çekilişte iyileşen |
|---|---:|---:|---:|---:|
| HUST | 10 | 15 | 3 | 6 |
| HUST | 20 | 15 | 1 | 7 |
| HUST | 50 | 15 | 0 | 8 |
| HUST | 100 | 15 | 0 | 8 |
| HUST | 200 | 15 | 0 | 10 |
| HUST | 500 | 15 | 0 | 10 |
| SEU | 10 | 10 | 2 | 7 |
| SEU | 20 | 10 | 2 | 7 |
| SEU | 50 | 10 | 2 | 7 |
| SEU | 100 | 10 | 2 | 8 |
| SEU | 200 | 10 | 2 | 8 |
| SEU | 500 | 10 | 2 | 8 |

## Tablo IV — SEU S2b, M1_CNN

**Doğruluk 0.5982** (makro F1 0.4655, ECE 0.3476), şans düzeyi 0.200, n_test 10220. **n = 1 tohum** (tohum 0); sd hesaplanamaz. Kaynak: `sonuc_seu_tam/sonda.csv`.

| model | doğruluk | makro F1 | ECE |
|---|---:|---:|---:|
| M1_CNN | 0.5982 | 0.4655 | 0.3476 |
| M2_RF | 0.2001 | 0.0731 | 0.5696 |
| M3_TRIVIAL | 0.1986 | 0.0692 | 0.7668 |
| M4_KNN | 0.5993 | 0.5316 | 0.3897 |

### Bu değer şans düzeyiyle karşılaştırılmamalı

Etiket permütasyonu model eğitilmeden yeniden kuruldu (`etiketi_degisen_oran` ve geçerli tohum CSV ile birebir uyuşuyor; betikte `assert`). S2b testi tek kattır (test koşulu 30Hz-2V, 5 kayıt).

| tohum | etiketi değişen oran | geçerli kat mı | testte, aynı gerçek sınıfın eğitimdeki etiketini taşıyan kayıt |
|---:|---:|---|---:|
| 0 | 0.90 | evet | 3/5 |
| 1 | 0.70 | hayır | 0/5 |
| 2 | 0.90 | hayır | 0/5 |
| 3 | 0.90 | hayır | 0/5 |
| 4 | 0.70 | hayır | 0/5 |

Geçerli olan tek tohumda (0), test kayıtlarının 3/5'i aynı gerçek sınıfın eğitimdeki etiketini koruyor. Yani bu tohumda permütasyon koşullar arasında sınıf bilgisini yok etmiyor. Gerçek arıza sınıfını koşullar arasında genelleyen bir model, bu 3 sınıfı tanıyorsa bu etiketlerle en çok 0.60 alır. Aynı test koşulunda gerçek etiketli P4 doğruluğu bu okumayla uyumlu:

| model | P4 gerçek etiket, tohum 0 | S2b permütasyon, tohum 0 |
|---|---:|---:|
| M1_CNN | 0.5898 | 0.5982 |
| M2_RF | 0.2000 | 0.2001 |
| M3_TRIVIAL | 0.2000 | 0.1986 |
| M4_KNN | 0.8000 | 0.5993 |

Gerçek etiketle şans düzeyinde kalan M2_RF ve M3_TRIVIAL S2b'de de şans düzeyinde; gerçek etiketle iyi olan M4_KNN ve M1_CNN 0.60 sınırında. Model tahminleri `sonda.csv`'de bulunmadığı için bu doğrudan gösterilmiş değil, verilerle tutarlı bir açıklamadır; `RAPOR.md`'nin M4_KNN için önerdiği kayıt "parmak izi" açıklamasından daha basittir.

Ayrıca geçerlilik filtresi (`gecerli_kat`: testteki her etiket eğitimde de olmalı) seçim etkisi yaratabilir: filtreden geçen tek tohum, etiket uyumu olan tek tohumdur. Bu nedenle değer, S2b'nin "şans düzeyine iniş" bulgusunu SEU için ne destekler ne de çürütür.

Tablo IV için önerilen dipnot:

> SEU S2b: n = 1 seed (0.598); the other four permutations gave no valid fold. In this permutation 3 of 5 test recordings keep the label of the same true class in training, so the attainable accuracy for a model that generalises across conditions is 0.60, not the nominal chance level of 0.20.
