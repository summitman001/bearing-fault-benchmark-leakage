# Dağılım notu — katlar arası ve tohumlar arası standart sapma

Yeni koşu yapılmadı. Tüm değerler `dagilim.py` ile `sonuc_cwru_tam/`, `sonuc_hust_tam/`, `sonuc_seu_tam/` içindeki `merdiven.csv` dosyalarından hesaplandı; tam tablo `dagilim_ozeti.csv`. Standart sapmalar `sekiller.py` ile aynı tanımla (`ddof=1`).

- `katlar_arasi_sd`: önce her kat için tohumlar üzerinden ortalama, sonra bu kat ortalamalarının sd'si.
- `tohumlar_arasi_sd`: önce her tohum için katlar üzerinden ortalama, sonra bu tohum ortalamalarının sd'si.
- `oran` = katlar arası sd / tohumlar arası sd.
- P1, P2, P3 tek katlıdır (`kat` = `-`, `tek_katli=True`); yalnızca tohumlar arası sd hesaplandı, `ortalama` ve kat kolonları bu satırlarda boştur.
- P4/P5'te 48 satır var (16 dataset×model×protokol birleşimi × 3 ölçüt). Her hücre dengeli: 5 tohum × tüm katlar, eksik veya yinelenen satır yok.

## 1) Katlar arası sd'nin tohumlar arası sd'nin 2 katından büyük olduğu birleşimler

Oran 24 satırda tanımlı; bunların 23 tanesinde 2'den büyük, 1 tanesinde değil. 24 satırda tohumlar arası sd sıfır olduğu için oran tanımsız (aşağıda ayrı listelendi).

Oranı 2'yi aşan satırlar 8 birleşime, oranı tanımsız satırlar 8 birleşime yayılıyor. Birleşim × ölçüt özeti ("tanımsız" = tohumlar arası sd sıfır):

| dataset | model | protokol | kat sayısı | dogruluk | makro_f1 | ece |
|---|---|---|---:|---:|---:|---:|
| CWRU | M1_CNN | P4_kosul_ayrik | 4 | 2.1 | 2.1 | 5.4 |
| CWRU | M2_RF | P4_kosul_ayrik | 4 | 11.0 | 9.7 | 16.6 |
| CWRU | M3_TRIVIAL | P4_kosul_ayrik | 4 | tanımsız | tanımsız | tanımsız |
| CWRU | M4_KNN | P4_kosul_ayrik | 4 | tanımsız | tanımsız | tanımsız |
| HUST | M1_CNN | P4_kosul_ayrik | 3 | 7.5 | 8.1 | 0.7 |
| HUST | M1_CNN | P5_numune_ayrik | 5 | 18.3 | 12.0 | 9.8 |
| HUST | M2_RF | P4_kosul_ayrik | 3 | 161.1 | 204.0 | 29.4 |
| HUST | M2_RF | P5_numune_ayrik | 5 | 15.2 | 25.6 | 10.6 |
| HUST | M3_TRIVIAL | P4_kosul_ayrik | 3 | tanımsız | tanımsız | tanımsız |
| HUST | M3_TRIVIAL | P5_numune_ayrik | 5 | tanımsız | tanımsız | tanımsız |
| HUST | M4_KNN | P4_kosul_ayrik | 3 | tanımsız | tanımsız | tanımsız |
| HUST | M4_KNN | P5_numune_ayrik | 5 | tanımsız | tanımsız | tanımsız |
| SEU | M1_CNN | P4_kosul_ayrik | 2 | 3.8 | 5.0 | 4.7 |
| SEU | M2_RF | P4_kosul_ayrik | 2 | 3.4 | 8.9 | 14.5 |
| SEU | M3_TRIVIAL | P4_kosul_ayrik | 2 | tanımsız | tanımsız | tanımsız |
| SEU | M4_KNN | P4_kosul_ayrik | 2 | tanımsız | tanımsız | tanımsız |

Okurken dikkat edilecekler:

- Katlar arası sd kat sayısı kadar değerden hesaplanır: CWRU P4=4, HUST P4=3, HUST P5=5, SEU P4=2. Özellikle SEU'da iki değerin sd'sidir.
- Oran, mutlak büyüklüğü göstermez; bu yüzden aşağıdaki tablolarda kat aralığı da verildi. Örneğin SEU M2_RF P4 dogruluk satırında oran 2'yi aşıyor ama iki sd de 0.002'nin altında.

### Oranı 2'yi aşan satırlar (orana göre azalan)

| dataset | model | protokol | ölçüt | ortalama | kat min–maks | katlar arası sd | tohumlar arası sd | oran |
|---|---|---|---|---:|---:|---:|---:|---:|
| HUST | M2_RF | P4_kosul_ayrik | makro_f1 | 0.9341 | 0.8726–0.9733 | 0.0539 | 0.0003 | 204.0 |
| HUST | M2_RF | P4_kosul_ayrik | dogruluk | 0.9394 | 0.8835–0.9769 | 0.0493 | 0.0003 | 161.1 |
| HUST | M2_RF | P4_kosul_ayrik | ece | 0.1026 | 0.0721–0.1275 | 0.0281 | 0.0010 | 29.4 |
| HUST | M2_RF | P5_numune_ayrik | makro_f1 | 0.3548 | 0.2765–0.4320 | 0.0643 | 0.0025 | 25.6 |
| HUST | M1_CNN | P5_numune_ayrik | dogruluk | 0.4831 | 0.2195–0.7824 | 0.2220 | 0.0121 | 18.3 |
| CWRU | M2_RF | P4_kosul_ayrik | ece | 0.1069 | 0.0479–0.1784 | 0.0593 | 0.0036 | 16.6 |
| HUST | M2_RF | P5_numune_ayrik | dogruluk | 0.4233 | 0.3233–0.4964 | 0.0630 | 0.0042 | 15.2 |
| SEU | M2_RF | P4_kosul_ayrik | ece | 0.5194 | 0.4756–0.5633 | 0.0620 | 0.0043 | 14.5 |
| HUST | M1_CNN | P5_numune_ayrik | makro_f1 | 0.4165 | 0.2308–0.6185 | 0.1600 | 0.0133 | 12.0 |
| CWRU | M2_RF | P4_kosul_ayrik | dogruluk | 0.9821 | 0.9288–1.0000 | 0.0355 | 0.0032 | 11.0 |
| HUST | M2_RF | P5_numune_ayrik | ece | 0.1895 | 0.1198–0.2980 | 0.0768 | 0.0072 | 10.6 |
| HUST | M1_CNN | P5_numune_ayrik | ece | 0.3557 | 0.0954–0.5793 | 0.2023 | 0.0206 | 9.8 |
| CWRU | M2_RF | P4_kosul_ayrik | makro_f1 | 0.9803 | 0.9217–1.0000 | 0.0391 | 0.0040 | 9.7 |
| SEU | M2_RF | P4_kosul_ayrik | makro_f1 | 0.0930 | 0.0738–0.1122 | 0.0272 | 0.0030 | 8.9 |
| HUST | M1_CNN | P4_kosul_ayrik | makro_f1 | 0.9727 | 0.9308–0.9998 | 0.0368 | 0.0045 | 8.1 |
| HUST | M1_CNN | P4_kosul_ayrik | dogruluk | 0.9739 | 0.9349–0.9998 | 0.0344 | 0.0046 | 7.5 |
| CWRU | M1_CNN | P4_kosul_ayrik | ece | 0.0313 | 0.0055–0.0618 | 0.0275 | 0.0051 | 5.4 |
| SEU | M1_CNN | P4_kosul_ayrik | makro_f1 | 0.3467 | 0.2547–0.4388 | 0.1302 | 0.0262 | 5.0 |
| SEU | M1_CNN | P4_kosul_ayrik | ece | 0.4680 | 0.3415–0.5946 | 0.1790 | 0.0383 | 4.7 |
| SEU | M1_CNN | P4_kosul_ayrik | dogruluk | 0.4593 | 0.3593–0.5593 | 0.1414 | 0.0370 | 3.8 |
| SEU | M2_RF | P4_kosul_ayrik | dogruluk | 0.2014 | 0.2003–0.2024 | 0.0015 | 0.0004 | 3.4 |
| CWRU | M1_CNN | P4_kosul_ayrik | makro_f1 | 0.9694 | 0.9007–1.0000 | 0.0471 | 0.0221 | 2.1 |
| CWRU | M1_CNN | P4_kosul_ayrik | dogruluk | 0.9726 | 0.9132–1.0000 | 0.0410 | 0.0193 | 2.1 |

### Tohumlar arası sd'nin sıfır olduğu satırlar (oran tanımsız)

Bu satırların tamamı M3_TRIVIAL, M4_KNN modellerine ait. Katlar arası sd pozitif olduğu için bütün dağılım katlardan geliyor; 2 eşiği sayısal olarak test edilemese de bu birleşimler metinde anılmalı.

Sıfır bir kararlılık bulgusu değil, tasarımın sonucu: P4 ve P5 bölmeleri tohuma bağlı değil (`splits.leave_one_cond_out`, `splits.leave_one_specimen_out`), tohum yalnızca modele geçiyor (`experiments.run_ladder`). `EnvelopeKNNModel` tohum kullanmıyor; `TrivialModel` kullanıyor ama tohumlar arasında aynı değerleri üretti. Bu yüzden tablolarda bu hücrelerde görülecek ± 0.000, "belirsizlik yok" olarak okunmamalı.

| dataset | model | protokol | ölçüt | ortalama | kat min–maks | katlar arası sd | tohumlar arası sd | oran |
|---|---|---|---|---:|---:|---:|---:|---:|
| CWRU | M3_TRIVIAL | P4_kosul_ayrik | dogruluk | 0.6835 | 0.6593–0.6983 | 0.0177 | 0.0000 | tanımsız |
| CWRU | M3_TRIVIAL | P4_kosul_ayrik | makro_f1 | 0.6204 | 0.5849–0.6383 | 0.0243 | 0.0000 | tanımsız |
| CWRU | M3_TRIVIAL | P4_kosul_ayrik | ece | 0.0233 | 0.0066–0.0630 | 0.0265 | 0.0000 | tanımsız |
| CWRU | M4_KNN | P4_kosul_ayrik | dogruluk | 0.9636 | 0.9085–0.9966 | 0.0391 | 0.0000 | tanımsız |
| CWRU | M4_KNN | P4_kosul_ayrik | makro_f1 | 0.9564 | 0.8800–0.9966 | 0.0526 | 0.0000 | tanımsız |
| CWRU | M4_KNN | P4_kosul_ayrik | ece | 0.0279 | 0.0037–0.0678 | 0.0278 | 0.0000 | tanımsız |
| HUST | M3_TRIVIAL | P4_kosul_ayrik | dogruluk | 0.5321 | 0.4864–0.5619 | 0.0402 | 0.0000 | tanımsız |
| HUST | M3_TRIVIAL | P4_kosul_ayrik | makro_f1 | 0.4767 | 0.4062–0.5175 | 0.0613 | 0.0000 | tanımsız |
| HUST | M3_TRIVIAL | P4_kosul_ayrik | ece | 0.0413 | 0.0192–0.0832 | 0.0363 | 0.0000 | tanımsız |
| HUST | M4_KNN | P4_kosul_ayrik | dogruluk | 0.8654 | 0.8207–0.9178 | 0.0490 | 0.0000 | tanımsız |
| HUST | M4_KNN | P4_kosul_ayrik | makro_f1 | 0.8542 | 0.8074–0.9076 | 0.0504 | 0.0000 | tanımsız |
| HUST | M4_KNN | P4_kosul_ayrik | ece | 0.0417 | 0.0196–0.0702 | 0.0259 | 0.0000 | tanımsız |
| HUST | M3_TRIVIAL | P5_numune_ayrik | dogruluk | 0.3759 | 0.2470–0.4988 | 0.0954 | 0.0000 | tanımsız |
| HUST | M3_TRIVIAL | P5_numune_ayrik | makro_f1 | 0.3176 | 0.2240–0.3836 | 0.0661 | 0.0000 | tanımsız |
| HUST | M3_TRIVIAL | P5_numune_ayrik | ece | 0.2471 | 0.1025–0.3435 | 0.1003 | 0.0000 | tanımsız |
| HUST | M4_KNN | P5_numune_ayrik | dogruluk | 0.2406 | 0.0780–0.4157 | 0.1334 | 0.0000 | tanımsız |
| HUST | M4_KNN | P5_numune_ayrik | makro_f1 | 0.2239 | 0.0857–0.3884 | 0.1197 | 0.0000 | tanımsız |
| HUST | M4_KNN | P5_numune_ayrik | ece | 0.5156 | 0.2988–0.6714 | 0.1493 | 0.0000 | tanımsız |
| SEU | M3_TRIVIAL | P4_kosul_ayrik | dogruluk | 0.2303 | 0.2000–0.2607 | 0.0429 | 0.0000 | tanımsız |
| SEU | M3_TRIVIAL | P4_kosul_ayrik | makro_f1 | 0.1077 | 0.0705–0.1450 | 0.0527 | 0.0000 | tanımsız |
| SEU | M3_TRIVIAL | P4_kosul_ayrik | ece | 0.7276 | 0.6860–0.7692 | 0.0588 | 0.0000 | tanımsız |
| SEU | M4_KNN | P4_kosul_ayrik | dogruluk | 0.8104 | 0.8000–0.8207 | 0.0147 | 0.0000 | tanımsız |
| SEU | M4_KNN | P4_kosul_ayrik | makro_f1 | 0.7762 | 0.7352–0.8171 | 0.0579 | 0.0000 | tanımsız |
| SEU | M4_KNN | P4_kosul_ayrik | ece | 0.1317 | 0.0806–0.1827 | 0.0722 | 0.0000 | tanımsız |

### Oranı 2'yi aşmayan satırlar

| dataset | model | protokol | ölçüt | ortalama | kat min–maks | katlar arası sd | tohumlar arası sd | oran |
|---|---|---|---|---:|---:|---:|---:|---:|
| HUST | M1_CNN | P4_kosul_ayrik | ece | 0.0244 | 0.0216–0.0298 | 0.0047 | 0.0069 | 0.7 |

### `ddof` duyarlılığı

`ddof=0` kullanılsaydı da hiçbir satır eşiğin diğer tarafına geçmezdi.

## 2) HUST P5: model ve kat (rulman modeli) bazında doğruluk

Kaynak: `sonuc_hust_tam/merdiven.csv`, `protokol==P5_numune_ayrik`, `dogruluk`. Her hücre 5 tohum üzerinden ortalama ± tohumlar arası sd. Kat adı, test edilen (eğitimden çıkarılan) rulman modelidir.

| model | 6204 | 6205 | 6206 | 6207 | 6208 |
|---|---:|---:|---:|---:|---:|
| M1_CNN | 0.7824 ± 0.0189 | 0.4186 ± 0.0454 | 0.6271 ± 0.0304 | 0.3680 ± 0.0288 | 0.2195 ± 0.0264 |
| M2_RF | 0.4185 ± 0.0117 | 0.3233 ± 0.0041 | 0.4964 ± 0.0108 | 0.4395 ± 0.0048 | 0.4386 ± 0.0085 |
| M3_TRIVIAL | 0.3266 ± 0.0000 | 0.3850 ± 0.0000 | 0.4221 ± 0.0000 | 0.2470 ± 0.0000 | 0.4988 ± 0.0000 |
| M4_KNN | 0.0780 ± 0.0000 | 0.1985 ± 0.0000 | 0.3325 ± 0.0000 | 0.4157 ± 0.0000 | 0.1784 ± 0.0000 |
| en yüksek | M1_CNN | M1_CNN | M1_CNN | M2_RF | M3_TRIVIAL |
| n_test | 885 | 1239 | 1239 | 1239 | 1239 |

- "En yüksek" satırı kat ortalamalarına göredir; modellerin P5 sıralaması kattan kata değişiyor, protokol ortalamasına dayalı sıra bunu göstermez.
- 6204 katı doğrudan karşılaştırılmamalı: bu modelde Ball ve Inner+Ball kaydı yok (`doldurma_kaydi.md`), test kümesi diğer katlardan küçük (n_test satırı) ve daha az sınıf içeriyor.
- `ortalama` kolonu istendiği gibi düz ortalamadır, kat büyüklüğüne göre ağırlıklı değildir.

Tohum bazında ham değerler:

| model | kat | tohum 0 | tohum 1 | tohum 2 | tohum 3 | tohum 4 |
|---|---|---:|---:|---:|---:|---:|
| M1_CNN | 6204 | 0.7593 | 0.7966 | 0.7650 | 0.7910 | 0.8000 |
| M1_CNN | 6205 | 0.4641 | 0.3454 | 0.4084 | 0.4407 | 0.4342 |
| M1_CNN | 6206 | 0.6755 | 0.6005 | 0.6077 | 0.6142 | 0.6376 |
| M1_CNN | 6207 | 0.3963 | 0.3898 | 0.3422 | 0.3793 | 0.3325 |
| M1_CNN | 6208 | 0.2115 | 0.2131 | 0.2647 | 0.2131 | 0.1953 |
| M2_RF | 6204 | 0.4124 | 0.4215 | 0.4068 | 0.4373 | 0.4147 |
| M2_RF | 6205 | 0.3245 | 0.3269 | 0.3253 | 0.3164 | 0.3236 |
| M2_RF | 6206 | 0.4826 | 0.4875 | 0.5052 | 0.4996 | 0.5069 |
| M2_RF | 6207 | 0.4366 | 0.4479 | 0.4391 | 0.4366 | 0.4374 |
| M2_RF | 6208 | 0.4278 | 0.4455 | 0.4310 | 0.4439 | 0.4447 |
| M3_TRIVIAL | 6204 | 0.3266 | 0.3266 | 0.3266 | 0.3266 | 0.3266 |
| M3_TRIVIAL | 6205 | 0.3850 | 0.3850 | 0.3850 | 0.3850 | 0.3850 |
| M3_TRIVIAL | 6206 | 0.4221 | 0.4221 | 0.4221 | 0.4221 | 0.4221 |
| M3_TRIVIAL | 6207 | 0.2470 | 0.2470 | 0.2470 | 0.2470 | 0.2470 |
| M3_TRIVIAL | 6208 | 0.4988 | 0.4988 | 0.4988 | 0.4988 | 0.4988 |
| M4_KNN | 6204 | 0.0780 | 0.0780 | 0.0780 | 0.0780 | 0.0780 |
| M4_KNN | 6205 | 0.1985 | 0.1985 | 0.1985 | 0.1985 | 0.1985 |
| M4_KNN | 6206 | 0.3325 | 0.3325 | 0.3325 | 0.3325 | 0.3325 |
| M4_KNN | 6207 | 0.4157 | 0.4157 | 0.4157 | 0.4157 | 0.4157 |
| M4_KNN | 6208 | 0.1784 | 0.1784 | 0.1784 | 0.1784 | 0.1784 |

## 3) Şekil 1 ile tablolar arasındaki tanım farkı (doğruluk)

Şekil 1'in P4/P5 hata çubukları (`sekil1_veri.csv`, `dogruluk_std`) tohum ve kat satırlarının birlikte alındığı havuzlanmış sd'dir (`sekil_kaydi.md`). Bu nedenle değerler tohumlar arası sd'ye değil, katlar arası sd'ye yakındır. Aynı hücre için şekildeki çubuk ile tablodaki ± farklı nicelikler gösterir; şekil altyazısı bunu belirtmeli.

| dataset | model | protokol | Şekil 1 (havuzlanmış) | katlar arası sd | tohumlar arası sd |
|---|---|---|---:|---:|---:|
| CWRU | M1_CNN | P4_kosul_ayrik | 0.0492 | 0.0410 | 0.0193 |
| CWRU | M2_RF | P4_kosul_ayrik | 0.0321 | 0.0355 | 0.0032 |
| CWRU | M3_TRIVIAL | P4_kosul_ayrik | 0.0157 | 0.0177 | 0.0000 |
| CWRU | M4_KNN | P4_kosul_ayrik | 0.0347 | 0.0391 | 0.0000 |
| HUST | M1_CNN | P4_kosul_ayrik | 0.0297 | 0.0344 | 0.0046 |
| HUST | M2_RF | P4_kosul_ayrik | 0.0417 | 0.0493 | 0.0003 |
| HUST | M3_TRIVIAL | P4_kosul_ayrik | 0.0340 | 0.0402 | 0.0000 |
| HUST | M4_KNN | P4_kosul_ayrik | 0.0414 | 0.0490 | 0.0000 |
| HUST | M1_CNN | P5_numune_ayrik | 0.2047 | 0.2220 | 0.0121 |
| HUST | M2_RF | P5_numune_ayrik | 0.0580 | 0.0630 | 0.0042 |
| HUST | M3_TRIVIAL | P5_numune_ayrik | 0.0871 | 0.0954 | 0.0000 |
| HUST | M4_KNN | P5_numune_ayrik | 0.1218 | 0.1334 | 0.0000 |
| SEU | M1_CNN | P4_kosul_ayrik | 0.1216 | 0.1414 | 0.0370 |
| SEU | M2_RF | P4_kosul_ayrik | 0.0013 | 0.0015 | 0.0004 |
| SEU | M3_TRIVIAL | P4_kosul_ayrik | 0.0320 | 0.0429 | 0.0000 |
| SEU | M4_KNN | P4_kosul_ayrik | 0.0109 | 0.0147 | 0.0000 |

## 4) Tek katlı protokoller (P1, P2, P3)

Bu protokollerde kat yok; katlar arası sd hesaplanmadı. Tohumlar arası sd `dagilim_ozeti.csv` içinde. 108 satırın 32 tanesinde tohumlar arası sd sıfır:

| dataset | model | protokol | ölçütler |
|---|---|---|---|
| CWRU | M1_CNN | P1_rastgele_ortusmeli | dogruluk, makro_f1 |
| CWRU | M1_CNN | P2_rastgele_ortusmesiz | dogruluk, makro_f1 |
| CWRU | M1_CNN | P3_zamansal_blok | dogruluk, makro_f1 |
| CWRU | M2_RF | P1_rastgele_ortusmeli | dogruluk, makro_f1 |
| CWRU | M2_RF | P3_zamansal_blok | dogruluk, makro_f1 |
| CWRU | M3_TRIVIAL | P3_zamansal_blok | dogruluk, makro_f1, ece |
| CWRU | M4_KNN | P3_zamansal_blok | dogruluk, makro_f1, ece |
| HUST | M1_CNN | P1_rastgele_ortusmeli | dogruluk, makro_f1 |
| HUST | M3_TRIVIAL | P3_zamansal_blok | dogruluk, makro_f1, ece |
| HUST | M4_KNN | P3_zamansal_blok | dogruluk, makro_f1, ece |
| SEU | M1_CNN | P3_zamansal_blok | dogruluk, makro_f1 |
| SEU | M3_TRIVIAL | P3_zamansal_blok | dogruluk, makro_f1, ece |
| SEU | M4_KNN | P3_zamansal_blok | dogruluk, makro_f1, ece |
