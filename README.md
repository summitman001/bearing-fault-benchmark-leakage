# Bearing fault diagnosis benchmarks: how much of the reported accuracy survives stricter splits?

Code, per-run results, and figure and table scripts for a measurement study on
bearing fault diagnosis benchmarks. The study asks how much of the near-perfect
accuracy reported on these benchmarks reflects real fault discrimination rather
than split leakage, and what happens to probability estimates when the operating
condition changes.

Identifiers, file names and column names are in Turkish; the glossary below
covers the ones needed to read the results.

## Setup

Python 3.11.9 was used for the reported runs. Exact package versions are recorded
in each results folder as `ortam.json`.

```bash
pip install -r requirements.txt
pip install torch   # required only for the CNN (M1_CNN)
```

## Reproducing the figures and tables

The results of the reported runs are included, so no dataset or new run is needed:

```bash
python sekiller.py
python cnn_tablolari.py
python dagilim.py
```

Outputs are written to `sekiller/`.

## Tests

```bash
python tests/test_pipeline.py
```

## Data

The datasets are third-party and are **not** redistributed here. Obtain them from
their distributors:

| Dataset | Source | Use |
|---|---|---|
| CWRU | Case Western Reserve University Bearing Data Center (`python run.py --asama indir --ham data` downloads the files) | 12 kHz drive-end, 10 classes, 4 loads, 40 recordings |
| HUST | HUST bearing dataset: Mendeley Data `cbv7jyx4p9`, described in arXiv:2302.12533 | 5 bearing models, 3 loads, 99 recordings |
| SEU | Southeast University gearbox dataset, `SEU bearingset` folder | 5 classes, 2 conditions, 10 recordings, Channel2 |

Convert the SEU CSV files before running:

```bash
python prepare_seu.py --ham "/path/to/SEU bearingset" --cikti data/seu_clean
```

## Re-running the experiments

The commands below write the same folders that are included here. The file paths in
each manifest are relative to `--ham`, so place the data accordingly.

```bash
python run.py --asama hepsi --veri cwru --ham data --cikti sonuc_cwru_tam
python run.py --asama hepsi --veri manifest --manifest sonuc/manifest_hust_run.csv --ham /path/to/data --cikti sonuc_hust_tam
python run.py --asama hepsi --veri manifest --manifest sonuc/manifest_seu_run.csv --ham data --cikti sonuc_seu_tam
```

To check the pipeline without any real data:

```bash
python run.py --asama hepsi --veri sentetik --hizli --tohum 0
```

Synthetic output is for pipeline checks only and is not used in the study.

## Design

**Split protocols**

| Protocol | Definition |
|---|---|
| P1 | Random split of overlapping windows |
| P2 | Random split of non-overlapping windows |
| P3 | Temporal block within each recording (first 70% train, last 30% test) |
| P4 | Condition-disjoint: one operating condition held out, one fold per condition |
| P5 | Specimen-disjoint: one bearing model held out, one fold per model (HUST only) |

**Models:** M1_CNN (1D CNN), M2_RF (random forest on time and frequency features),
M3_TRIVIAL (depth-4 decision tree on RMS and kurtosis only), M4_KNN (k-NN on the
envelope spectrum).

**Seeds:** 0, 1, 2, 3, 4. P4 and P5 folds do not depend on the seed, so the seed only
reaches the model. Seed-by-fold rows are repeated measurements of the same folds, not
independent observations. Folds also share training data with one another.

## Results files

Each `sonuc_*_tam/` folder contains:

| File | Content |
|---|---|
| `merdiven.csv` | Split-ladder results, one row per protocol, model, seed and fold |
| `sonda.csv` | Mechanism probes: recording identity and label permutation |
| `guvenilirlik.csv` | Reliability curves |
| `sicaklik_k.csv` | Temperature scaling with k labelled target samples |
| `onkayit_hukmu.csv` | Outcome of the pre-registered calibration criterion |
| `veri_envanteri_*.csv` | Inventory of the recordings used |
| `ortam.json` | Software environment |

Common column values: `protokol` protocol, `kat` fold, `tohum` seed, `dogruluk`
accuracy.

## Known data limitations

- **SEU labels:** the class in the file name disagrees with the `Title` field in the
  file header. File-name labels are used; which is authoritative could not be verified.
- **SEU recording length:** all ten distributed files have exactly 1,048,576 lines, the
  spreadsheet row limit, while their headers imply longer recordings. They appear to be
  truncated; this is an inference, not a confirmed fact.
- **SEU channel and sampling rate:** Channel2 is described as the planetary gearbox x
  direction by the distributing repository, which the files themselves cannot confirm.
  The headers are consistent with 5120 Hz, also an inference.
- **HUST sampling rate:** the metadata value could not be verified against the data note.
- **HUST design:** the 6204 bearing has no Ball or Inner+Ball recordings.

## License

Code is released under the MIT License (see `LICENSE`). The license does not cover the
third-party datasets.
