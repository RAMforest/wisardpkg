# Epoch training benchmark

Compares, on several datasets, the one-shot WiSARD with the WiSARD trained by epochs
(`Wisard.trainEpochs`), with two binarizations.

| file | role |
|---|---|
| `datasets.py` | loaders (OpenML, MNIST/FashionMNIST IDX, CIFAR-10 binary), cached in `--dataDir` |
| `encoders.py` | thermometer and distributive thermometer, fitted on the training part only |
| `run_grid.py` | trains and evaluates every configuration of the grid; one JSON line per configuration |
| `summarize.py` | chooses each model's configuration on validation and writes the tables (test) |

Requirements (only for the experiments, not for the package): `numpy`, `scikit-learn`
(OpenML loader), `pandas`, `scipy`, and `wisardpkg` installed (`pip install .`).

## Protocol

- **Datasets**: iris, phoneme, skin-seg, higgs, australian, nomao, segment (binarized),
  miniboone, christine, jasmine, sylvine, blood, jsc (OpenML ids in `datasets.py`), MNIST,
  FashionMNIST, CIFAR-10. KWS and ToyADMOS/car are not included yet (audio pipelines).
- **Splits**: the official test set for MNIST, FashionMNIST and CIFAR-10, with 15% of the
  official training set as validation; a stratified 70/15/15 split for the others. Split, RAM
  mapping (complete addressing) and epoch shuffling change with the seed (0..4).
- **Encodings**: `thermometer` (thresholds evenly spaced between the training min and max of each
  feature) and `distributive` (thresholds at the training quantiles of each feature); nominal
  features are one-hot.
- **Grid**, joint and repeated for every seed and encoding:
  - bits per feature × address size: tabular {4, 8, 16} × {8, 16, 24, 32};
    images {2, 4, 8} × {16, 24, 32, 40};
  - epoch training: b ∈ {1, 2} × rho0 ∈ {0.1, 0.15, 0.25, 0.35, 0.5, 0.6} ×
    tau ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}; at most 20 epochs, early stopping with patience 3
    on the validation accuracy with bleaching (`valMode="bleaching"`);
  - one-shot classified with iterative bleaching (the baseline).
- **Choice**: each model keeps, per seed, the configuration with the best validation accuracy;
  the test set is only reported. Confidence is C = (R_w − R_l)/R_w on the votes where the model
  decides (0 on ties).

## Running

```
python experiments/run_grid.py --pilot --datasets iris blood   # timing of a few configurations + estimate
python experiments/run_grid.py --datasets iris blood --jobs 10   # full grid; resumes if interrupted
python experiments/summarize.py --out experiments/results/summary.md
```

Images with many bits make large models (CIFAR-10 with 8 bits and address 16: about 1500 RAMs per
class and millions of stored addresses); run them with fewer `--jobs` to bound the memory.
