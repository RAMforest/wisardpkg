# Epoch training benchmark

Compares, on several datasets, the one-shot WiSARD with the WiSARD trained by epochs
(`Wisard.trainEpochs`), with two binarizations.

| file | role |
|---|---|
| `datasets.py` | loaders (OpenML, MNIST/FashionMNIST IDX, CIFAR-10 binary), cached in `--dataDir` |
| `encoders.py` | thermometer and distributive thermometer, fitted on the training part only |
| `run_grid.py` | trains and evaluates every configuration of the grid; one JSON line per configuration |
| `summarize.py` | chooses each model's configuration on validation and writes the tables (test) |
| `measure_memory.py` | RAM used while training (and classifying) configurations of the results, one new process per configuration |
| `run_dwn.py` | DWN (`torch-dwn`, NVIDIA GPU only) on the same splits, seeds and binary inputs; accuracy, training and inference time per configuration in `results_dwn/` |
| `measure_time.py` | training and inference time on higgs of the one-shot and epoch WiSARD (iterative bleaching, DAB, rDAB) and the DWN, all with the same encoding, bits and tuple size, per seed, in `results_time/` |
| `dwn_benchmark.ipynb` | tables of the DWN against the WiSARD (epochs and one-shot) |

Requirements (only for the experiments, not for the package): `numpy`, `scikit-learn`
(OpenML loader), `pandas`, `scipy`, `psutil` (`measure_memory.py`), and `wisardpkg` installed
(`pip install .`).

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
  - validation mode (`VAL_MODES` in `run_grid.py`): the bleaching of the val/test classification of
    both models and of the validation of every epoch: `bleaching` (iterative), `DAB` or `rDAB`
    (`--valModes` runs only some); `summarize.py --valMode` and `summarize.load(dir, valMode)`
    read one mode at a time (default `bleaching`, which also covers the older results);
  - one-shot classified with iterative bleaching (the baseline).
- **Choice**: each model keeps, per seed, the configuration with the best validation accuracy;
  the test set is only reported. Confidence is C = (R_w − R_l)/R_w on the votes where the model
  decides (0 on ties).

## Running

```
python experiments/run_grid.py --pilot --datasets iris blood   # timing of a few configurations + estimate
python experiments/run_grid.py --datasets iris blood --jobs 10   # full grid; resumes if interrupted
python experiments/summarize.py --out experiments/results/summary.md
python experiments/measure_memory.py --dry-run                   # configurations to measure + time estimate
python experiments/measure_memory.py --seeds 1 --jobs 2          # RAM of training and classification; resumes
```

Images with many bits make large models (CIFAR-10 with 8 bits and address 16: about 1500 RAMs per
class and millions of stored addresses); run them with fewer `--jobs` to bound the memory.

## Multi-bleaching benchmark

Compares every bleaching base (iterative, bbleaching, dab) with and without the multi-bleaching
layer (`Wisard(multiBleaching=True)`), on the same trained one-shot network.

| file | role |
|---|---|
| `run_multi_bleaching.py` | trains every network of the grid once and classifies validation and test with the 18 rules (3 bases x {no multi, multi with C_th in 0.05, 0.1, 0.2, 0.5, 1.0}); one JSON line per network in `results_multi/`, plus a checkpoint and a log per network |
| `summarize_multi_bleaching.py` | paired comparison of every multi rule with its base: mean and standard deviation of the difference over the networks, Wilcoxon with Holm correction, and the same difference on the best network of validation |

- **Datasets**: the ones of the epoch benchmark with results, plus `covertype`, `letter`,
  `wine-white` and `wine-red` (`OPENML_EXTRA` in `datasets.py`; they are not in `NAMES`, so the
  default grids above do not change).
- **Encoding**: `Encoder(..., binary_as_bit=True)`: distributive thermometer only on numeric
  features with three or more values in training, one-hot only on nominal features with three or
  more categories, one bit for a feature with two values (nominal or numeric), nothing for a
  constant one. The JSON line and the log of every network say how many features fell in each case.
- **Networks**: the bits x address grid of `run_grid.py`, three mappings (`random`, the one of
  `run_grid.py`; `per-class`; `linear`) and seeds 0..4. On datasets without binary or constant
  features, the base rules with the random mapping reproduce the one-shot results of `run_grid.py`.
- **Outputs of every network**, under `results_multi/`: the JSON line with the scores of every rule;
  `checkpoints/.../<network>.json.gz`, the trained model (`run_multi_bleaching.load_checkpoint`
  gives the `Wisard` back); `logs/.../<network>.log`; and, with `--traces`,
  `traces/.../<network>.npz` with the predictions and the threshold and confidence of every round.

```
python experiments/run_multi_bleaching.py --dry-run                       # networks per dataset
python experiments/run_multi_bleaching.py --datasets iris blood --jobs 6  # resumes if interrupted
python experiments/summarize_multi_bleaching.py --out experiments/results_multi/summary.md
```

OpenML datasets are parsed once and kept in `<dataDir>/parsed/<name>.npz`.
