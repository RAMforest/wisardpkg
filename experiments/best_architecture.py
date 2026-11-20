"""
Best base configuration of the WiSARD per dataset, averaged over seeds (run_grid.py results).

summarize.py chooses the best configuration inside each seed and then averages the chosen ones.
This script does the opposite: every configuration (all hyperparameters of the input and of the
WiSARD, everything but the seed) is averaged over the seeds, and the configuration with the best
MEAN validation accuracy is the base configuration of the dataset. The test set never takes part
in the choice; it is only reported.

Configuration (the key averaged over seeds):
  --kind oneshot (default)  encoding, bits, address, valMode
  --kind epochs             encoding, bits, address, valMode, b, rho0, tau
valMode is the bleaching of the classification ("bleaching" = iterative, "DAB", "rDAB"); results
written before the modes existed have no valMode and count as "bleaching".

Note: in run_grid.py the seed changes the RAM mapping AND the split (and, for epochs, the
shuffling of the epochs), so the spread over seeds is the spread over mapping + split.

Seeds: the mean of configurations seen on different seeds is not comparable, so, per dataset,
only the configurations present in the largest number of seeds compete (or at least --minSeeds;
--seeds fixes the seeds to use). Ties on the mean validation accuracy: fewer bits, then smaller
address.

Usage:
  python experiments/best_architecture.py
  python experiments/best_architecture.py --kind epochs --top 10 --out experiments/best_epochs.md
  python experiments/best_architecture.py --encodings distributive --valModes bleaching --seeds 1 2 3
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

ONESHOT_KEY = ("encoding", "bits", "address", "valMode")
EPOCHS_KEY = ONESHOT_KEY + ("b", "rho0", "tau")


def val_mode(r):
    return r.get("valMode", "bleaching")


def load(results_dir, kind, encodings, val_modes, seeds):
    """{dataset: {config: {seed: row}}}; a configuration repeated in a seed file keeps its last line"""
    fields = ONESHOT_KEY if kind == "oneshot" else EPOCHS_KEY
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in glob.glob(os.path.join(results_dir, "*", "*", "seed*.jsonl")):
        dataset, encoding = path.split(os.sep)[-3:-1]
        seed = int(os.path.basename(path)[4:-6])
        if (encodings and encoding not in encodings) or (seeds and seed not in seeds):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "error" in r or r["kind"] != kind or "bleaching" not in r:
                    continue
                r["valMode"], r["encoding"] = val_mode(r), encoding
                if val_modes and r["valMode"] not in val_modes:
                    continue
                out[dataset][tuple(r[k] for k in fields)][seed] = r
    return out, fields


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def rank(configs, min_seeds):
    """configurations with enough seeds, best mean validation accuracy first"""
    most = max(len(s) for s in configs.values())
    need = min_seeds or most
    rows = []
    for config, by_seed in configs.items():
        if len(by_seed) < need:
            continue
        seeds = sorted(by_seed)
        val = mean_std([by_seed[s]["bleaching"]["val"]["acc"] for s in seeds])
        test = mean_std([by_seed[s]["bleaching"]["test"]["acc"] for s in seeds])
        conf = mean_std([by_seed[s]["bleaching"]["test"]["conf"] for s in seeds])
        rows.append(dict(config=config, seeds=seeds, val=val, test=test, conf=conf))
    # config[1] = bits, config[2] = address
    rows.sort(key=lambda r: (-r["val"][0], r["config"][1], r["config"][2]))
    return rows, need, most


def best_configs(results_dir="experiments/results", kind="oneshot", encodings=None, val_modes=None, seeds=None,
                 min_seeds=None, datasets=None):
    """{dataset: configuration} of the best mean validation accuracy over seeds, e.g.
    {"blood": {"kind": "oneshot", "encoding": "distributive", "bits": 16, "address": 16, "valMode": "DAB",
               "b": None, "rho0": None, "tau": None, "seeds": [0, 1, 2, 3, 4], "val": 0.7823, "test": 0.7532}}
    (b, rho0, tau are None for kind="oneshot"); datasets without a competing configuration are left out"""
    data, fields = load(results_dir, kind, encodings, val_modes, seeds)
    out = {}
    for dataset in sorted(data):
        if datasets and dataset not in datasets:
            continue
        rows, _, _ = rank(data[dataset], min_seeds)
        if not rows:
            continue
        best = rows[0]
        config = dict(kind=kind, b=None, rho0=None, tau=None)
        config.update(zip(fields, best["config"]))
        config.update(seeds=best["seeds"], val=best["val"][0], test=best["test"][0])
        out[dataset] = config
    return out


def fmt(m, s):
    return "n/a" if np.isnan(m) else "%.4f ± %.4f" % (m, s)


def report(data, fields, args):
    head = ["# Best base configuration per dataset (%s)\n" % args.kind,
            "Mean over seeds of each configuration; chosen on validation, test only reported.\n",
            "| dataset | %s | seeds | val acc | test acc | competing |" % " | ".join(fields),
            "|---|%s---|---|---|---|" % ("---|" * len(fields))]
    body = []
    for dataset in sorted(data):
        configs = data[dataset]
        rows, need, most = rank(configs, args.minSeeds)
        if not rows:
            body.append("\n## %s\n\n_no configuration with %d seeds._\n" % (dataset, need))
            continue
        best = rows[0]
        head.append("| %s | %s | %s | %s | %s | %d of %d |" % (
            dataset, " | ".join(str(v) for v in best["config"]), ",".join(map(str, best["seeds"])),
            fmt(*best["val"]), fmt(*best["test"]), len(rows), len(configs)))
        body.append("\n## %s\n" % dataset)
        body.append("%d of %d configurations have >= %d seeds (most: %d).\n" % (len(rows), len(configs), need, most))
        body.append("| # | %s | seeds | val acc | test acc | test conf |" % " | ".join(fields))
        body.append("|---|%s---|---|---|---|" % ("---|" * len(fields)))
        for i, r in enumerate(rows[:args.top], 1):
            body.append("| %d | %s | %s | %s | %s | %s |" % (
                i, " | ".join(str(v) for v in r["config"]), ",".join(map(str, r["seeds"])),
                fmt(*r["val"]), fmt(*r["test"]), fmt(*r["conf"])))
    return "\n".join(head + body) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resultsDir", default="experiments/results")
    parser.add_argument("--kind", default="oneshot", choices=("oneshot", "epochs"))
    parser.add_argument("--encodings", nargs="+", help="only these encodings (default: all)")
    parser.add_argument("--valModes", nargs="+", choices=("bleaching", "DAB", "rDAB"), help="only these modes (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", help="only these seeds (default: all found)")
    parser.add_argument("--minSeeds", type=int,
                        help="seeds a configuration needs to compete (default: the most seen in the dataset)")
    parser.add_argument("--top", type=int, default=5, help="configurations listed per dataset")
    parser.add_argument("--out", help="also write the tables to this markdown file")
    args = parser.parse_args()
    data, fields = load(args.resultsDir, args.kind, args.encodings, args.valModes, args.seeds)
    text = report(data, fields, args)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
