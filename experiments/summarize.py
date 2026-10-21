"""
Tables of the grid benchmark (run_grid.py). For every dataset, encoding and seed, each model
keeps the configuration with the best validation accuracy, and is reported on test:

  one-shot + bleaching   best over bits x address              (ties: fewer bits, smaller address)
  epochs + bleaching     best over bits x address x b x rho0 x tau
                         (ties: weakest correction - smaller rho0, tau, b - then fewer bits, smaller address)

The test set never takes part in the choice. Usage:
  python experiments/summarize.py --resultsDir experiments/results [--out summary.md]
"""
import argparse
import collections
import glob
import json
import math
import os

import numpy as np
from scipy import stats

MODELS = ("epochs + bleaching", "one-shot + bleaching")


def load(results_dir):
    """{(dataset, encoding): {seed: [results]}}"""
    out = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, "*", "*", "seed*.jsonl")):
        dataset, encoding = path.split(os.sep)[-3:-1]
        seed = int(os.path.basename(path)[4:-6])
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        out[(dataset, encoding)][seed] = [r for r in rows if "error" not in r]
    return out


def choose(rows):
    """{model: (configuration, test metrics, validation metrics)} for the rows of one seed"""
    chosen = {}
    oneshot = [r for r in rows if r["kind"] == "oneshot"]
    epochs = [r for r in rows if r["kind"] == "epochs"]
    if oneshot:
        r = max(oneshot, key=lambda r: (r["bleaching"]["val"]["acc"], -r["bits"], -r["address"]))
        chosen["one-shot + bleaching"] = (dict(bits=r["bits"], address=r["address"]), r["bleaching"]["test"], r["bleaching"]["val"])
    if epochs:
        r = max(epochs, key=lambda r: (r["bleaching"]["val"]["acc"], -r["rho0"], -r["tau"], -r["b"], -r["bits"], -r["address"]))
        chosen["epochs + bleaching"] = (dict(bits=r["bits"], address=r["address"], b=r["b"], rho0=r["rho0"], tau=r["tau"],
                                             bestEpoch=r["bestEpoch"], ran=len(r["accValPerEpoch"])),
                                        r["bleaching"]["test"], r["bleaching"]["val"])
    return chosen


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def paired(a, b):
    d = np.array(a) - np.array(b)
    wins, ties, losses = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    if len(d) < 2 or np.all(d == d[0]):
        p = 0.0 if len(d) >= 2 and d[0] != 0 else float("nan")
    else:
        p = float(stats.ttest_rel(a, b).pvalue)
    return float(d.mean()), (float(d.std(ddof=1)) if len(d) > 1 else 0.0), wins, ties, losses, p


def fmt(m, s):
    return "n/a" if math.isnan(m) else "%.4f ± %.4f" % (m, s)


def summarize(data, expected_seeds):
    lines, overview = [], []
    for (dataset, encoding) in sorted(data):
        seeds = data[(dataset, encoding)]
        per_seed = {seed: choose(rows) for seed, rows in sorted(seeds.items())}
        complete = [s for s, c in per_seed.items() if len(c) == len(MODELS)]
        lines.append("\n## %s — %s\n" % (dataset, encoding))
        if len(complete) < expected_seeds:
            lines.append("_%d of %d seeds have results for every model._\n" % (len(complete), expected_seeds))
        if not complete:
            continue
        lines.append("| model | test accuracy | test confidence | conf (hits) | conf (errors) | val accuracy |")
        lines.append("|---|---|---|---|---|---|")
        tests = {}
        for model in MODELS:
            test = [per_seed[s][model][1] for s in complete]
            val = [per_seed[s][model][2] for s in complete]
            tests[model] = [t["acc"] for t in test]
            lines.append("| %s | %s | %s | %.4f | %.4f | %s |" % (
                model, fmt(*mean_std(tests[model])), fmt(*mean_std([t["conf"] for t in test])),
                mean_std([t["conf_hit"] for t in test])[0], mean_std([t["conf_miss"] for t in test])[0],
                fmt(*mean_std([v["acc"] for v in val]))))
        lines.append("")
        row = dict(dataset=dataset, encoding=encoding, seeds=len(complete),
                   **{m: mean_std(tests[m])[0] for m in MODELS})
        for other in MODELS[1:]:
            m, s, w, t, l, p = paired(tests["epochs + bleaching"], tests[other])
            lines.append("- epochs vs %s: %+.4f ± %.4f (wins %d, ties %d, losses %d), paired t p = %.4g" % (other, m, s, w, t, l, p))
            row["vs " + other] = (m, p)
        configs = [per_seed[s]["epochs + bleaching"][0] for s in complete]
        lines.append("- epochs, chosen per seed: " + "; ".join(
            "bits %(bits)d, address %(address)d, b %(b)d, rho0 %(rho0)s, tau %(tau)s, best epoch %(bestEpoch)d/%(ran)d" % c
            for c in configs))
        lines.append("- one-shot + bleaching, chosen per seed: " + "; ".join(
            "bits %(bits)d, address %(address)d" % per_seed[s]["one-shot + bleaching"][0] for s in complete))
        overview.append(row)

    head = ["# Epoch training benchmark\n",
            "Test accuracy (mean over seeds) of the configuration chosen on validation; "
            "Δ = epochs − baseline, p = paired t test over the seeds.\n",
            "| dataset | encoding | seeds | epochs | one-shot + bleaching | Δ (p) |",
            "|---|---|---|---|---|---|"]
    for r in overview:
        head.append("| %s | %s | %d | %.4f | %.4f | %+.4f (%.3g) |" % (
            r["dataset"], r["encoding"], r["seeds"], r["epochs + bleaching"], r["one-shot + bleaching"],
            r["vs one-shot + bleaching"][0], r["vs one-shot + bleaching"][1]))
    return "\n".join(head + lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resultsDir", default="experiments/results")
    parser.add_argument("--seeds", type=int, default=5, help="seeds expected per dataset (to flag incomplete ones)")
    parser.add_argument("--out", help="also write the tables to this markdown file")
    args = parser.parse_args()
    text = summarize(load(args.resultsDir), args.seeds)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
