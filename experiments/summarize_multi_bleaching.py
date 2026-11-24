"""
Tables of the multi-bleaching benchmark (run_multi_bleaching.py results), on the test set.

For every dataset, base (iterative, bbleaching, dab) and C_th, the multi rule is compared with its
base on the SAME network (dataset, encoding, bits, address, mapping, seed):

  all networks   delta = metric(base + multi@C_th) - metric(base) of every network; mean and
                 standard deviation of the delta, networks where the multi rule wins / ties /
                 loses, two-sided Wilcoxon signed-rank test, and its p-value corrected by Holm
                 over the base x C_th comparisons of the dataset
  best network   the configuration (bits, address, mapping) with the best mean VALIDATION metric
                 of the base over the seeds; mean and standard deviation of the delta over its seeds

The second cut is there because a mean over every network of the grid can be positive only
because of the weak networks. The test set never chooses anything.

Usage:
  python experiments/summarize_multi_bleaching.py --out experiments/results_multi/summary.md
  python experiments/summarize_multi_bleaching.py --metric acc --datasets higgs jsc
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_multi_bleaching as rm  # noqa: E402

ALPHA = 0.05


def load(results_dir):
    """{dataset: {(encoding, bits, address, mapping): {seed: rules}}}; a repeated network keeps its last line"""
    out = collections.OrderedDict()
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "seed*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "error" in r:
                    continue
                key = (r["encoding"], r["bits"], r["address"], r["mapping"])
                out.setdefault(r["dataset"], collections.OrderedDict()).setdefault(key, {})[r["seed"]] = r["rules"]
    return out


def thresholds_of(nets):
    rules = next(iter(next(iter(nets.values())).values()))
    return sorted(float(name.split("@")[1]) for name in rules if name.startswith(rm.BASES[0] + "+multi@"))


def wilcoxon(deltas):
    nonzero = deltas[deltas != 0]
    if len(nonzero) < 6:  # too few pairs for the test to mean anything
        return float("nan")
    return float(stats.wilcoxon(nonzero).pvalue)


def holm(pvalues):
    """Holm step-down adjusted p-values; nan stays nan"""
    p = np.array(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out


def analyse(nets, metric, thresholds):
    """rows of one dataset: one per base x C_th"""
    seeds = sorted(set(s for per_seed in nets.values() for s in per_seed))
    full = [k for k, per_seed in nets.items() if len(per_seed) == len(seeds)]  # configurations seen on every seed
    rows = []
    for base in rm.BASES:
        best = max(full, key=lambda k: np.mean([nets[k][s][base]["val"][metric] for s in seeds])) if full else None
        for c_th in thresholds:
            rule = rm.rule_name(base, c_th)
            pairs = [(rules[base]["test"], rules[rule]["test"]) for per_seed in nets.values() for rules in per_seed.values()]
            delta = np.array([m[metric] - b[metric] for b, m in pairs])
            row = dict(base=base, c_th=c_th, n=len(delta), mean=delta.mean(), std=delta.std(), wins=int((delta > 0).sum()),
                       ties=int((delta == 0).sum()), losses=int((delta < 0).sum()), p=wilcoxon(delta),
                       base_metric=np.mean([b[metric] for b, _ in pairs]),
                       base_steps=np.mean([b["steps"] for b, _ in pairs]), multi_steps=np.mean([m["steps"] for _, m in pairs]),
                       rounds=np.mean([m["rounds"] for _, m in pairs]), fraction=np.mean([m["multiFraction"] for _, m in pairs]),
                       by_mapping={kind: float(np.mean([rules[rule]["test"][metric] - rules[base]["test"][metric]
                                                        for k, per_seed in nets.items() if k[3] == kind for rules in per_seed.values()]))
                                   for kind in sorted(set(k[3] for k in nets))})
            if best is not None:
                b = np.array([nets[best][s][base]["test"][metric] for s in seeds])
                m = np.array([nets[best][s][rule]["test"][metric] for s in seeds])
                row.update(best="%db a%d %s" % best[1:], best_base=b.mean(), best_base_std=b.std(),
                           best_mean=(m - b).mean(), best_std=(m - b).std())
            rows.append(row)
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
        significant = not np.isnan(adjusted) and adjusted < ALPHA
        row["verdict"] = ("better" if row["mean"] > 0 else "worse") if significant and row["mean"] != 0 else "same"
    return rows


def fmt_p(p):
    return "-" if np.isnan(p) else ("<0.001" if p < 0.001 else "%.3f" % p)


def report(results, metric):
    lines = ["# Multi-bleaching benchmark", "",
             "Metric: `%s` on the test set. Delta = base + multi - base, on the same network. "
             "`p Holm`: Wilcoxon signed-rank over the networks, corrected over the base x C_th comparisons of the dataset; "
             "significant below %.2f." % (metric, ALPHA), ""]
    per_dataset = collections.OrderedDict((name, analyse(nets, metric, thresholds_of(nets))) for name, nets in results.items())
    if not per_dataset:
        return "no results"
    thresholds = sorted(set(r["c_th"] for rows in per_dataset.values() for r in rows))

    lines += ["## Over the datasets", "",
              "Datasets where the multi rule is significantly better / not different / significantly worse than its base "
              "(all networks), the mean of the dataset deltas, and the datasets where the best network of the base "
              "improves / ties / worsens.", "",
              "| base | C_th | better / same / worse | mean delta | best network: up / tie / down | mean delta (best network) |",
              "|---|---|---|---|---|---|"]
    for base in rm.BASES:
        for c_th in thresholds:
            rows = [r for rs in per_dataset.values() for r in rs if r["base"] == base and r["c_th"] == c_th]
            verdicts = collections.Counter(r["verdict"] for r in rows)
            best = [r["best_mean"] for r in rows if "best_mean" in r]
            lines.append("| %s | %g | %d / %d / %d | %+.4f | %d / %d / %d | %+.4f |" % (
                base, c_th, verdicts["better"], verdicts["same"], verdicts["worse"], np.mean([r["mean"] for r in rows]),
                sum(b > 0 for b in best), sum(b == 0 for b in best), sum(b < 0 for b in best), np.mean(best) if best else float("nan")))
    half = len(per_dataset) / 2.0
    winners = ["%s C_th=%g" % (base, c_th) for base in rm.BASES for c_th in thresholds
               if sum(r["verdict"] == "better" for rs in per_dataset.values() for r in rs if r["base"] == base and r["c_th"] == c_th) > half]
    lines += ["", "Success criterion of RFC-001 (a base and C_th significantly better on most of the %d datasets): %s" % (
        len(per_dataset), "**met** by " + ", ".join(winners) if winners else "**not met**"), ""]

    kinds = sorted(set(k for rs in per_dataset.values() for r in rs for k in r["by_mapping"]))
    lines += ["## Mapping", "", "Mean over the datasets of the mean delta of the networks of each mapping.", "",
              "| base | mapping | " + " | ".join("C_th=%g" % c for c in thresholds) + " |", "|---|---|" + "---|" * len(thresholds)]
    for base in rm.BASES:
        for kind in kinds:
            cells = ["%+.4f" % np.mean([r["by_mapping"][kind] for rs in per_dataset.values() for r in rs
                                        if r["base"] == base and r["c_th"] == c and kind in r["by_mapping"]]) for c in thresholds]
            lines.append("| %s | %s | %s |" % (base, kind, " | ".join(cells)))

    lines += ["", "## Per dataset", ""]
    for name, rows in per_dataset.items():
        lines += ["### %s (%d networks)" % (name, rows[0]["n"]), "",
                  "| base | C_th | base %s | delta mean ± std | win / tie / loss | p | p Holm | verdict | steps base → multi | rounds | "
                  "more than 1 round | best network | its base | its delta mean ± std |" % metric,
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            best = ("%s | %.4f ± %.4f | %+.4f ± %.4f" % (r["best"], r["best_base"], r["best_base_std"], r["best_mean"], r["best_std"])
                    if "best" in r else "- | - | -")
            lines.append("| %s | %g | %.4f | %+.4f ± %.4f | %d / %d / %d | %s | %s | %s | %.2f → %.2f | %.2f | %.1f%% | %s |" % (
                r["base"], r["c_th"], r["base_metric"], r["mean"], r["std"], r["wins"], r["ties"], r["losses"], fmt_p(r["p"]),
                fmt_p(r["p_holm"]), r["verdict"], r["base_steps"], r["multi_steps"], r["rounds"], 100 * r["fraction"], best))
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resultsDir", default="experiments/results_multi")
    parser.add_argument("--metric", default="macro_f1", choices=("macro_f1", "acc", "bal_acc"))
    parser.add_argument("--datasets", nargs="+", help="only these datasets (default: every one with results)")
    parser.add_argument("--out", help="writes the Markdown here (default: prints it)")
    args = parser.parse_args()
    results = load(args.resultsDir)
    if args.datasets:
        results = collections.OrderedDict((k, v) for k, v in results.items() if k in args.datasets)
    order = [name for name in rm.DATASETS if name in results] + [name for name in results if name not in rm.DATASETS]
    text = report(collections.OrderedDict((name, results[name]) for name in order), args.metric)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print("written to %s" % args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
