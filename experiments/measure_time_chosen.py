"""
Training and inference time of the WiSARD models of the grid of run_grid.py (one-shot and epochs,
each with iterative bleaching, DAB and rDAB), for every dataset and seed, all of them on the same
architecture: times are compared only between models of the same architecture.

Architecture (bits per feature and tuple size, hence the number of RAMs): one per dataset and
encoding, the same for the six models and for every seed. It is the one of the one-shot model with
the highest validation accuracy, averaged over the seeds of --configSeeds and over the three
bleaching modes (--archFrom picks one mode instead), among the architectures with one-shot and epoch
results in every mode and seed; ties: fewer bits, smaller tuple. The test accuracy takes no part.

Inside that architecture, for each seed and bleaching mode, the epoch model keeps the b, rho0 and tau
with the best validation accuracy of that seed (summarize.choose, same ties: weakest correction).
These are training settings, not architecture. A seed enters only when the three modes have both
models (as benchmark_table.ipynb, which leaves the others out of the comparison).

Each chosen configuration is trained again on the split, encoding and RAM mapping of run_grid.py,
so its test accuracy should repeat the one of the grid (column "acc grid"; a difference means the
grid ran another version of the code). Timed with time.perf_counter, the data already encoded:
  train (s)        Wisard.train / Wisard.trainEpochs; trainEpochs includes the validation of every
                   epoch (its early stopping), done with --wisardThreads threads. Wisard.train does
                   not use the bleaching policy, so the one-shot model of a seed is the same in the
                   three modes: its training is timed once, in the first mode measured (trainTimedIn,
                   the iterative one), and that time is the one of the three modes. Each mode still
                   trains its own model to classify with its policy (the policy is fixed when the
                   model is built); that training time is kept in secondsTrainOwn, out of the table
  inference (s)    classify of the test set, one thread; us/ex.: per test example
  epochs run       epochs of trainEpochs before early stopping (at most run_grid.EPOCHS)
  best epoch       the epoch of the delivered model, 1 = the first
The models run one after the other in this process; do not run other heavy jobs meanwhile.

Results: one JSON line per model in <outDir>/<dataset>/<encoding>/seed<k>.jsonl; the models already
there are skipped, so an interrupted run resumes where it stopped. At the end (or with --summary)
the table (mean +- std over the seeds) is printed and written to <outDir>/summary_<encoding>.md.

Usage (from the root of the repository):
  python experiments/measure_time_chosen.py --dry-run                 # what would run
  python experiments/measure_time_chosen.py                           # every dataset
  python experiments/measure_time_chosen.py --datasets higgs iris --seeds 1 2
  python experiments/measure_time_chosen.py --summary
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_time as mt  # noqa: E402
import run_grid as rg  # noqa: E402
import summarize as sm  # noqa: E402
from encoders import ENCODINGS  # noqa: E402

MODES = collections.OrderedDict([("bleaching", "Iterativo"), ("DAB", "DAB"), ("rDAB", "rDAB")])  # valMode -> name
KINDS = collections.OrderedDict([("oneshot", "One-shot"), ("epochs", "Épocas")])
SM_MODEL = {"oneshot": "one-shot + bleaching", "epochs": "epochs + bleaching"}  # summarize.choose
MODELS = ["%s-%s" % ("wisard" if kind == "oneshot" else "epochs", mode) for kind in KINDS for mode in MODES]
CONFIG_FIELDS = {"oneshot": ("bits", "address"), "epochs": ("bits", "address", "b", "rho0", "tau")}


# ---------------------------------------------------------------- chosen configurations

def grid_row(rows, kind, cfg):
    """the result of the grid with the chosen configuration"""
    return next(r for r in rows if r["kind"] == kind and all(r[k] == cfg[k] for k in CONFIG_FIELDS[kind]))


def architecture(rows, seeds, modes):
    """dict(bits, address, val, candidates): the architecture whose one-shot model has the highest
    validation accuracy averaged over the seeds and the modes, among those with one-shot and epoch
    results in every mode and seed of the dataset; None when there is none.
    rows: {mode: {seed: [results of the grid]}}"""
    oneshot = collections.defaultdict(dict)  # (bits, address) -> (mode, seed) -> validation accuracy
    present = collections.defaultdict(set)  # (bits, address) -> (kind, mode, seed) with results
    for mode in MODES:
        for seed in seeds:
            for r in rows[mode][seed]:
                arch = (r["bits"], r["address"])
                present[arch].add((r["kind"], mode, seed))
                if r["kind"] == "oneshot":
                    oneshot[arch][(mode, seed)] = r["bleaching"]["val"]["acc"]
    complete = [a for a in present if len(present[a]) == len(KINDS) * len(MODES) * len(seeds)]
    if not complete:
        return None

    def mean_val(arch):
        return float(np.mean([v for (mode, _), v in oneshot[arch].items() if mode in modes]))

    best = max(complete, key=lambda a: (mean_val(a), -a[0], -a[1]))  # ties: fewer bits, smaller tuple
    return dict(bits=best[0], address=best[1], val=mean_val(best), candidates=len(complete))


def chosen_tasks(args):
    """(tasks, seeds left out, architectures): one task per dataset, seed and model, all the models of
    a dataset on its architecture, with the results of the grid for that configuration"""
    by_mode = {m: sm.load(args.gridResults, m) for m in MODES}
    keys = sorted({k for d in by_mode.values() for k in d
                   if k[1] == args.encoding and (not args.datasets or k[0] in args.datasets)})
    arch_modes = list(MODES) if args.archFrom == "mean" else [args.archFrom]
    tasks, left_out, archs = [], [], collections.OrderedDict()
    for key in keys:
        name = key[0]
        rows, seeds = {m: {} for m in MODES}, []  # seeds of the comparison
        for seed in sorted({s for m in MODES for s in by_mode[m].get(key, {})}):
            seed_rows = {m: by_mode[m].get(key, {}).get(seed, []) for m in MODES}
            lacking = [MODES[m] for m in MODES if len(sm.choose(seed_rows[m])) < len(sm.MODELS)]
            if lacking:
                left_out.append("%s, seed %d: %s without both models" % (name, seed, " and ".join(lacking)))
                continue
            seeds.append(seed)
            for m in MODES:
                rows[m][seed] = seed_rows[m]
        config_seeds = [s for s in seeds if args.configSeeds is None or s in args.configSeeds]
        arch = architecture(rows, config_seeds, arch_modes) if config_seeds else None
        if arch is None:
            left_out.append("%s: no architecture with one-shot and epoch results in every mode and seed %s"
                            % (name, ",".join(map(str, config_seeds)) or "-"))
            continue
        archs[name] = dict(arch, seeds=config_seeds)

        for seed in seeds:
            if args.seeds is not None and seed not in args.seeds:
                continue
            for kind in KINDS:
                for mode in MODES:
                    model = "%s-%s" % ("wisard" if kind == "oneshot" else "epochs", mode)
                    if model not in args.models:
                        continue
                    same = [r for r in rows[mode][seed]
                            if r["kind"] == kind and (r["bits"], r["address"]) == (arch["bits"], arch["address"])]
                    if not same:  # a seed out of --configSeeds may lack the architecture
                        left_out.append("%s, seed %d: %s without %d/%d in the grid"
                                        % (name, seed, model, arch["bits"], arch["address"]))
                        continue
                    cfg, test, val = sm.choose(same)[SM_MODEL[kind]]
                    task = dict(dataset=name, encoding=args.encoding, seed=seed, model=model, kind=kind,
                                valMode=mode, bits=cfg["bits"], address=cfg["address"])
                    if kind == "epochs":
                        task.update(b=cfg["b"], rho0=cfg["rho0"], tau=cfg["tau"], epochs=rg.EPOCHS,
                                    patience=rg.PATIENCE, threads=args.wisardThreads,
                                    gridEpochsRun=cfg["ran"], gridBestEpoch=cfg["bestEpoch"])
                    task.update(gridVal=val["acc"], gridTest=test["acc"],
                                gridSeconds=grid_row(same, kind, cfg).get("seconds"))
                    tasks.append(task)
    # one dataset at a time, and the models of a seed together: the encoded data is reused
    tasks.sort(key=lambda t: (t["dataset"], t["seed"]))
    return tasks, left_out, archs


def print_archs(archs, args):
    print("architecture of every dataset (%s): the one-shot model with the best validation accuracy, mean over "
          "the seeds and %s" % (args.encoding, "the three bleaching modes" if args.archFrom == "mean"
                                 else "the mode " + MODES[args.archFrom]))
    print("  %-15s %-11s %-9s %-10s %s" % ("dataset", "bits/tuple", "mean val", "candidates", "seeds"))
    for name, a in archs.items():
        print("  %-15s %-11s %-9.4f %-10d %s" % (name, "%d/%d" % (a["bits"], a["address"]), a["val"],
                                               a["candidates"], ",".join(map(str, a["seeds"]))))


def config_text(t):
    text = "%d/%d" % (t["bits"], t["address"])
    if t["kind"] == "epochs":
        text += " b=%d rho0=%g tau=%g" % (t["b"], t["rho0"], t["tau"])
    return text


# ---------------------------------------------------------------- measurement

def oneshot_key(r):
    """the one-shot model of a seed and architecture: the same trained model in every bleaching mode"""
    return r["dataset"], r["encoding"], r["seed"], r["bits"], r["address"]


def timed_trainings(out_dir, tasks):
    """{oneshot_key: (seconds, valMode)}: the training time of the one-shot model of every seed and
    architecture, the first one measured (in the order of the files)"""
    timed = {}
    for path in sorted({rg.results_path(out_dir, t) for t in tasks}):
        if os.path.exists(path):
            with open(path) as f:
                for r in map(json.loads, f):
                    if r["kind"] == "oneshot" and "error" not in r:
                        timed.setdefault(oneshot_key(r), (r["secondsTrain"], r.get("trainTimedIn", r["valMode"])))
    return timed


def measure(task, timed):
    """timed: timed_trainings, completed with the one-shot models measured here"""
    result = dict(task)
    try:
        p = rg.prepared(task["dataset"], task["encoding"], task["seed"], task["bits"])
        result.update(mt.run_wisard(task, p))
        if task["kind"] == "oneshot":  # the training of the first mode measured is the one of the three
            key = oneshot_key(task)
            timed.setdefault(key, (result["secondsTrain"], task["valMode"]))
            result["secondsTrainOwn"] = result["secondsTrain"]
            result["secondsTrain"], result["trainTimedIn"] = timed[key]
        result["usPerSample"] = result["secondsInfer"] / result["testSize"] * 1e6
    except Exception as e:  # recorded and reported; the other models go on
        import traceback
        result["error"] = "%s: %s" % (type(e).__name__, e)
        result["traceback"] = traceback.format_exc()
    return result


# ---------------------------------------------------------------- table

def cell(values, fmt, comma=False):
    if not values:
        return "-"
    text = fmt % np.mean(values)
    if len(values) > 1:
        text += " ± " + fmt % np.std(values, ddof=1)
    return text.replace(".", ",") if comma else text


def table(out_dir, tasks):
    """rows of the table: one per dataset and model, mean ± std over the seeds measured"""
    wanted = {mt.task_key(t) for t in tasks}
    timed = timed_trainings(out_dir, tasks)
    groups = collections.defaultdict(list)
    for path in sorted({rg.results_path(out_dir, t) for t in tasks}):
        if os.path.exists(path):
            with open(path) as f:
                for r in map(json.loads, f):
                    if "error" not in r and mt.task_key(r) in wanted:
                        if r["kind"] == "oneshot":  # also for results written before trainTimedIn
                            r["secondsTrain"] = timed[oneshot_key(r)][0]
                        groups[(r["dataset"], r["model"])].append(r)
    order = {m: k for k, m in enumerate(MODELS)}
    return [(key, sorted(groups[key], key=lambda r: r["seed"]))
            for key in sorted(groups, key=lambda k: (k[0], order[k[1]]))]


COLUMNS = ["conjunto", "bits/tupla", "modelo", "bleaching", "sementes", "épocas rodadas", "melhor época", "treino (s)",
           "inferência (s)", "inferência (µs/ex.)", "acc teste", "acc grid"]


def summary_rows(rows, comma):
    out = []
    for (name, model), group in rows:
        kind, mode = model.split("-")
        epochs = [r["epochsRun"] for r in group if "epochsRun" in r]
        out.append([
            name, ", ".join(sorted({"%d/%d" % (r["bits"], r["address"]) for r in group})),
            KINDS["oneshot" if kind == "wisard" else "epochs"], MODES[mode],
            ",".join(str(r["seed"]) for r in group),
            cell(epochs, "%.1f", comma) if epochs else "-",
            cell([r["bestEpoch"] + 1 for r in group if "bestEpoch" in r], "%.1f", comma) if epochs else "-",
            cell([r["secondsTrain"] for r in group], "%.4f", comma),
            cell([r["secondsInfer"] for r in group], "%.4f", comma),
            cell([r["usPerSample"] for r in group], "%.2f", comma),
            cell([100 * r["accTest"] for r in group], "%.2f", comma),
            cell([100 * r["gridTest"] for r in group], "%.2f", comma)])
    return out


def summary(out_dir, tasks, encoding):
    rows = table(out_dir, tasks)
    if not rows:
        print("no results in %s for these configurations" % out_dir)
        return
    text = summary_rows(rows, comma=False)
    widths = [max(len(c), *(len(r[k]) for r in text)) for k, c in enumerate(COLUMNS)]
    print("\n%s: every model of a dataset on the same architecture (mean ± std over the seeds; accuracies in %%)"
          % encoding)
    for line in [COLUMNS] + text:
        print("  ".join(c.ljust(w) for c, w in zip(line, widths)))

    path = os.path.join(out_dir, "summary_%s.md" % encoding)
    md = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    md += ["| " + " | ".join(r) + " |" for r in summary_rows(rows, comma=True)]
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Tempo de treino e inferência (%s), todos os modelos de um conjunto na mesma arquitetura "
                "(bits/tupla: a do melhor one-shot na validação, a mesma em todas as sementes); b, rho0 e tau "
                "do modelo por épocas escolhidos na validação de cada semente; o treino one-shot não depende do "
                "bleaching e é medido uma vez por semente, o mesmo nos três modos; média ± dp entre as sementes, "
                "acurácias em %%.\n\n" % encoding + "\n".join(md) + "\n")
    print("table written to %s" % path)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoding", default="distributive", choices=ENCODINGS)
    parser.add_argument("--datasets", nargs="+", help="default: every dataset with results in --gridResults")
    parser.add_argument("--seeds", type=int, nargs="+", help="seeds measured; default: every seed of the comparison")
    parser.add_argument("--configSeeds", type=int, nargs="+",
                        help="seeds whose mean validation accuracy chooses the architecture; default: every seed of "
                             "the comparison (--seeds does not change the architecture)")
    parser.add_argument("--archFrom", default="mean", choices=["mean"] + list(MODES),
                        help="one-shot validation that chooses the architecture: mean of the three bleaching modes, "
                             "or one of them")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--gridResults", default="experiments/results", help="results of run_grid.py")
    parser.add_argument("--wisardThreads", type=int, default=8,
                        help="trainEpochs: threads of the validation of each epoch and of the addresses (0: every core)")
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--outDir", default="experiments/results_time_chosen")
    parser.add_argument("--dry-run", action="store_true", help="only list what would run")
    parser.add_argument("--summary", action="store_true", help="only print the table of the results already measured")
    args = parser.parse_args()

    tasks, left_out, archs = chosen_tasks(args)
    if left_out:
        print("left out:\n  " + "\n  ".join(left_out))
    print_archs(archs, args)
    if args.summary:
        summary(args.outDir, tasks, args.encoding)
        return

    done = mt.done_keys(args.outDir, tasks)
    pending = [t for t in tasks if mt.task_key(t) not in done]
    print("\n%s: %d models to measure, %d already measured, %d to run" % (
        args.encoding, len(tasks), len(tasks) - len(pending), len(pending)))
    per_dataset = collections.OrderedDict()
    for t in pending:
        n, seconds = per_dataset.get(t["dataset"], (0, 0.0))
        per_dataset[t["dataset"]] = (n + 1, seconds + (t["gridSeconds"] or 0))
    for name, (n, seconds) in per_dataset.items():
        print("  %-15s %3d models   %7.1f min in the grid" % (name, n, seconds / 60))
    if per_dataset:
        print("  grid time of these models: %.1f min (older code, with several grid workers sharing the CPU,"
              " and classifying validation and test; this run should take less)"
              % (sum(s for _, s in per_dataset.values()) / 60))
    if args.dry_run or not pending:
        if args.dry_run:
            for t in pending:
                print("  %-15s seed %d  %-17s %s" % (t["dataset"], t["seed"], t["model"], config_text(t)))
        else:
            summary(args.outDir, tasks, args.encoding)
        return

    rg.init_worker(dict(dataDir=args.dataDir))
    timed = timed_trainings(args.outDir, tasks)
    start, errors = time.time(), 0
    for n, task in enumerate(pending, 1):
        result = measure(task, timed)
        path = rg.results_path(args.outDir, result)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
        head = "[%d/%d %.0fs] %-15s seed %d %-17s %-28s" % (
            n, len(pending), time.time() - start, task["dataset"], task["seed"], task["model"], config_text(task))
        if "error" in result:
            errors += 1
            print("%s ERROR %s\n%s" % (head, result["error"], result["traceback"]), flush=True)
        else:
            extra = "  epochs %d" % result["epochsRun"] if "epochsRun" in result else ""
            if result.get("trainTimedIn", task["valMode"]) != task["valMode"]:
                extra = "  (train of %s; own %.3fs)" % (MODES[result["trainTimedIn"]], result["secondsTrainOwn"])
            print("%s train %8.3fs  infer %7.3fs  acc test %.4f (grid %.4f)%s" % (
                head, result["secondsTrain"], result["secondsInfer"], result["accTest"], result["gridTest"], extra),
                flush=True)
    print("finished: %d measured, %d errors, %.1f min" % (len(pending), errors, (time.time() - start) / 60))
    summary(args.outDir, tasks, args.encoding)


if __name__ == "__main__":
    main()
