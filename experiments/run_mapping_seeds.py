"""
Mapping-seed exploration of the best base configuration of each dataset.

For every dataset, best_architecture.best_configs chooses, on the grid results of run_grid.py
(--gridDir), the configuration with the best mean validation accuracy over the grid seeds. That
configuration is then trained and evaluated with every mapping seed of --seeds, keeping the rest
fixed, so the spread of the results is the spread of the RAM mapping only:

  split          fixed by --splitSeed (the same split of run_grid.py with seed=splitSeed)
  RAM mapping    the mapping seed (the same mapping of run_grid.py: RandomState(10007 + seed))
  epoch shuffle  fixed by --splitSeed (kind epochs)

So mapping seed s with --splitSeed s reproduces the run_grid.py result of seed s.

Every run is one JSON line in <resultsDir>/<dataset>/<encoding>/split<k>.jsonl, with the
configuration, splitSeed and mappingSeed. Runs already in the files are skipped, so an interrupted
run resumes where it stopped; lines with "error" are run again. If the best configuration of a
dataset changes (new grid results), its runs get new keys and the old ones stay in the file.

With --trainings, the architecture of the best configuration (encoding, bits, address, valMode)
is kept and every chosen training is run on it for every mapping seed: "oneshot" and/or "epochs"
(every b x rho0 x tau of the grid of run_grid.py, or of --b/--rho0/--tau).

Usage:
  python experiments/run_mapping_seeds.py --seeds $(seq 0 99) --jobs 10
  python experiments/run_mapping_seeds.py --valModes bleaching --trainings oneshot epochs
  python experiments/run_mapping_seeds.py --datasets blood iris --seeds 0 1 2 --valModes DAB rDAB
  python experiments/run_mapping_seeds.py --kind epochs --seeds $(seq 0 29) --dry-run
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import best_architecture  # noqa: E402
import datasets  # noqa: E402
import run_grid as rg  # noqa: E402
from encoders import ENCODINGS  # noqa: E402

import wisardpkg as wp  # noqa: E402

CONFIG_FIELDS = ("kind", "encoding", "bits", "address", "valMode", "b", "rho0", "tau")
# every line has these keys in this order, one-shot or epochs (the epoch fields are None for the
# one-shot); "error" is added at the end of failed runs
ROW_FIELDS = ("encoding", "bits", "address", "valMode", "kind", "b", "rho0", "tau", "dataset", "splitSeed",
              "mappingSeed", "selectedOn", "bestEpoch", "accValPerEpoch", "nPenError", "nPenMargin", "bleaching",
              "size", "seconds")
SELECTED_FIELDS = ("gridDir", "kind", "seeds", "val", "test")


def normalize(row):
    """the line of a run with the keys of ROW_FIELDS, in order"""
    out = {k: row.get(k) for k in ROW_FIELDS}
    out["selectedOn"] = {k: (row.get("selectedOn") or {}).get(k) for k in SELECTED_FIELDS}
    if "error" in row:
        out["error"] = row["error"]
    return out


def task_key(t):
    return rg.task_key(t) + "|mappingSeed=%d" % t["mappingSeed"]


def results_path(results_dir, t):
    return os.path.join(results_dir, t["dataset"], t["encoding"], "split%d.jsonl" % t["splitSeed"])


def run_task(task):
    start = time.time()
    result = dict(task)
    try:
        p = rg.prepared(task["dataset"], task["encoding"], task["splitSeed"], task["bits"])
        model = wp.Wisard(task["address"], returnClassesDegrees=True, bleaching=rg.POLICY[task["valMode"]],
                          mapping=rg.mapping(p["size"], task["address"], task["mappingSeed"], p["labels"]),
                          **rg.DAB_OPTIONS)
        if task["kind"] == "oneshot":
            model.train(p["train"])
        else:
            report = model.trainEpochs(p["train"], p["val"], b=task["b"], rho0=task["rho0"], tau=task["tau"],
                                       epochs=rg.EPOCHS, patience=rg.PATIENCE, valMode="bleaching",
                                       seed=task["splitSeed"])
            result["bestEpoch"] = report["bestEpoch"]
            result["accValPerEpoch"] = [row["accVal"] for row in report["perEpoch"]]
            result["nPenError"] = sum(row["nPenError"] for row in report["perEpoch"])
            result["nPenMargin"] = sum(row["nPenMargin"] for row in report["perEpoch"])
        result["bleaching"] = dict(val=rg.evaluate_bleaching(model, p["val"], p["yval"]),
                                   test=rg.evaluate_bleaching(model, p["test"], p["ytest"]))
        result["size"] = p["size"]
    except Exception as e:  # recorded and reported; the rest goes on
        result["error"] = "%s: %s" % (type(e).__name__, e)
    result["seconds"] = time.time() - start
    return result


def trainings(best_config, args):
    """the training settings run on the architecture of the best configuration: the one-shot and/or
    every epoch configuration of the grid (b x rho0 x tau); without --trainings, the best
    configuration itself (with its own b, rho0, tau when --kind epochs)"""
    if not args.trainings:
        return [{k: best_config[k] for k in ("kind", "b", "rho0", "tau")}]
    out = []
    if "oneshot" in args.trainings:
        out.append(dict(kind="oneshot", b=None, rho0=None, tau=None))
    if "epochs" in args.trainings:
        out += [dict(kind="epochs", b=b, rho0=rho0, tau=tau) for rho0 in args.rho0 for tau in args.tau for b in args.b]
    return out


def build_tasks(best, args):
    """tasks ordered by dataset (the workers keep one dataset in memory), then mapping seed"""
    tasks = []
    for name in args.datasets:
        if name not in best:
            continue
        architecture = {k: best[name][k] for k in ("encoding", "bits", "address", "valMode")}
        for seed in args.seeds:
            for training in trainings(best[name], args):
                tasks.append(dict(architecture, **training, dataset=name, splitSeed=args.splitSeed, mappingSeed=seed,
                                  selectedOn=dict(gridDir=args.gridDir, kind=args.kind, seeds=best[name]["seeds"],
                                                  val=best[name]["val"], test=best[name]["test"])))
    return tasks


def done_keys(results_dir, tasks):
    done = set()
    for path in set(results_path(results_dir, t) for t in tasks):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if "error" not in r:
                        done.add((path, task_key(r)))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(datasets.NAMES), choices=datasets.NAMES)
    parser.add_argument("--encodings", nargs="+", default=list(ENCODINGS), choices=ENCODINGS,
                        help="encodings the best configuration may use (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)), help="mapping seeds to run")
    parser.add_argument("--valModes", nargs="+", default=rg.VAL_MODES, choices=rg.VAL_MODES,
                        help="validation modes the best configuration may use (default: all)")
    parser.add_argument("--kind", default="oneshot", choices=("oneshot", "epochs"),
                        help="training whose grid results choose the architecture (encoding, bits, address, valMode)")
    parser.add_argument("--trainings", nargs="+", choices=("oneshot", "epochs"),
                        help="trainings run on the chosen architecture: the one-shot and/or every epoch configuration "
                             "(--b x --rho0 x --tau); default: only the best configuration itself")
    parser.add_argument("--b", type=int, nargs="+", default=rg.B_GRID)
    parser.add_argument("--rho0", type=float, nargs="+", default=rg.RHO_GRID)
    parser.add_argument("--tau", type=float, nargs="+", default=rg.TAU_GRID)
    parser.add_argument("--splitSeed", type=int, default=0, help="seed of the split, fixed for every mapping seed")
    parser.add_argument("--gridDir", default="experiments/results", help="run_grid.py results used to choose the configuration")
    parser.add_argument("--gridSeeds", type=int, nargs="+", help="grid seeds used to choose (default: all found)")
    parser.add_argument("--minSeeds", type=int, help="grid seeds a configuration needs to compete (default: the most seen)")
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--resultsDir", default="experiments/results_mapping")
    parser.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument("--dry-run", action="store_true", help="print the chosen configurations and the counts, run nothing")
    args = parser.parse_args()

    best = best_architecture.best_configs(args.gridDir, args.kind, args.encodings, args.valModes, args.gridSeeds,
                                          args.minSeeds, args.datasets)
    print("best configuration per dataset (chosen on %s):" % args.gridDir)
    for name in args.datasets:
        if name in best:
            c = best[name]
            print("  %-14s %s  (grid seeds %s, val %.4f, test %.4f)" % (
                name, " ".join("%s=%s" % (k, c[k]) for k in CONFIG_FIELDS if c[k] is not None),
                ",".join(map(str, c["seeds"])), c["val"], c["test"]))
        else:
            print("  %-14s no grid results: skipped" % name)

    tasks = build_tasks(best, args)
    done = done_keys(args.resultsDir, tasks)
    pending = [t for t in tasks if (results_path(args.resultsDir, t), task_key(t)) not in done]
    print("%d runs, %d already done, %d to run with %d jobs" % (len(tasks), len(tasks) - len(pending), len(pending), args.jobs))
    if args.dry_run or not pending:
        return

    pool = multiprocessing.Pool(args.jobs, initializer=rg.init_worker, initargs=(dict(dataDir=args.dataDir),))
    start, errors = time.time(), 0
    try:
        for n, result in enumerate(pool.imap_unordered(run_task, pending), 1):
            result = normalize(result)
            path = results_path(args.resultsDir, result)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(result) + "\n")
            if "error" in result:
                errors += 1
                print("ERROR %s %s: %s" % (result["dataset"], task_key(result), result["error"]))
            elapsed = time.time() - start
            print("[%d/%d %.0fs, eta %.1fh] %s %s split%d %s  val %.4f  test %.4f  (%.0fs)" % (
                n, len(pending), elapsed, elapsed / n * (len(pending) - n) / 3600, result["dataset"], result["encoding"],
                result["splitSeed"], task_key(result), result.get("bleaching", {}).get("val", {}).get("acc") or 0,
                result.get("bleaching", {}).get("test", {}).get("acc") or 0, result["seconds"]), flush=True)
    finally:
        pool.terminate()
    print("finished: %d results, %d errors, %.1f h" % (len(pending), errors, (time.time() - start) / 3600))


if __name__ == "__main__":
    main()
