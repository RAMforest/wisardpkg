"""
RAM used while training (and classifying) configurations of the grid benchmark, the one-shot
WiSARD and the WiSARD trained by epochs. The configurations come from the results of
run_grid.py and are trained again exactly as there (same split, encoding, mapping and seed).

Every configuration runs in a new process, and the memory is read from the operating system:

  rssBefore   resident memory after the data is encoded and the model created, before training
  peakTrain   peak resident memory of the process at the end of the training
  trainBytes  peakTrain - rssBefore: RAM added by the training at its worst moment (counters,
              the float counters and the copy of the best epoch in epoch training, temporaries)
  peakInfer   peak at the end of the classification of validation and test (bleaching)
  inferBytes  peakInfer - resident memory after training: RAM added by the classification
  modelBytes  Wisard.getsizeof() of the trained model (stored addresses, no hash-map overhead)

The peak of a process only grows, so a peak reached while the data was encoded can hide the one
of the training: trainValid / inferValid are false when the peak did not grow during that phase
(then trainBytes / inferBytes are only upper bounds). Tiny datasets are often in this case.
tracemalloc cannot be used (the RAMs are C++ maps), nor a sampling thread (the bindings keep the
GIL while training).

Which configurations (--select):
  best          (default) per dataset, encoding and seed: the one-shot and the epoch configuration
                with the best validation accuracy, the ones summarize.py and benchmark_table.ipynb
                report (same ties)
  architecture  for every dataset, encoding, seed, bits and address: the one-shot and the best
                epoch configuration on validation

Results: one JSON line per configuration in <outDir>/<dataset>/<encoding>/seed<k>.jsonl;
configurations already there are skipped, so an interrupted run resumes where it stopped.
Parallel jobs compete for the RAM of the machine (Windows trims the working sets under pressure,
which lowers the peaks); keep --jobs low, 1 for the most reliable numbers.

Usage:
  python experiments/measure_memory.py --dry-run                     # how many configurations, estimate
  python experiments/measure_memory.py --datasets iris blood --seeds 1 --jobs 2
  python experiments/measure_memory.py --noInfer --seeds 1            # only the training
"""
import argparse
import collections
import gc
import json
import multiprocessing
import os
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_grid as rg  # noqa: E402

import wisardpkg as wp  # noqa: E402

MB = float(2 ** 20)


def memory():
    """(resident memory, peak resident memory) of this process, in bytes"""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows: peak working set
        return info.rss, info.peak_wset
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux, bytes on macOS
    return info.rss, peak * (1 if sys.platform == "darwin" else 1024)


# ---------------------------------------------------------------- task (run in a new process)

def measure_task(task):
    result = dict(task)
    try:
        p = rg.prepared(task["dataset"], task["encoding"], task["seed"], task["bits"])
        rg._raw.clear()  # the raw features are no longer needed
        model = rg.make_model(task, p)  # with the bleaching of the validation mode of the task
        gc.collect()
        before, peak_before = memory()

        start = time.time()
        if task["kind"] == "oneshot":
            model.train(p["train"])
        else:
            model.trainEpochs(p["train"], p["val"], b=task["b"], rho0=task["rho0"], tau=task["tau"],
                              epochs=rg.EPOCHS, patience=rg.PATIENCE, valMode="bleaching", seed=task["seed"])
        result["secondsTrain"] = time.time() - start
        after_train, peak_train = memory()

        result.update(
            rssBefore=before, peakBefore=peak_before,
            peakTrain=peak_train, trainBytes=peak_train - before, trainValid=peak_train > peak_before,
            rssAfterTrain=after_train, modelBytes=model.getsizeof())

        if not task["noInfer"]:
            start = time.time()
            val = rg.evaluate_bleaching(model, p["val"], p["yval"])
            test = rg.evaluate_bleaching(model, p["test"], p["ytest"])
            result["secondsInfer"] = time.time() - start
            _, peak_infer = memory()
            result.update(peakInfer=peak_infer, inferBytes=peak_infer - after_train, inferValid=peak_infer > peak_train,
                          valAcc=val["acc"], testAcc=test["acc"])
    except Exception as e:  # recorded and reported; the rest goes on
        result["error"] = "%s: %s" % (type(e).__name__, e)
    return result


# ---------------------------------------------------------------- task list

def oneshot_key(r):
    return (r["bleaching"]["val"]["acc"], -r["bits"], -r["address"])


def epochs_key(r):
    return (r["bleaching"]["val"]["acc"], -r["rho0"], -r["tau"], -r["b"], -r["bits"], -r["address"])


def select(rows, mode):
    """the rows of one dataset/encoding/seed to measure"""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["bits"], r["address"]) if mode == "architecture" else None].append(r)
    chosen = []
    for _, group in sorted(groups.items(), key=lambda kv: kv[0] or (0, 0)):
        oneshot = [r for r in group if r["kind"] == "oneshot"]
        epochs = [r for r in group if r["kind"] == "epochs"]
        chosen += [max(oneshot, key=oneshot_key)] if oneshot else []
        chosen += [max(epochs, key=epochs_key)] if epochs else []
    return chosen


def build_tasks(args):
    import summarize  # only here: the workers do not need scipy
    tasks = []
    for (dataset, encoding), seeds in sorted(summarize.load(args.resultsDir, args.valMode).items()):
        if (args.datasets and dataset not in args.datasets) or (args.encodings and encoding not in args.encodings):
            continue
        for seed, rows in sorted(seeds.items()):
            if args.seeds is not None and seed not in args.seeds:
                continue
            rows = [r for r in rows if (not args.bits or r["bits"] in args.bits)
                    and (not args.address or r["address"] in args.address)]
            for r in select(rows, args.select):
                tasks.append(dict(dataset=dataset, encoding=encoding, seed=seed, kind=r["kind"], bits=r["bits"],
                                  address=r["address"], b=r["b"], rho0=r["rho0"], tau=r["tau"], valMode=args.valMode,
                                  gridSeconds=r["seconds"], noInfer=args.noInfer))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", choices=("best", "architecture"), default="best")
    parser.add_argument("--valMode", default="bleaching", choices=rg.VAL_MODES,
                        help="validation mode of run_grid.py whose configurations are measured")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--encodings", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: every seed in the results")
    parser.add_argument("--bits", type=int, nargs="+")
    parser.add_argument("--address", type=int, nargs="+")
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--resultsDir", default="experiments/results", help="results of run_grid.py")
    parser.add_argument("--outDir", default="experiments/results_memory")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--noInfer", action="store_true",
                        help="measure only the training (the bleaching of the one-shot with small addresses is slow)")
    parser.add_argument("--dry-run", action="store_true", help="only count the configurations and estimate the time")
    args = parser.parse_args()

    tasks = build_tasks(args)
    done = rg.done_keys(args.outDir, tasks)
    pending = [t for t in tasks if (rg.results_path(args.outDir, t), rg.task_key(t)) not in done]
    counts = collections.Counter((t["dataset"], t["encoding"]) for t in pending)
    for (dataset, encoding), n in sorted(counts.items()):
        print("  %-14s %-12s %5d configurations, %6.2f h in the grid" % (dataset, encoding, n, sum(
            t["gridSeconds"] for t in pending if (t["dataset"], t["encoding"]) == (dataset, encoding)) / 3600))
    grid_hours = sum(t["gridSeconds"] for t in pending) / 3600
    print("%d configurations, %d already measured, %d to run with %d jobs; about %.1f h%s "
          "(their time in the grid, training + classification, plus the encoding of the data in every process)"
          % (len(tasks), len(tasks) - len(pending), len(pending), args.jobs, grid_hours / args.jobs,
             ", less with --noInfer" if args.noInfer else ""))
    if args.dry_run or not pending:
        return

    # maxtasksperchild=1: every configuration in a new process, so the peaks are its own
    pool = multiprocessing.Pool(args.jobs, initializer=rg.init_worker, initargs=(dict(dataDir=args.dataDir),),
                                maxtasksperchild=1)
    start, errors = time.time(), 0
    try:
        for n, result in enumerate(pool.imap_unordered(measure_task, pending), 1):
            path = rg.results_path(args.outDir, result)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(result) + "\n")
            elapsed = time.time() - start
            head = "[%d/%d %.0fs, eta %.1fh] %s %s seed%d %s" % (
                n, len(pending), elapsed, elapsed / n * (len(pending) - n) / 3600,
                result["dataset"], result["encoding"], result["seed"], rg.task_key(result))
            if "error" in result:
                errors += 1
                print("%s  ERROR %s" % (head, result["error"]), flush=True)
                continue
            line = "%s  train +%.1f MB%s  model %.1f MB (%.0fs)" % (
                head, result["trainBytes"] / MB, "" if result["trainValid"] else " (upper bound)",
                result["modelBytes"] / MB, result["secondsTrain"])
            if "inferBytes" in result:
                line += "  infer +%.1f MB%s  test %.4f (%.0fs)" % (
                    result["inferBytes"] / MB, "" if result["inferValid"] else " (upper bound)",
                    result["testAcc"], result["secondsInfer"])
            print(line, flush=True)
    finally:
        pool.terminate()
    print("finished: %d measured, %d errors, %.1f h" % (len(pending), errors, (time.time() - start) / 3600))


if __name__ == "__main__":
    main()
