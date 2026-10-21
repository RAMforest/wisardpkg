"""
Grid search benchmark: one-shot WiSARD vs epoch training (trainEpochs), for every dataset,
encoding and seed. The configuration is chosen later, on validation, by summarize.py.

Every configuration of the grid is trained and evaluated on validation and test, and written as
one JSON line in <resultsDir>/<dataset>/<encoding>/seed<k>.jsonl. Configurations already in the
files are skipped, so an interrupted run resumes where it stopped.

Grid (joint, repeated for every seed):
  encoding bits x address size                       -> one-shot, classified with iterative
                                                        bleaching
  encoding bits x address size x b x rho0 x tau      -> trainEpochs with early stopping on
                                                        validation, classified with bleaching
Splits: the official test set when the dataset has one (MNIST, FashionMNIST, CIFAR-10), with
15% of the official training set as validation; otherwise a stratified 70/15/15 split. The split,
the RAM mapping and the shuffling of the epochs change with the seed.

Usage:
  python experiments/run_grid.py --datasets iris blood --jobs 10
  python experiments/run_grid.py --pilot      # a few configurations per dataset + time estimate
"""
import argparse
import collections
import json
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
from encoders import ENCODINGS, Encoder, to_dataset  # noqa: E402

import wisardpkg as wp  # noqa: E402

GRID = {
    "tabular": dict(bits=[2, 4, 8, 16, 32], address=[2, 4, 8, 16, 24, 32]),
    "image": dict(bits=[2, 4, 8, 16], address=[8, 16, 24, 32, 40]),
}
B_GRID = [1, 2]
RHO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
TAU_GRID = [0.0, 0.05, 0.1, 0.2]
EPOCHS, PATIENCE, VAL_MODE = 20, 3, "bleaching"


# ---------------------------------------------------------------- splits and models

def stratified(y, rng, fractions):
    parts = [[] for _ in fractions]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        start = 0
        for k, fraction in enumerate(fractions):
            end = len(idx) if k == len(fractions) - 1 else start + int(round(fraction * len(idx)))
            parts[k].append(idx[start:end])
            start = end
    return [np.sort(np.concatenate(p)) for p in parts]


def split(ds, seed):
    rng = np.random.RandomState(seed)
    if ds.has_official_test:
        train, val = stratified(ds.y, rng, (0.85, 0.15))
        return (ds.X[train], ds.y[train]), (ds.X[val], ds.y[val]), (ds.X_test, ds.y_test)
    train, val, test = stratified(ds.y, rng, (0.70, 0.15, 0.15))
    return (ds.X[train], ds.y[train]), (ds.X[val], ds.y[val]), (ds.X[test], ds.y[test])


def mapping(size, address, seed, labels):
    """complete addressing: a seeded permutation of the bits, the last RAM filled with random bits"""
    rng = np.random.RandomState(10007 + seed)
    rams = -(-size // address)
    order = np.concatenate([rng.permutation(size), rng.randint(0, size, rams * address - size)])
    indexes = order.reshape(rams, address).tolist()
    return {label: indexes for label in labels}


def confidence(values):
    """C = (R_w - R_l) / R_w; 0 on ties"""
    top = sorted(values, reverse=True) + [0]
    return (top[0] - top[1]) / float(top[0]) if top[0] > 0 else 0.0


def metrics(predicted, confidences, expected):
    hits = [p == t for p, t in zip(predicted, expected)]
    mean = lambda v: float(np.mean(v)) if len(v) else None
    return dict(acc=mean(hits), conf=mean(confidences),
                conf_hit=mean([c for c, h in zip(confidences, hits) if h]),
                conf_miss=mean([c for c, h in zip(confidences, hits) if not h]))


def evaluate_bleaching(model, data, labels):
    out = model.classify(data)
    return metrics([o["class"] for o in out], [confidence([c["degree"] for c in o["classesDegrees"]]) for o in out], labels)


# ---------------------------------------------------------------- tasks (run in the workers)

_raw = collections.OrderedDict()
_prepared = collections.OrderedDict()
_options = {}


def init_worker(options):
    _options.update(options)


def raw_dataset(name):
    if name not in _raw:
        _raw.clear()  # tasks come ordered by dataset; keep only one in memory
        _raw[name] = datasets.load(name, _options["dataDir"])
    return _raw[name]


def prepared(name, encoding, seed, bits):
    key = (name, encoding, seed, bits)
    if key not in _prepared:
        ds = raw_dataset(name)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(ds, seed)
        encoder = Encoder(encoding, bits).fit(Xtr, ds.nominal)
        _prepared[key] = dict(
            size=encoder.size, labels=sorted(set(ytr)),
            train=to_dataset(encoder.transform(Xtr), ytr, encoder.size),
            val=to_dataset(encoder.transform(Xva), yva, encoder.size), yval=list(yva),
            test=to_dataset(encoder.transform(Xte), yte, encoder.size), ytest=list(yte))
        while len(_prepared) > 2:
            _prepared.popitem(last=False)
    _prepared.move_to_end(key)
    return _prepared[key]


def task_key(t):
    return "%(kind)s|bits=%(bits)d|address=%(address)d|b=%(b)s|rho0=%(rho0)s|tau=%(tau)s" % t


def run_task(task):
    start = time.time()
    result = dict(task)
    try:
        p = prepared(task["dataset"], task["encoding"], task["seed"], task["bits"])
        model = wp.Wisard(task["address"], returnClassesDegrees=True,
                          mapping=mapping(p["size"], task["address"], task["seed"], p["labels"]))
        if task["kind"] == "oneshot":
            model.train(p["train"])
            result["bleaching"] = dict(val=evaluate_bleaching(model, p["val"], p["yval"]),
                                       test=evaluate_bleaching(model, p["test"], p["ytest"]))
        else:
            report = model.trainEpochs(p["train"], p["val"], b=task["b"], rho0=task["rho0"], tau=task["tau"],
                                       epochs=EPOCHS, patience=PATIENCE, valMode=VAL_MODE, seed=task["seed"])
            result["bleaching"] = dict(val=evaluate_bleaching(model, p["val"], p["yval"]),
                                       test=evaluate_bleaching(model, p["test"], p["ytest"]))
            result["bestEpoch"] = report["bestEpoch"]
            result["accValPerEpoch"] = [row["accVal"] for row in report["perEpoch"]]
            result["nPenError"] = sum(row["nPenError"] for row in report["perEpoch"])
            result["nPenMargin"] = sum(row["nPenMargin"] for row in report["perEpoch"])
        result["size"] = p["size"]
    except Exception as e:  # recorded and reported; the rest of the grid goes on
        result["error"] = "%s: %s" % (type(e).__name__, e)
    result["seconds"] = time.time() - start
    return result


# ---------------------------------------------------------------- task list

def entry_size(name, bits, data_dir):
    ds = datasets.load(name, data_dir)
    categories = sum(len(np.unique(ds.X[:, j])) for j in np.where(ds.nominal)[0])
    return int((~ds.nominal).sum()) * bits + categories


def build_tasks(args):
    tasks = []
    for name in args.datasets:
        grid = GRID["image" if name in datasets.IMAGES else "tabular"]
        bits_grid = args.bits or grid["bits"]
        address_grid = args.address or grid["address"]
        sizes = {bits: entry_size(name, bits, args.dataDir) for bits in bits_grid}
        epoch_grid = [(b, rho0, tau) for rho0 in RHO_GRID for tau in TAU_GRID for b in B_GRID]
        if args.pilot:  # the weakest and the strongest correction of the grid
            epoch_grid = [epoch_grid[0], epoch_grid[-1]]
        for seed in ([args.seeds[0]] if args.pilot else args.seeds):
            for encoding in args.encodings:
                for bits in bits_grid:
                    for address in address_grid:
                        if address > sizes[bits]:
                            continue
                        base = dict(dataset=name, encoding=encoding, seed=seed, bits=bits, address=address)
                        tasks.append(dict(base, kind="oneshot", b=None, rho0=None, tau=None))
                        for b, rho0, tau in epoch_grid:
                            tasks.append(dict(base, kind="epochs", b=b, rho0=rho0, tau=tau))
    return tasks


def results_path(results_dir, t):
    return os.path.join(results_dir, t["dataset"], t["encoding"], "seed%d.jsonl" % t["seed"])


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


def estimate(results_dir, args):
    """time of the full grid from the pilot: mean seconds per epoch configuration of each
    (dataset, encoding, bits, address), times the configurations and seeds of the full grid"""
    full = (len(B_GRID) * len(RHO_GRID) * len(TAU_GRID)) * len(args.seeds)
    total = 0.0
    print("\nestimate of the full grid (%d epoch configurations x %d seeds per bits/address), %d jobs:"
          % (full // len(args.seeds), len(args.seeds), args.jobs))
    for name in args.datasets:
        for encoding in args.encodings:
            path = os.path.join(results_dir, name, encoding, "seed%d.jsonl" % args.seeds[0])
            if not os.path.exists(path):
                continue
            groups = collections.defaultdict(list)
            oneshot = 0.0
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["kind"] == "epochs":
                        groups[(r["bits"], r["address"])].append(r["seconds"])
                    else:
                        oneshot += r["seconds"]
            seconds = sum(np.mean(v) * full for v in groups.values()) + oneshot * len(args.seeds)
            total += seconds
            print("  %-14s %-12s %8.1f h of CPU  (%s)" % (name, encoding, seconds / 3600, ", ".join(
                "%db/a%d: %.0fs" % (k[0], k[1], np.mean(v)) for k, v in sorted(groups.items()))))
    print("  total: %.1f h of CPU, about %.1f h with %d jobs" % (total / 3600, total / 3600 / args.jobs, args.jobs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(datasets.NAMES), choices=datasets.NAMES)
    parser.add_argument("--encodings", nargs="+", default=list(ENCODINGS), choices=ENCODINGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--bits", type=int, nargs="+", help="overrides the bits grid")
    parser.add_argument("--address", type=int, nargs="+", help="overrides the address grid")
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--resultsDir", default="experiments/results")
    parser.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument("--pilot", action="store_true",
                        help="first seed, every one-shot and 2 epoch configurations per bits/address; "
                             "then estimates the time of the full grid")
    args = parser.parse_args()

    tasks = build_tasks(args)
    done = done_keys(args.resultsDir, tasks)
    pending = [t for t in tasks if (results_path(args.resultsDir, t), task_key(t)) not in done]
    print("%d configurations, %d already done, %d to run with %d jobs" % (len(tasks), len(tasks) - len(pending), len(pending), args.jobs))

    pool = multiprocessing.Pool(args.jobs, initializer=init_worker, initargs=(dict(dataDir=args.dataDir),))
    start, errors = time.time(), 0
    try:
        for n, result in enumerate(pool.imap_unordered(run_task, pending), 1):
            path = results_path(args.resultsDir, result)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(result) + "\n")
            if "error" in result:
                errors += 1
                print("ERROR %s %s: %s" % (result["dataset"], task_key(result), result["error"]))
            elapsed = time.time() - start
            print("[%d/%d %.0fs, eta %.1fh] %s %s seed%d %s  val %.4f  test %.4f  (%.0fs)" % (
                n, len(pending), elapsed, elapsed / n * (len(pending) - n) / 3600, result["dataset"], result["encoding"],
                result["seed"], task_key(result), result.get("bleaching", {}).get("val", {}).get("acc") or 0,
                result.get("bleaching", {}).get("test", {}).get("acc") or 0, result["seconds"]), flush=True)
    finally:
        pool.terminate()
    print("finished: %d results, %d errors, %.1f h" % (len(pending), errors, (time.time() - start) / 3600))
    if args.pilot:
        estimate(args.resultsDir, args)


if __name__ == "__main__":
    main()
