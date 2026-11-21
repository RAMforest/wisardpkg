"""
Training and inference time on higgs of the best configuration of every WiSARD model and of a DWN
trained on the CPU, over --seeds (default 0..9):

  wisard-bleaching / wisard-DAB / wisard-rDAB   one-shot WiSARD (Wisard.train), classified with
                                                iterative bleaching, DAB or rDAB
  epochs-bleaching / epochs-DAB / epochs-rDAB   WiSARD trained by epochs (Wisard.trainEpochs); the
                                                validation of every epoch and the classification
                                                use that bleaching, as in run_grid.py
  dwn                                           DWN trained by the CPU kernel efd_fused_cpu
                                                (experiments/dwn_higgs_cpu), the model of dwn_higgs2.py

Configuration of every WiSARD model: the best of the grid of run_grid.py on higgs (--gridResults),
the one with the highest mean validation accuracy over --configSeeds (default 1 2 3 4: seed 0 ran an
older grid), among the configurations present in all of them; ties: fewer bits, smaller tuple. The
test accuracy takes no part in the choice. One-shot: bits and tuple size; epochs: bits, tuple size,
b, rho0 and tau, with the epochs and patience of run_grid.py. The configuration chosen for a model
is the same for every seed.

DWN, as in dwn_higgs2.py: 40 LUTs of 6 inputs and 2 classes (fixed in the kernel), learning rate
--dwnLr (100), --dwnEpochs (1), tau = sqrt(40 / 2) / 0.5, torch.manual_seed(seed) for the initial
LUTs and mapping, --dwnThreads CPU threads (8). Input: --dwnBits bits per feature of --encoding.

Same for every model of a seed: the split of run_grid.py (run_grid.split, stratified 70/15/15) and
the encoder (encoders.py, fitted on the training part). The DWN trains on the training part only
(it has no validation), the epoch training uses the validation part for its early stopping.

Timed with time.perf_counter, with the data already encoded:
  secondsTrain   WiSARD: train / trainEpochs (trainEpochs includes the validation of every epoch,
                 used by its early stopping); DWN: efd_fused_cpu.train
  secondsInfer   classification of the test set. WiSARD: classify, one CPU thread; DWN:
                 efd_fused_cpu.predict, --dwnThreads threads
  usPerSample    secondsInfer / test size, in microseconds
  epochsRun      epoch training: epochs run before early stopping
  secondsEpochsTrain, secondsEpochsVal
                 epoch training: the sum over the epochs of the pass over the training set and
                 of the validation (wall times measured inside trainEpochs)
The epoch training runs the passes over the training set in one thread (each example changes
the model seen by the next one); the validation of every epoch and the addresses computed before
the first epoch use --wisardThreads threads (8, as the DWN). The trained model does not depend
on the number of threads.
Every model runs alone in this process, one after the other, so they do not compete for the CPU;
do not run other heavy jobs during the measurement.

Results: one JSON line per model in <outDir>/higgs/<encoding>/seed<k>.jsonl. Models already there
with the same configuration are skipped, so an interrupted run resumes where it stopped. At the
end (or with --summary) the mean and standard deviation over the seeds are printed.

The DWN needs torch and the kernel, built once in the venv (MSVC on Windows):
  cd experiments/dwn_higgs_cpu
  pip install --no-build-isolation .
Without them the DWN is skipped with a warning.

Usage (from the root of the repository):
  python experiments/measure_time.py --dry-run          # chosen configurations and what would run
  python experiments/measure_time.py                    # 7 models x 10 seeds; resumes if interrupted
  python experiments/measure_time.py --models dwn wisard-DAB --seeds 0
  python experiments/measure_time.py --summary
  python experiments/measure_time.py --models epochs-DAB --epochs 12   # at most 12 epochs
The maximum of epochs is part of the configuration: runs with another --epochs are kept apart in
the same files, and --summary --epochs K prints those with K. As b_t and rho_t depend only on the
epoch, a run with --epochs K is the run with more epochs stopped after its epoch K.
"""
import argparse
import collections
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_grid as rg  # noqa: E402
from encoders import ENCODINGS, Encoder  # noqa: E402

import wisardpkg as wp  # noqa: E402

DATASET = "higgs"
WISARD_MODELS = ["wisard-bleaching", "wisard-DAB", "wisard-rDAB",
                 "epochs-bleaching", "epochs-DAB", "epochs-rDAB"]
MODELS = WISARD_MODELS + ["dwn"]
DWN_LUTS, DWN_N = 40, 6  # fixed in efd_fused_cpu.cpp (kNumLuts, kFanIn), as are its 2 classes
CONFIG_FIELDS = {"oneshot": ("bits", "address"), "epochs": ("bits", "address", "b", "rho0", "tau")}


# ---------------------------------------------------------------- best configurations

def model_name(r):
    """model of a result of run_grid.py"""
    return ("wisard-" if r["kind"] == "oneshot" else "epochs-") + rg.val_mode(r)


def model_kind(model):
    return "oneshot" if model.startswith("wisard-") else "epochs"


def best_configs(grid_dir, encoding, seeds):
    """{model: dict(bits, address[, b, rho0, tau], val, test, candidates)}: the configuration with the
    highest mean validation accuracy over the seeds, among those present in all of them"""
    seeds = sorted(set(seeds))
    found = collections.defaultdict(lambda: collections.defaultdict(dict))  # model -> config -> seed -> (val, test)
    folder = os.path.join(grid_dir, DATASET, encoding)
    for seed in seeds:
        path = os.path.join(folder, "seed%d.jsonl" % seed)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "error" in r:
                    continue
                config = tuple(r[k] for k in CONFIG_FIELDS[r["kind"]])
                found[model_name(r)][config][seed] = (r["bleaching"]["val"]["acc"], r["bleaching"]["test"]["acc"])
    chosen = {}
    for model in WISARD_MODELS:
        complete = sorted(c for c, by_seed in found[model].items() if len(by_seed) == len(seeds))
        if not complete:
            continue

        def mean(config, k):
            return float(np.mean([v[k] for v in found[model][config].values()]))

        best = max(complete, key=lambda c: (mean(c, 0), -c[0], -c[1]))  # ties: fewer bits, smaller tuple
        chosen[model] = dict(zip(CONFIG_FIELDS[model_kind(model)], best),
                             val=mean(best, 0), test=mean(best, 1), candidates=len(complete))
    return chosen


def config_text(t):
    if t["model"] == "dwn":
        return "bits=%d, %d LUTs n=%d, lr=%g, %d ep., %d thr." % (
            t["bits"], t["luts"], t["n"], t["lr"], t["epochs"], t["threads"])
    text = "bits=%d n-tuple=%d" % (t["bits"], t["address"])
    if "rho0" in t and t["rho0"] is not None:
        text += " b=%d rho0=%g tau=%g" % (t["b"], t["rho0"], t["tau"])
        if "epochs" in t:  # the configurations of the grid have no maximum of epochs of their own
            text += " max %d ep." % t["epochs"]
    if "threads" in t:
        text += ", %d thr." % t["threads"]
    return text


def print_configs(configs, args):
    print("best configuration of every WiSARD model: mean validation accuracy over seeds %s of %s/%s/%s"
          % (" ".join(map(str, sorted(set(args.configSeeds)))), args.gridResults, DATASET, args.encoding))
    print("  %-17s %-42s %-9s %-16s %s" % ("model", "configuration", "mean val", "mean test (grid)", "candidates"))
    for model in WISARD_MODELS:
        if model in configs:
            c = configs[model]
            print("  %-17s %-42s %-9.4f %-16.4f %d" % (model, config_text(dict(c, model=model)), c["val"], c["test"],
                                                     c["candidates"]))
        else:
            print("  %-17s no configuration present in all of these seeds" % model)


# ---------------------------------------------------------------- tasks

TASK_FIELDS = {"dataset", "encoding", "seed", "bits", "address", "model", "kind", "valMode", "b", "rho0", "tau",
               "epochs", "patience", "luts", "n", "lr", "threads"}


def make_task(model, seed, args, configs):
    task = dict(dataset=DATASET, encoding=args.encoding, seed=seed, model=model)
    if model == "dwn":
        task.update(kind="dwn-cpu", bits=args.dwnBits, address=DWN_N, luts=DWN_LUTS, n=DWN_N, lr=args.dwnLr,
                    epochs=args.dwnEpochs, threads=args.dwnThreads)
    else:
        c = configs[model]
        kind = model_kind(model)
        task.update(kind=kind, valMode=model.split("-")[1], bits=c["bits"], address=c["address"])
        if kind == "epochs":
            task.update(b=c["b"], rho0=c["rho0"], tau=c["tau"], epochs=args.epochs, patience=args.patience,
                        threads=args.wisardThreads)
    return task


def task_key(t, seed=True):
    """the task without the measurements: a model is done when the same configuration is in the file"""
    fields = TASK_FIELDS if seed else TASK_FIELDS - {"seed"}
    return json.dumps({k: t[k] for k in sorted(t) if k in fields}, sort_keys=True)


def done_keys(out_dir, tasks):
    done = set()
    for path in set(rg.results_path(out_dir, t) for t in tasks):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if "error" not in r:
                        done.add(task_key(r))
    return done


# ---------------------------------------------------------------- measurements

def run_wisard(task, p):
    """p: rg.prepared(...), the encoded split of the seed"""
    model = wp.Wisard(task["address"], bleaching=rg.POLICY[task["valMode"]],
                      mapping=rg.mapping(p["size"], task["address"], task["seed"], p["labels"]), **rg.DAB_OPTIONS)
    result = {}
    start = time.perf_counter()
    if task["kind"] == "oneshot":
        model.train(p["train"])
    else:
        # valMode="bleaching": every epoch is validated with the bleaching policy of the model (run_grid.py)
        report = model.trainEpochs(p["train"], p["val"], b=task["b"], rho0=task["rho0"], tau=task["tau"],
                                   epochs=task["epochs"], patience=task["patience"], valMode="bleaching",
                                   seed=task["seed"], threads=task["threads"])
        rows = report["perEpoch"]
        result.update(epochsRun=len(rows), bestEpoch=report["bestEpoch"],
                      secondsEpochsTrain=sum(r["secondsTrain"] for r in rows),
                      secondsEpochsVal=sum(r["secondsVal"] for r in rows))
    result["secondsTrain"] = time.perf_counter() - start

    start = time.perf_counter()
    predicted = model.classify(p["test"])
    result["secondsInfer"] = time.perf_counter() - start
    result["accTest"] = float(np.mean([str(o) == str(t) for o, t in zip(predicted, p["ytest"])]))
    result.update(size=p["size"], testSize=len(p["ytest"]), host=platform.node(), device="cpu")
    return result


def dwn_data(task, torch):
    """the split of run_grid.py for the seed, binarized by the same encoder: 0/1 uint8 tensors and int64 labels"""
    ds = rg.raw_dataset(DATASET)
    (Xtr, ytr), _, (Xte, yte) = rg.split(ds, task["seed"])
    encoder = Encoder(task["encoding"], task["bits"]).fit(Xtr, ds.nominal)
    index = {label: k for k, label in enumerate(sorted(set(ytr)))}

    def binary(X):  # encoders.py packs 8 bits per byte, most significant first
        return torch.from_numpy(np.ascontiguousarray(np.unpackbits(encoder.transform(X), axis=1)[:, :encoder.size]))

    def targets(y):  # a class absent from training never matches a prediction
        return torch.tensor([index.get(label, -1) for label in y], dtype=torch.long)

    return encoder.size, len(index), binary(Xtr), targets(ytr), binary(Xte), targets(yte)


def run_dwn(task, dwn):
    """dwn_higgs2.py on the split and encoding of the seed"""
    kernel, torch = dwn
    size, classes, x_train, y_train, x_test, y_test = dwn_data(task, torch)
    if classes != 2:
        raise ValueError("efd_fused_cpu only handles 2 classes, higgs has %d" % classes)
    torch.set_num_threads(task["threads"])
    torch.manual_seed(task["seed"])
    mapping_weights = torch.rand(size, task["luts"] * task["n"], dtype=torch.float32).contiguous()
    initial_luts = (torch.rand(task["luts"], 1 << task["n"], dtype=torch.float32) * 2.0 - 1.0).contiguous()
    tau = (1.0 * task["luts"] / classes) ** 0.5 / 0.5

    start = time.perf_counter()
    binary_luts, mapping = kernel.train(x_train, y_train, initial_luts, mapping_weights, float(tau),
                                        float(task["lr"]), int(task["epochs"]))
    seconds_train = time.perf_counter() - start

    start = time.perf_counter()
    predicted = kernel.predict(x_test, binary_luts, mapping)
    seconds_infer = time.perf_counter() - start
    return dict(secondsTrain=seconds_train, secondsInfer=seconds_infer,
                accTest=(predicted == y_test).float().mean().item(), size=size, testSize=len(y_test),
                host=platform.node(), device="cpu", threadsUsed=torch.get_num_threads(), dwnTau=tau)


def measure(task, dwn):
    result = dict(task)
    try:
        if task["model"] == "dwn":
            result.update(run_dwn(task, dwn))
        else:
            result.update(run_wisard(task, rg.prepared(DATASET, task["encoding"], task["seed"], task["bits"])))
        result["usPerSample"] = result["secondsInfer"] / result["testSize"] * 1e6
    except Exception as e:  # recorded and reported; the other models go on
        result["error"] = "%s: %s" % (type(e).__name__, e)
        result["traceback"] = traceback.format_exc()
    return result


def load_dwn():
    """(efd_fused_cpu, torch), or None when the kernel is not built"""
    try:
        import torch
        import efd_fused_cpu
    except ImportError as e:
        print("WARNING: DWN skipped, efd_fused_cpu is not available (%s); build it with\n"
              "  cd experiments/dwn_higgs_cpu && pip install --no-build-isolation ." % e, flush=True)
        return None
    return efd_fused_cpu, torch


# ---------------------------------------------------------------- summary

def summary(out_dir, args, configs):
    expected = collections.OrderedDict()  # configuration (without the seed) -> model
    for model in args.models:
        if model == "dwn" or model in configs:
            task = make_task(model, 0, args, configs)
            expected[task_key(task, seed=False)] = task
    groups = collections.OrderedDict((key, []) for key in expected)
    folder = os.path.join(out_dir, DATASET, args.encoding)
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if name.startswith("seed") and name.endswith(".jsonl"):
            with open(os.path.join(folder, name)) as f:
                for r in map(json.loads, f):
                    key = task_key(r, seed=False)
                    if "error" not in r and key in groups:
                        groups[key].append(r)
    if not any(groups.values()):
        print("no results in %s for these configurations" % folder)
        return

    def cell(values, fmt):
        return (fmt % np.mean(values)) + " +- " + (fmt % np.std(values)) if values else "-"

    print("\n%s, %s (mean +- std over the seeds)" % (DATASET, args.encoding))
    print("%-17s %-60s %-20s %-20s %-20s %-20s %-18s %s" % (
        "model", "configuration", "seeds", "train (s)", "inference test (s)", "inference (us/ex.)", "acc test",
        "epochs run"))
    for key, group in groups.items():
        if not group:
            continue
        epochs = [r["epochsRun"] for r in group if "epochsRun" in r]
        print("%-17s %-60s %-20s %-20s %-20s %-20s %-18s %s" % (
            expected[key]["model"], config_text(expected[key]), ",".join(str(s) for s in sorted(r["seed"] for r in group)),
            cell([r["secondsTrain"] for r in group], "%.3f"), cell([r["secondsInfer"] for r in group], "%.3f"),
            cell([r["usPerSample"] for r in group], "%.2f"), cell([r["accTest"] for r in group], "%.4f"),
            cell(epochs, "%.1f") if epochs else "-"))


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--encoding", default="distributive", choices=ENCODINGS)
    # choice of the WiSARD configurations
    parser.add_argument("--gridResults", default="experiments/results", help="results of run_grid.py")
    parser.add_argument("--configSeeds", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="seeds of the grid whose mean validation accuracy chooses the configurations")
    # epoch training (the epochs and patience of run_grid.py)
    parser.add_argument("--epochs", type=int, default=rg.EPOCHS, help="maximum epochs of trainEpochs")
    parser.add_argument("--patience", type=int, default=rg.PATIENCE, help="early stopping of trainEpochs")
    parser.add_argument("--wisardThreads", type=int, default=8,
                        help="trainEpochs: threads of the validation of each epoch and of the addresses (0: every core)")
    # DWN (dwn_higgs2.py)
    parser.add_argument("--dwnBits", type=int, default=8, help="DWN: bits per feature")
    parser.add_argument("--dwnLr", type=float, default=100.0, help="DWN: learning rate")
    parser.add_argument("--dwnEpochs", type=int, default=1, help="DWN: epochs")
    parser.add_argument("--dwnThreads", type=int, default=8, help="DWN: CPU threads (torch.set_num_threads)")
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--outDir", default="experiments/results_time_best")
    parser.add_argument("--dry-run", action="store_true", help="only list the configurations and what would run")
    parser.add_argument("--summary", action="store_true", help="only print the table of the results already measured")
    args = parser.parse_args()

    configs = best_configs(args.gridResults, args.encoding, args.configSeeds)
    print_configs(configs, args)
    missing = [m for m in args.models if m != "dwn" and m not in configs]
    if missing:
        print("WARNING: without a configuration, not measured: %s" % ", ".join(missing))
    models = [m for m in args.models if m not in missing]
    if args.summary:
        summary(args.outDir, args, configs)
        return

    tasks = [make_task(model, seed, args, configs) for seed in args.seeds for model in models]
    done = done_keys(args.outDir, tasks)
    pending = [t for t in tasks if task_key(t) not in done]
    print("\n%s %s: %d models x %d seeds, %d already measured, %d to run" % (
        DATASET, args.encoding, len(models), len(args.seeds), len(tasks) - len(pending), len(pending)))
    for t in pending:
        print("  seed %d  %-17s %s" % (t["seed"], t["model"], config_text(t)))
    if args.dry_run or not pending:
        if not pending:
            summary(args.outDir, args, configs)
        return

    dwn = load_dwn() if any(t["model"] == "dwn" for t in pending) else None
    if dwn is None:
        pending = [t for t in pending if t["model"] != "dwn"]
    rg.init_worker(dict(dataDir=args.dataDir))

    start, errors = time.time(), 0
    for n, task in enumerate(pending, 1):
        result = measure(task, dwn)
        path = rg.results_path(args.outDir, result)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
        head = "[%d/%d %.0fs] seed %d %-17s" % (n, len(pending), time.time() - start, task["seed"], task["model"])
        if "error" in result:
            errors += 1
            print("%s ERROR %s\n%s" % (head, result["error"], result["traceback"]), flush=True)
        else:
            extra = "  epochs %d" % result["epochsRun"] if "epochsRun" in result else ""
            print("%s train %8.3fs  infer %7.3fs (%.2f us/ex.)  acc test %.4f%s" % (
                head, result["secondsTrain"], result["secondsInfer"], result["usPerSample"], result["accTest"],
                extra), flush=True)
    print("finished: %d measured, %d errors, %.1f min" % (len(pending), errors, (time.time() - start) / 60))
    summary(args.outDir, args, configs)


if __name__ == "__main__":
    main()
