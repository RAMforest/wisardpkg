"""
Training and inference time of the models on higgs, every model with the same binary input and
the same tuple size:

  wisard-bleaching / wisard-DAB / wisard-rDAB   one-shot WiSARD (Wisard.train), classified with
                                                iterative bleaching, DAB or rDAB
  epochs-bleaching / epochs-DAB / epochs-rDAB   WiSARD trained by epochs (Wisard.trainEpochs); the
                                                validation of every epoch and the classification
                                                use that bleaching, as in run_grid.py
  dwn                                           DWN (torch-dwn), the model of run_dwn.py

Same for every model of a seed: the split (run_grid.split), the encoding (encoders.py, fitted on
the training part), the bits per feature and the tuple size: --address is the address size of the
WiSARD RAMs and the number of inputs n of every DWN LUT. The configuration of the epoch training,
of DAB/rDAB and of the DWN is fixed (options below); it is not what is measured.

Timed with time.perf_counter, with the data already encoded:
  secondsTrain   WiSARD: train / trainEpochs (trainEpochs includes the validation of every epoch,
                 used by its early stopping); DWN: its epochs, without the validation of each epoch
                 (run_dwn.py measures it this way)
  secondsInfer   classification of the test set. WiSARD: classify, one CPU thread. DWN: forward
                 on the GPU in batches of --evalBatch, including the copy of the batches to the GPU
  usPerSample    secondsInfer / test size, in microseconds
  epochsRun      epoch training: epochs run before early stopping (--patience >= --epochs runs them
                 all, the same number for the three bleachings)
The training of the three one-shot models is the same (the bleaching only changes classify).
Every model runs alone in this process, one after the other, so they do not compete for the CPU;
do not run other heavy jobs during the measurement.

Results: one JSON line per model in <outDir>/higgs/<encoding>/seed<k>.jsonl. Models already there
with the same configuration are skipped, so an interrupted run resumes where it stopped. At the
end (or with --summary) the mean and standard deviation over the seeds are printed.

Needs wisardpkg and, for the DWN, torch-dwn on an NVIDIA GPU (run_dwn.py); without torch-dwn the
DWN is skipped with a warning. The DWN never falls back to the CPU: with --device cuda (the
default) and no CUDA the script stops. Every DWN result records where the model parameters were
(device, with the name of the GPU) and peakGpuMB. --check-device prints the GPUs torch sees and
trains one batch of the DWN that is measured (the higgs input of the first seed, --luts,
n = --address) on the device, then exits (run_dwn.check_device).

Usage (from the root of the repository):
  python experiments/measure_time.py --check-device                # does the DWN run on the GPU?
  python experiments/measure_time.py --dry-run
  python experiments/measure_time.py --seeds 0 1 2 3 4
  python experiments/measure_time.py --models wisard-DAB epochs-DAB dwn --seeds 0
  python experiments/measure_time.py --summary
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
from encoders import ENCODINGS  # noqa: E402

import wisardpkg as wp  # noqa: E402

DATASET = "higgs"
WISARD_MODELS = ["wisard-bleaching", "wisard-DAB", "wisard-rDAB",
                 "epochs-bleaching", "epochs-DAB", "epochs-rDAB"]
MODELS = WISARD_MODELS + ["dwn"]


# ---------------------------------------------------------------- tasks

def make_task(model, seed, args):
    task = dict(dataset=DATASET, encoding=args.encoding, seed=seed, bits=args.bits, address=args.address, model=model)
    if model == "dwn":
        task.update(kind="dwn", luts=args.luts, n=args.address, epochs=args.dwnEpochs, batchSize=args.batchSize,
                    lr=args.lr, groupSumTau=args.groupSumTau)
    else:
        kind, mode = model.split("-")
        task.update(kind="oneshot" if kind == "wisard" else "epochs", valMode=mode)
        if kind == "epochs":
            task.update(b=args.b, rho0=args.rho0, tau=args.tau, epochs=args.epochs, patience=args.patience)
    return task


def task_key(t):
    """the task without the measurements: a model is done when the same configuration is in the file"""
    return json.dumps({k: t[k] for k in sorted(t) if k in TASK_FIELDS}, sort_keys=True)


TASK_FIELDS = {"dataset", "encoding", "seed", "bits", "address", "model", "kind", "valMode", "b", "rho0", "tau",
               "epochs", "patience", "luts", "n", "batchSize", "lr", "groupSumTau"}


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
                                   seed=task["seed"])
        result.update(epochsRun=len(report["perEpoch"]), bestEpoch=report["bestEpoch"])
    result["secondsTrain"] = time.perf_counter() - start

    start = time.perf_counter()
    predicted = model.classify(p["test"])
    result["secondsInfer"] = time.perf_counter() - start
    result["accTest"] = float(np.mean([str(o) == str(t) for o, t in zip(predicted, p["ytest"])]))
    result.update(size=p["size"], testSize=len(p["ytest"]), host=platform.node(), device="cpu")
    return result


def run_dwn(task, args, dwn_module, device):
    """run_dwn.run_task: the same split and encoder; the epochs are timed without the validation of each epoch"""
    dwn_args = argparse.Namespace(dataDir=args.dataDir, stepSize=args.stepSize, gamma=args.gamma,
                                  evalBatch=args.evalBatch)
    out = dwn_module.run_task(dict(task, luts=list(task["luts"])), dwn_args, device)
    if "error" in out:
        raise RuntimeError("%s\n%s" % (out["error"], out.get("traceback", "")))
    test_size = len(rg.split(rg.raw_dataset(DATASET), task["seed"])[2][1])  # the same split as run_dwn.prepare
    # device: where the model parameters were (next(model.parameters()).device), with the name of the GPU
    return dict(secondsTrain=out["secondsTrain"], secondsInfer=out["secondsInferTest"], accTest=out["acc"]["test"],
                lutsUsed=out["lutsUsed"], size=out["size"], testSize=test_size, host=platform.node(),
                device=out["device"] + (" " + out["deviceName"] if "deviceName" in out else ""),
                peakGpuMB=out.get("peakGpuMB"))


def measure(task, args, dwn_module, device):
    result = dict(task)
    try:
        if task["model"] == "dwn":
            result.update(run_dwn(task, args, dwn_module, device))
        else:
            result.update(run_wisard(task, rg.prepared(DATASET, task["encoding"], task["seed"], task["bits"])))
        result["usPerSample"] = result["secondsInfer"] / result["testSize"] * 1e6
    except Exception as e:  # recorded and reported; the other models go on
        result["error"] = "%s: %s" % (type(e).__name__, e)
        result["traceback"] = traceback.format_exc()
    return result


def load_dwn(models, device_name):
    """run_dwn (torch, torch-dwn) and the device, or None when the DWN is not asked or torch-dwn is
    missing; stops when CUDA is asked and not available (never falls back to the CPU)"""
    if "dwn" not in models:
        return None, None
    try:
        import run_dwn
    except ImportError as e:
        print("WARNING: DWN skipped, torch-dwn is not available (%s)" % e, flush=True)
        return None, None
    torch = run_dwn.torch
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: CUDA is not available to torch (torch %s, built with CUDA %s); run --check-device, "
                 "or pass --device cpu explicitly" % (torch.__version__, torch.version.cuda))
    if device.type == "cuda":
        print("DWN device: %s (%s)" % (device, torch.cuda.get_device_name(device)), flush=True)
    else:
        print("WARNING: DWN on %s; the LUT layers of torch-dwn are meant for NVIDIA GPUs" % device, flush=True)
    return run_dwn, device


def check_device(args):
    """run_dwn.check_device with the DWN that is measured: the higgs input of the first seed"""
    try:
        import run_dwn
    except ImportError as e:
        print("RESULT: torch-dwn is not available (%s)" % e)
        return
    rg.init_worker(dict(dataDir=args.dataDir))
    p = rg.prepared(DATASET, args.encoding, args.seeds[0], args.bits)
    check_args = argparse.Namespace(luts=args.luts, n=args.address, groupSumTau=args.groupSumTau, lr=args.lr,
                                    batchSize=args.batchSize, checkSize=p["size"], checkClasses=len(p["labels"]))
    run_dwn.check_device(check_args, run_dwn.torch.device(args.device))


# ---------------------------------------------------------------- summary

def summary(out_dir, args):
    folder = os.path.join(out_dir, DATASET, args.encoding)
    rows = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if name.startswith("seed") and name.endswith(".jsonl"):
            with open(os.path.join(folder, name)) as f:
                rows += [r for r in map(json.loads, f) if "error" not in r
                         and r["bits"] == args.bits and r["address"] == args.address]
    if not rows:
        print("no results in %s for bits=%d address=%d" % (folder, args.bits, args.address))
        return
    groups = collections.OrderedDict((m, []) for m in MODELS)
    for r in rows:
        groups.setdefault(r["model"], []).append(r)

    def cell(values, fmt):
        return (fmt % np.mean(values)) + " +- " + (fmt % np.std(values)) if values else "-"

    print("\n%s, %s, %d bits per feature, tuple size %d (mean +- std over the seeds)" % (
        DATASET, args.encoding, args.bits, args.address))
    print("%-17s %-10s %-22s %-22s %-20s %-18s %s" % (
        "model", "seeds", "train (s)", "inference test (s)", "inference (us/ex.)", "acc test", "epochs run"))
    for model, group in groups.items():
        if not group:
            continue
        epochs = [r["epochsRun"] for r in group if "epochsRun" in r]
        print("%-17s %-10s %-22s %-22s %-20s %-18s %s" % (
            model, ",".join(str(s) for s in sorted(r["seed"] for r in group)),
            cell([r["secondsTrain"] for r in group], "%.3f"), cell([r["secondsInfer"] for r in group], "%.3f"),
            cell([r["usPerSample"] for r in group], "%.2f"), cell([r["accTest"] for r in group], "%.4f"),
            cell(epochs, "%.1f") if epochs else "-"))
    devices = sorted(set(r.get("device", "not recorded") for r in groups.get("dwn", [])))
    if devices:
        print("DWN device: %s" % "; ".join(devices))
    configs = set(task_key({k: v for k, v in r.items() if k != "seed"}) for r in rows)
    if len(configs) > sum(1 for group in groups.values() if group):
        print("NOTE: the file has more than one configuration per model (other epoch/DWN options); they are pooled")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--encoding", default="distributive", choices=ENCODINGS)
    parser.add_argument("--bits", type=int, default=8, help="bits per feature, the same for every model")
    parser.add_argument("--address", type=int, default=6,
                        help="tuple size: address of the WiSARD RAMs and inputs n of the DWN LUTs")
    # epoch training (one fixed configuration of the grid of run_grid.py)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--rho0", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=rg.EPOCHS, help="maximum epochs of trainEpochs")
    parser.add_argument("--patience", type=int, default=rg.PATIENCE,
                        help="early stopping of trainEpochs; >= --epochs runs every epoch")
    # DWN (the defaults of run_dwn.py for tabular data)
    parser.add_argument("--luts", type=int, nargs="+", default=[1000], help="DWN: LUTs per layer")
    parser.add_argument("--dwnEpochs", type=int, default=30)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--stepSize", type=int, default=14, help="DWN: StepLR step size (epochs)")
    parser.add_argument("--gamma", type=float, default=0.1, help="DWN: StepLR factor")
    parser.add_argument("--groupSumTau", type=float, default=1 / 0.3)
    parser.add_argument("--evalBatch", type=int, default=4096, help="DWN: batch of the inference")
    parser.add_argument("--device", default="cuda",
                        help="DWN: cuda (default; stops if CUDA is not available), cuda:1 for a second NVIDIA GPU, or cpu")
    parser.add_argument("--check-device", action="store_true",
                        help="only check that the DWN runs on the device (one training batch), then exit")
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--outDir", default="experiments/results_time")
    parser.add_argument("--dry-run", action="store_true", help="only list what would run")
    parser.add_argument("--summary", action="store_true", help="only print the table of the results already measured")
    args = parser.parse_args()

    if args.check_device:
        check_device(args)
        return
    if args.summary:
        summary(args.outDir, args)
        return

    tasks = [make_task(model, seed, args) for seed in args.seeds for model in args.models]
    done = done_keys(args.outDir, tasks)
    pending = [t for t in tasks if task_key(t) not in done]
    print("%s %s, %d bits, tuple size %d: %d models x %d seeds, %d already measured, %d to run" % (
        DATASET, args.encoding, args.bits, args.address, len(args.models), len(args.seeds),
        len(tasks) - len(pending), len(pending)))
    for t in pending:
        print("  seed %d  %s" % (t["seed"], t["model"]))
    if args.dry_run or not pending:
        if not pending:
            summary(args.outDir, args)
        return

    dwn_module, device = load_dwn([t["model"] for t in pending], args.device)
    if dwn_module is None:
        pending = [t for t in pending if t["model"] != "dwn"]
    rg.init_worker(dict(dataDir=args.dataDir))

    start, errors = time.time(), 0
    for n, task in enumerate(pending, 1):
        result = measure(task, args, dwn_module, device)
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
            if task["model"] == "dwn":
                extra += "  on %s" % result["device"]
                if result.get("peakGpuMB") is not None:
                    extra += ", %.0f MB GPU" % result["peakGpuMB"]
            print("%s train %8.3fs  infer %7.3fs (%.2f us/ex.)  acc test %.4f%s" % (
                head, result["secondsTrain"], result["secondsInfer"], result["usPerSample"], result["accTest"],
                extra), flush=True)
    print("finished: %d measured, %d errors, %.1f min" % (len(pending), errors, (time.time() - start) / 60))
    summary(args.outDir, args)


if __name__ == "__main__":
    main()
