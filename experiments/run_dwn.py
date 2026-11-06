"""
DWN benchmark (Differentiable Weightless Neural Network, torch-dwn) on the datasets, encodings
and seeds of the WiSARD benchmark (run_grid.py), with the same splits and the same binary inputs:

  split        run_grid.split(dataset, seed), the seeds of the WiSARD results
  binarization encoders.py (fitted on the training part), the bits the WiSARD ran for the dataset
  model        fixed: the example of the torch-dwn README (only the bits change)
                 images:  LUTLayer(input, 2000, n=6, learnable mapping) -> LUTLayer(2000, 1000, n=6)
                 tabular: LUTLayer(input, 1000, n=6, learnable mapping)
                 -> GroupSum(classes, tau=1/0.3); the last layer rounded up to a multiple of the classes
               Adam lr 1e-2, StepLR(14, 0.1), batch 32, 30 epochs; the model of the last epoch is reported

Every configuration is written as one JSON line in <resultsDir>/<dataset>/<encoding>/seed<k>.jsonl:
  acc            val / test accuracy of the final model
  secondsTrain   training (the epochs, without the validation of every epoch)
  secondsInferVal, secondsInferTest, secondsInfer (= val + test): classification of the final model
  accValPerEpoch validation accuracy after every epoch (not timed)
  seconds        wall time of the whole configuration (encoding, training, validations, inference)
Configurations already in the files are skipped, so an interrupted run resumes where it stopped.

torch-dwn only runs on NVIDIA GPUs (its LUT layers have only a CUDA kernel), and is compiled at
install time, against the installed torch:
  pip install torch --index-url https://download.pytorch.org/whl/cu124   # a CUDA build of torch
  pip install torch-dwn --no-build-isolation
It needs the CUDA toolkit (nvcc) with CUDA_HOME set and, on Windows, the MSVC Build Tools.
wisardpkg is not needed.

Usage:
  python experiments/run_dwn.py --dry-run                       # what would run
  python experiments/run_dwn.py --datasets iris blood --seeds 1
  python experiments/run_dwn.py                                 # everything; resumes if interrupted
"""
import argparse
import collections
import json
import os
import sys
import time
import traceback
import types

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import wisardpkg  # noqa: F401  encoders.py and run_grid.py import it; the DWN does not use it
except ImportError:
    sys.modules["wisardpkg"] = types.ModuleType("wisardpkg")
import datasets  # noqa: E402
import run_grid as rg  # noqa: E402
from encoders import ENCODINGS, Encoder  # noqa: E402

import torch_dwn as dwn  # noqa: E402

IMAGE_LUTS = [2000, 1000]
TABULAR_LUTS = [1000]


# ---------------------------------------------------------------- data and model

_raw = {}


def raw_dataset(name, data_dir):
    if name not in _raw:
        _raw.clear()  # tasks come ordered by dataset; keep only one in memory
        _raw[name] = datasets.load(name, data_dir)
    return _raw[name]


def prepare(name, encoding, seed, bits, data_dir):
    """the split of run_grid.py for this seed, binarized by the same encoder: bits as uint8 tensors on the CPU"""
    ds = raw_dataset(name, data_dir)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = rg.split(ds, seed)
    encoder = Encoder(encoding, bits).fit(Xtr, ds.nominal)
    index = {label: k for k, label in enumerate(sorted(set(ytr)))}

    def binary(X):  # encoders.py packs 8 bits per byte, most significant first
        return torch.from_numpy(np.unpackbits(encoder.transform(X), axis=1)[:, :encoder.size])

    def targets(y):  # a class absent from training never matches a prediction
        return torch.tensor([index.get(label, -1) for label in y])

    return dict(size=encoder.size, classes=len(index),
                train=(binary(Xtr), targets(ytr)), val=(binary(Xva), targets(yva)), test=(binary(Xte), targets(yte)))


def build_model(size, classes, luts, n, tau):
    luts = list(luts)
    luts[-1] = -(-luts[-1] // classes) * classes  # GroupSum splits the last layer evenly among the classes
    layers, inputs = [], size
    for k, outputs in enumerate(luts):
        layers.append(dwn.LUTLayer(inputs, outputs, n=n, mapping="learnable" if k == 0 else "random"))
        inputs = outputs
    return nn.Sequential(*layers, dwn.GroupSum(classes, tau=tau)), luts


@torch.no_grad()
def accuracy(model, data, device, batch_size):
    X, y = data
    model.eval()
    hits = 0
    for start in range(0, len(X), batch_size):
        predicted = model(X[start:start + batch_size].to(device).float()).argmax(dim=1).cpu()  # .cpu() waits for the GPU
        hits += (predicted == y[start:start + batch_size]).sum().item()
    return hits / len(X)


def timed_accuracy(model, data, device, batch_size):
    start = time.time()
    acc = accuracy(model, data, device, batch_size)
    return acc, time.time() - start


# ---------------------------------------------------------------- one configuration

def run_task(task, args, device):
    start = time.time()
    result = dict(task)
    try:
        p = prepare(task["dataset"], task["encoding"], task["seed"], task["bits"], args.dataDir)
        torch.manual_seed(task["seed"])
        model, luts = build_model(p["size"], p["classes"], task["luts"], task["n"], task["groupSumTau"])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=task["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepSize, gamma=args.gamma)
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(task["seed"])
        X, y = p["train"]
        acc_val, seconds_train = [], 0.0
        for epoch in range(task["epochs"]):
            epoch_start = time.time()
            model.train()
            order = torch.randperm(len(X), generator=generator)
            for s in range(0, len(X), task["batchSize"]):
                batch = order[s:s + task["batchSize"]]
                optimizer.zero_grad()
                loss_fn(model(X[batch].to(device).float()), y[batch].to(device)).backward()
                optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds_train += time.time() - epoch_start
            acc_val.append(accuracy(model, p["val"], device, args.evalBatch))
        val, seconds_val = timed_accuracy(model, p["val"], device, args.evalBatch)
        test, seconds_test = timed_accuracy(model, p["test"], device, args.evalBatch)
        result.update(size=p["size"], classes=p["classes"], lutsUsed=luts, acc=dict(val=val, test=test),
                      accValPerEpoch=acc_val, secondsTrain=seconds_train, secondsInferVal=seconds_val,
                      secondsInferTest=seconds_test, secondsInfer=seconds_val + seconds_test)
    except Exception as e:  # recorded and reported (e.g. out of GPU memory); the rest goes on
        result["error"] = "%s: %s" % (type(e).__name__, e)
        result["traceback"] = traceback.format_exc()
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["seconds"] = time.time() - start
    return result


# ---------------------------------------------------------------- task list

def task_key(t):
    return "dwn|bits=%(bits)d|luts=%(luts)s|n=%(n)d|epochs=%(epochs)d|batch=%(batchSize)d|lr=%(lr)s" % t


def wisard_runs(results_dir):
    """{(dataset, encoding): {seed: set of bits}} of the WiSARD results (run_grid.py)"""
    runs = collections.defaultdict(lambda: collections.defaultdict(set))
    for dataset in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        for encoding in sorted(os.listdir(os.path.join(results_dir, dataset))):
            folder = os.path.join(results_dir, dataset, encoding)
            for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
                if name.startswith("seed") and name.endswith(".jsonl"):
                    with open(os.path.join(folder, name)) as f:
                        for line in f:
                            r = json.loads(line)
                            if "error" not in r:
                                runs[(dataset, encoding)][int(name[4:-6])].add(r["bits"])
    return runs


def build_tasks(args):
    runs = wisard_runs(args.wisardResults)
    tasks = []
    for name in args.datasets:
        image = name in datasets.IMAGES
        config = dict(luts=args.luts or (IMAGE_LUTS if image else TABULAR_LUTS), n=args.n, epochs=args.epochs,
                      batchSize=args.batchSize, lr=args.lr, groupSumTau=args.groupSumTau)
        for encoding in args.encodings:
            seeds = runs.get((name, encoding), {})
            for seed in (args.seeds if args.seeds is not None else sorted(seeds)):
                bits_grid = args.bits or sorted(seeds.get(seed, ())) or rg.GRID["image" if image else "tabular"]["bits"]
                for bits in bits_grid:
                    tasks.append(dict(dataset=name, encoding=encoding, seed=seed, bits=bits, kind="dwn", **config))
    return tasks


def done_keys(results_dir, tasks):
    done = set()
    for path in set(rg.results_path(results_dir, t) for t in tasks):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if "error" not in r:
                        done.add((path, task_key(r)))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=[n for n in datasets.NAMES if n != "cifar10"], choices=datasets.NAMES)
    parser.add_argument("--encodings", nargs="+", default=list(ENCODINGS), choices=ENCODINGS)
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the seeds of the WiSARD results of each dataset/encoding")
    parser.add_argument("--bits", type=int, nargs="+", help="default: the bits the WiSARD ran (or the bits grid of run_grid.py)")
    parser.add_argument("--luts", type=int, nargs="+", help="LUTs per layer (default: 2000 1000 for images, 1000 for tabular)")
    parser.add_argument("--n", type=int, default=6, help="inputs per LUT")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--stepSize", type=int, default=14, help="StepLR step size (epochs)")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR factor")
    parser.add_argument("--groupSumTau", type=float, default=1 / 0.3)
    parser.add_argument("--evalBatch", type=int, default=4096)
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--wisardResults", default="experiments/results", help="results of run_grid.py (seeds and bits)")
    parser.add_argument("--resultsDir", default="experiments/results_dwn")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="only list what would run")
    args = parser.parse_args()

    tasks = build_tasks(args)
    done = done_keys(args.resultsDir, tasks)
    pending = [t for t in tasks if (rg.results_path(args.resultsDir, t), task_key(t)) not in done]
    counts = collections.Counter((t["dataset"], t["encoding"]) for t in pending)
    for (dataset, encoding), n in sorted(counts.items()):
        seeds = sorted({t["seed"] for t in pending if (t["dataset"], t["encoding"]) == (dataset, encoding)})
        print("  %-14s %-12s %3d configurations, seeds %s" % (dataset, encoding, n, seeds))
    print("%d configurations, %d already done, %d to run on %s" % (len(tasks), len(tasks) - len(pending), len(pending), args.device))
    if args.dry_run or not pending:
        return
    device = torch.device(args.device)
    if device.type != "cuda":
        print("WARNING: no CUDA; the LUT layers of torch-dwn are not implemented on the CPU", flush=True)

    start, errors = time.time(), 0
    for n, task in enumerate(pending, 1):
        result = run_task(task, args, device)
        path = rg.results_path(args.resultsDir, result)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
        elapsed = time.time() - start
        head = "[%d/%d %.0fs, eta %.1fh] %s %s seed%d %s" % (
            n, len(pending), elapsed, elapsed / n * (len(pending) - n) / 3600,
            task["dataset"], task["encoding"], task["seed"], task_key(task))
        if "error" in result:
            errors += 1
            print("%s  ERROR %s\n%s" % (head, result["error"], result["traceback"]), flush=True)
        else:
            print("%s  val %.4f  test %.4f  train %.0fs  infer %.1fs" % (
                head, result["acc"]["val"], result["acc"]["test"], result["secondsTrain"], result["secondsInfer"]), flush=True)
    print("finished: %d results, %d errors, %.1f h" % (len(pending), errors, (time.time() - start) / 3600))


if __name__ == "__main__":
    main()
