"""
Multi-bleaching benchmark: every bleaching base with and without the multi-bleaching layer, on
the same trained network (RFC-001).

A network is one (dataset, encoding, seed, bits, address). It is trained once (one-shot)
and classified, on validation and test, with every rule:

  base                 iterative, bbleaching, dab                      (Wisard(bleaching=...))
  base + multi@C_th    the same base with multiBleaching=True and confidenceThreshold=C_th, for
                       every C_th of --thresholds (fixed values, not chosen on validation)

so the rules differ only in the decision and can be paired sample by sample. Wisard.setBleaching
changes the rule of the trained model.

Splits and the "random" mapping are the ones of run_grid.py. The binarization is the one of
encoders.Encoder with binary_as_bit=True: the thermometer only on numeric features with three or
more values, one-hot only on nominal features with three or more categories, ONE bit for a
feature with two values (nominal or numeric) and no bit for a constant one. On datasets without
binary or constant features this is the encoding of run_grid.py, and the base rules with the
random mapping reproduce its one-shot results. The mapping is always "random": a seeded
permutation of the bits, the same for every class (run_grid.mapping), with complete addressing:
the last RAM is filled with random bits. Every network keeps mapping="random" in its JSON line and
in the name of its files.

Every network is one JSON line in <resultsDir>/<dataset>/<encoding>/seed<k>.jsonl; networks
already in the files are skipped, so an interrupted run resumes. Per rule and part: accuracy,
balanced accuracy, macro F1, recall per class, mean confidence, vote counts (steps), deciding
threshold; for the multi rules also the number of thresholds traced per example (one per round: mean,
median and max), the fraction of examples with more than one and, against the base, the examples only one of the two gets right.

Every network also leaves, under <resultsDir>, with the name b<bits>_a<address>_<mapping>:
  checkpoints/<dataset>/<encoding>/seed<k>/<network>.json.gz   the trained model (Wisard.json(),
      gzipped: configuration, mapping and counters), written right after the training;
      load_checkpoint(path) gives the Wisard back (--no-checkpoints skips them)
  logs/<dataset>/<encoding>/seed<k>/<network>.log              what was done and when: sizes,
      training time, one line per rule and part with its scores and time, the checkpoint, the
      traceback when the network fails, and, for every base and part, one line per ROUND of the
      multi-bleaching with the largest C_th: how many examples reached the round, the threshold
      found (mean, median, max) and the confidence at it (also in "perRound" of the JSON line)
  traces/<dataset>/<encoding>/seed<k>/<network>.npz            per example: the prediction of
      every rule ("<part>|<rule>") and, for the largest C_th of every base, one entry per round
      ("<part>|<base>|example", "|round", "|threshold", "|confidence"): the threshold found in
      that round and the confidence at it. The rounds of a smaller C_th are the first ones, up to
      the first confidence above it. --no-traces skips this file (it is the large one).

Usage:
  python experiments/run_multi_bleaching.py --dry-run
  python experiments/run_multi_bleaching.py --datasets iris blood australian --jobs 6
  python experiments/run_multi_bleaching.py --datasets jsc covertype --jobs 2 --no-traces
"""
import argparse
import collections
import gzip
import json
import multiprocessing
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import run_grid as rg  # noqa: E402
from encoders import ENCODINGS, Encoder, to_dataset  # noqa: E402

import wisardpkg as wp  # noqa: E402

BASES = ["iterative", "bbleaching", "dab"]
THRESHOLDS = [0.05, 0.1, 0.2, 0.5, 1.0]
MAPPING = "random"  # the only mapping; kept in the results and in the file names
# from the lightest to the heaviest; cifar10 has no grid results and stays out
DATASETS = ["iris", "blood", "australian", "wine-red", "segment", "jasmine", "wine-white", "sylvine", "phoneme",
            "christine", "letter", "nomao", "higgs", "miniboone", "skin-seg", "mnist", "fashion-mnist", "covertype", "jsc"]


def rule_name(base, threshold=None):
    return base if threshold is None else "%s+multi@%g" % (base, threshold)


_prepared = collections.OrderedDict()


def prepared(name, encoding, seed, bits):
    """run_grid.prepared with the binary features as one bit; also the kinds of the features"""
    key = (name, encoding, seed, bits)
    if key not in _prepared:
        ds = rg.raw_dataset(name)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = rg.split(ds, seed)
        encoder = Encoder(encoding, bits, binary_as_bit=True).fit(Xtr, ds.nominal)
        _prepared[key] = dict(
            size=encoder.size, labels=sorted(set(ytr)),
            features=dict(thermometer=len(encoder.numeric), binary=len(encoder.binary), onehot=len(encoder.nominal),
                          onehotBits=sum(len(c) for c in encoder.categories),
                          dropped=Xtr.shape[1] - len(encoder.numeric) - len(encoder.binary) - len(encoder.nominal)),
            train=to_dataset(encoder.transform(Xtr), ytr, encoder.size),
            val=to_dataset(encoder.transform(Xva), yva, encoder.size), yval=list(yva),
            test=to_dataset(encoder.transform(Xte), yte, encoder.size), ytest=list(yte))
        while len(_prepared) > 2:
            _prepared.popitem(last=False)
    _prepared.move_to_end(key)
    return _prepared[key]


def scores(predicted, expected, n_classes):
    """accuracy, balanced accuracy (mean recall) and macro F1 over the classes present in expected"""
    cm = np.bincount(expected * n_classes + predicted, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(float)
    tp, support, called = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    present = support > 0
    recall = tp[present] / support[present]
    f1 = 2 * tp[present] / (support[present] + called[present])
    return dict(acc=float(tp.sum() / cm.sum()), bal_acc=float(recall.mean()), macro_f1=float(f1.mean()),
                recall=[round(float(r), 6) if ok else None for r, ok in zip(tp / np.maximum(support, 1), present)])


def classify(model, data, index):
    """prediction (class index), confidence of the decision, vote counts, deciding threshold,
    inference latency (microseconds) and the raw output of every example"""
    out = model.classify(data)
    predicted = np.array([index[o["class"]] for o in out], dtype=np.int64)
    confidence = np.array([rg.confidence([c["degree"] for c in o["classesDegrees"]]) for o in out])
    steps = np.array([o["steps"] for o in out], dtype=np.int64)
    threshold = np.array([o["finalThreshold"] for o in out], dtype=np.int64)
    latency = np.array([o["latencyNs"] for o in out], dtype=np.float64) / 1e3
    return predicted, confidence, steps, threshold, latency, out


def load_checkpoint(path, **options):
    """the Wisard of a checkpoint; options (returnBleachingInfo, bleaching, ...) replace the saved ones"""
    with gzip.open(path, "rt") as f:
        config = json.load(f)
    return wp.Wisard(json.dumps(dict(config, **options)))


class TaskLog(object):
    """log of one network, written line by line so a network that dies leaves what it did"""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file, self.start = open(path, "w"), time.time()

    def __call__(self, message, *args):
        self.file.write("%s +%7.1fs  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), time.time() - self.start, message % args))
        self.file.flush()

    def close(self):
        self.file.close()


def run_task(task):
    start = time.time()
    result = dict(task)
    log = TaskLog(artifact_path(task["resultsDir"], "logs", task, ".log"))
    log("network %s %s seed%d %s  thresholds %s  tieTolerance %s  nextCut %s  wisardpkg %s", task["dataset"], task["encoding"],
        task["seed"], task_key(task), task["thresholds"], task["tieTolerance"], task["nextCut"], getattr(wp, "__version__", "?"))
    try:
        p = prepared(task["dataset"], task["encoding"], task["seed"], task["bits"])
        f = p["features"]
        log("data ready: %d input bits (%d numeric features x %d thermometer bits, %d binary features as 1 bit, %d nominal features "
            "as %d one-hot bits, %d constant features dropped), train %d, val %d, test %d, %d classes", p["size"], f["thermometer"],
            task["bits"], f["binary"], f["onehot"], f["onehotBits"], f["dropped"], p["train"].size(), len(p["yval"]),
            len(p["ytest"]), len(p["labels"]))
        labels = sorted(set(p["labels"]) | set(p["yval"]) | set(p["ytest"]))
        index = {label: i for i, label in enumerate(labels)}
        model = wp.Wisard(task["address"], returnClassesDegrees=True, returnBleachingInfo=True, returnLatency=True,
                          mapping=rg.mapping(p["size"], task["address"], task["seed"], p["labels"]),
                          tieTolerance=task["tieTolerance"], nextCut=task["nextCut"], **rg.DAB_OPTIONS)
        t0 = time.time()
        model.train(p["train"])
        result["trainSeconds"] = time.time() - t0
        log("trained one-shot in %.2fs: %d RAMs per class", result["trainSeconds"], -(-p["size"] // task["address"]))
        if task["checkpoints"]:
            path = artifact_path(task["resultsDir"], "checkpoints", task, ".json.gz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with gzip.open(path + ".part", "wt", compresslevel=4) as f:
                f.write(model.json())
            os.replace(path + ".part", path)
            result["checkpoint"] = os.path.relpath(path, task["resultsDir"])
            log("checkpoint %s (%.1f kB)", result["checkpoint"], os.path.getsize(path) / 1e3)
        rules = collections.OrderedDict()
        traces = {}
        for part, data, y in (("val", p["val"], p["yval"]), ("test", p["test"], p["ytest"])):
            expected = np.array([index[label] for label in y], dtype=np.int64)
            for base in BASES:
                model.setBleaching(bleaching=rg.POLICY.get(base, base), multiBleaching=False)
                t0 = time.time()
                b_pred, conf, steps, threshold, latency, _ = classify(model, data, index)
                row = rules.setdefault(rule_name(base), {})[part] = dict(
                    scores(b_pred, expected, len(labels)), conf=float(conf.mean()), steps=float(steps.mean()),
                    threshold=float(threshold.mean()), thresholdMedian=float(np.median(threshold)), thresholdMax=int(threshold.max()),
                    latencyUs=float(latency.mean()), latencyStdUs=float(latency.std()), seconds=time.time() - t0)
                log("%-4s %-22s acc %.4f  balAcc %.4f  F1 %.4f  conf %.3f  steps %.2f  threshold %.2f  latency %.2f +- %.2f us  (%.2fs)",
                    part, rule_name(base), row["acc"], row["bal_acc"], row["macro_f1"], row["conf"], row["steps"], row["threshold"],
                    row["latencyUs"], row["latencyStdUs"], row["seconds"])
                traces["%s|%s" % (part, rule_name(base))] = b_pred.astype(np.int16)
                for c_th in task["thresholds"]:
                    model.setBleaching(multiBleaching=True, confidenceThreshold=c_th)
                    t0 = time.time()
                    m_pred, conf, steps, threshold, latency, out = classify(model, data, index)
                    rounds = np.array([len(o["thresholds"]) for o in out], dtype=np.int64)
                    first = np.array([o["confidences"][0] for o in out])
                    b_ok, m_ok = b_pred == expected, m_pred == expected
                    row = rules.setdefault(rule_name(base, c_th), {})[part] = dict(
                        scores(m_pred, expected, len(labels)), conf=float(conf.mean()), steps=float(steps.mean()),
                        threshold=float(threshold.mean()), rounds=float(rounds.mean()), roundsMedian=float(np.median(rounds)), roundsMax=int(rounds.max()),
                        multiFraction=float((rounds > 1).mean()), firstConf=float(first.mean()),
                        changed=int((b_pred != m_pred).sum()), onlyBase=int((b_ok & ~m_ok).sum()), onlyMulti=int((m_ok & ~b_ok).sum()),
                        latencyUs=float(latency.mean()), latencyStdUs=float(latency.std()), seconds=time.time() - t0)
                    log("%-4s %-22s acc %.4f  balAcc %.4f  F1 %.4f  conf %.3f  steps %.2f  thresholds traced mean %.2f median %g max %d "
                        "(more than one %.1f%%)  changed %d (only base %d, only multi %d)  latency %.2f +- %.2f us  (%.2fs)",
                        part, rule_name(base, c_th), row["acc"], row["bal_acc"], row["macro_f1"], row["conf"], row["steps"],
                        row["rounds"], row["roundsMedian"], row["roundsMax"], 100 * row["multiFraction"],
                        row["changed"], row["onlyBase"], row["onlyMulti"], row["latencyUs"], row["latencyStdUs"], row["seconds"])
                    traces["%s|%s" % (part, rule_name(base, c_th))] = m_pred.astype(np.int16)
                    if c_th == max(task["thresholds"]):
                        # every round of every example: the threshold found and the confidence at it. The
                        # rounds of a smaller C_th are the first ones, up to the first confidence above it.
                        cut = np.concatenate([o["thresholds"] for o in out]).astype(np.int32)
                        cut_conf = np.concatenate([o["confidences"] for o in out]).astype(np.float32)
                        number = (np.arange(len(cut)) - np.repeat(np.cumsum(rounds) - rounds, rounds) + 1).astype(np.int16)
                        per_round = row["perRound"] = []
                        for k in range(1, int(rounds.max()) + 1):
                            here = number == k
                            per_round.append(dict(round=k, examples=int(here.sum()), threshold=float(cut[here].mean()),
                                                  thresholdMedian=float(np.median(cut[here])), thresholdMax=int(cut[here].max()),
                                                  confidence=float(cut_conf[here].mean()), tied=float((cut_conf[here] == 0).mean())))
                            log("%-4s %-10s round %2d: %7d examples  threshold mean %9.2f median %7.0f max %7d  confidence mean %.4f  tied %5.1f%%",
                                part, base, k, per_round[-1]["examples"], per_round[-1]["threshold"], per_round[-1]["thresholdMedian"],
                                per_round[-1]["thresholdMax"], per_round[-1]["confidence"], 100 * per_round[-1]["tied"])
                        if task["traces"]:
                            key = "%s|%s|" % (part, base)
                            traces[key + "example"] = np.repeat(np.arange(len(out), dtype=np.int32), rounds)
                            traces[key + "round"] = number
                            traces[key + "threshold"] = cut
                            traces[key + "confidence"] = cut_conf
            traces[part + "|expected"] = expected.astype(np.int16)
        result.update(size=p["size"], features=p["features"], rams=-(-p["size"] // task["address"]), labels=labels,
                      trainCounts=[int(n) for n in np.bincount([index[l] for l in _train_labels(p)], minlength=len(labels))],
                      rules=rules)
        if task["traces"]:
            path = artifact_path(task["resultsDir"], "traces", task, ".npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, labels=np.array(labels), **traces)
            log("traces %s", os.path.relpath(path, task["resultsDir"]))
    except Exception as e:  # recorded and reported; the rest of the grid goes on
        result["error"] = "%s: %s" % (type(e).__name__, e)
        log("ERROR %s\n%s", result["error"], traceback.format_exc())
    result["seconds"] = time.time() - start
    result["log"] = os.path.relpath(artifact_path(task["resultsDir"], "logs", task, ".log"), task["resultsDir"])
    log("done in %.1fs", result["seconds"])
    log.close()
    return result


def _train_labels(p):
    return [p["train"].getLabel(i) for i in range(p["train"].size())]


def task_key(t):
    return "bits=%(bits)d|address=%(address)d|mapping=%(mapping)s" % t


def results_path(results_dir, t):
    return os.path.join(results_dir, t["dataset"], t["encoding"], "seed%d.jsonl" % t["seed"])


def artifact_path(results_dir, kind, t, extension):
    """checkpoints, logs and traces of a network: <resultsDir>/<kind>/<dataset>/<encoding>/seed<k>/<network><extension>"""
    return os.path.join(results_dir, kind, t["dataset"], t["encoding"], "seed%d" % t["seed"],
                        "b%d_a%d_%s%s" % (t["bits"], t["address"], t["mapping"], extension))


def build_tasks(args):
    tasks = []
    for name in args.datasets:
        grid = rg.GRID["image" if name in datasets.IMAGES else "tabular"]
        bits_grid = args.bits or grid["bits"]
        ds = datasets.load(name, args.dataDir)  # once per dataset: the input size of every number of bits
        values = np.array([len(np.unique(ds.X[:, j])) for j in range(ds.X.shape[1])])  # as Encoder(binary_as_bit=True)
        sizes = {bits: int((~ds.nominal & (values >= 3)).sum()) * bits + int((values == 2).sum())
                 + int(values[ds.nominal & (values >= 3)].sum()) for bits in bits_grid}
        del ds
        for encoding in args.encodings:
            for seed in args.seeds:
                for bits in bits_grid:
                    for address in (args.address or grid["address"]):
                        if address > sizes[bits]:
                            continue
                        tasks.append(dict(dataset=name, encoding=encoding, seed=seed, bits=bits, address=address, mapping=MAPPING,
                                          thresholds=args.thresholds, tieTolerance=args.tieTolerance, nextCut=args.nextCut,
                                          traces=not args.no_traces, checkpoints=not args.no_checkpoints,
                                          resultsDir=args.resultsDir))
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
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=datasets.ALL_NAMES)
    parser.add_argument("--encodings", nargs="+", default=["distributive"], choices=ENCODINGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--bits", type=int, nargs="+", help="overrides the bits grid of run_grid.py")
    parser.add_argument("--address", type=int, nargs="+", help="overrides the address grid of run_grid.py")
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS, help="C_th of the multi rules")
    parser.add_argument("--tieTolerance", type=float, default=0.0, help="bbleaching: confidence <= it is a tie")
    parser.add_argument("--nextCut", default="ref", choices=("ref", "untie"), help="bbleaching in the later rounds")
    parser.add_argument("--no-traces", action="store_true", help="does not save, per example, the predictions and the threshold and confidence of every round")
    parser.add_argument("--no-checkpoints", action="store_true", help="does not save the trained model of every network")
    parser.add_argument("--dataDir", default="experiments/data")
    parser.add_argument("--resultsDir", default="experiments/results_multi")
    parser.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument("--dry-run", action="store_true", help="print the counts per dataset, run nothing")
    args = parser.parse_args()

    tasks = build_tasks(args)
    done = done_keys(args.resultsDir, tasks)
    pending = [t for t in tasks if (results_path(args.resultsDir, t), task_key(t)) not in done]
    count = collections.Counter(t["dataset"] for t in tasks)
    left = collections.Counter(t["dataset"] for t in pending)
    for name in args.datasets:
        print("  %-14s %5d networks, %5d to run" % (name, count[name], left[name]))
    print("%d networks x %d rules, %d already done, %d to run with %d jobs" % (
        len(tasks), len(BASES) * (1 + len(args.thresholds)), len(tasks) - len(pending), len(pending), args.jobs))
    if args.dry_run or not pending:
        return

    pool = multiprocessing.Pool(args.jobs, initializer=rg.init_worker, initargs=(dict(dataDir=args.dataDir),))
    start, errors = time.time(), 0
    try:
        for n, result in enumerate(pool.imap_unordered(run_task, pending), 1):
            for key in ("traces", "checkpoints", "resultsDir"):  # options of the run, not of the network
                result.pop(key, None)
            path = results_path(args.resultsDir, result)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(result) + "\n")
            if "error" in result:
                errors += 1
                print("ERROR %s %s: %s" % (result["dataset"], task_key(result), result["error"]))
                continue
            elapsed = time.time() - start
            test = lambda rule: result["rules"][rule]["test"]["macro_f1"]
            latency = lambda rule: "%.1f+-%.1f" % (result["rules"][rule]["test"]["latencyUs"], result["rules"][rule]["test"]["latencyStdUs"])
            print("[%d/%d %.0fs, eta %.1fh] %s seed%d %s  F1 test iter %.4f bb %.4f dab %.4f  latency us iter %s bb %s dab %s  (%.0fs)" % (
                n, len(pending), elapsed, elapsed / n * (len(pending) - n) / 3600, result["dataset"], result["seed"],
                task_key(result), test("iterative"), test("bbleaching"), test("dab"),
                latency("iterative"), latency("bbleaching"), latency("dab"), result["seconds"]), flush=True)
    finally:
        pool.terminate()
    print("finished: %d networks, %d errors, %.1f h" % (len(pending), errors, (time.time() - start) / 3600))


if __name__ == "__main__":
    main()
