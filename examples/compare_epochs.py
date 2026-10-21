"""
One-shot WiSARD (with iterative bleaching) vs WiSARD trained by epochs, following a reduced
version of the protocol of section 8 of requisito-implementacao.md:

  1. stratified 70/15/15 split (train / validation / test), the same for every model;
  2. b, rho0 and tau of the epoch training are chosen together by a grid on the validation,
     classifying with bleaching (the inference that is used); on ties the weakest correction
     (smallest rho0, then tau, then b) is kept;
  3. one run per seed, reporting mean and standard deviation;
  4. the test set is evaluated once per seed, on the chosen configuration.

Two baselines are reported: one-shot with iterative bleaching (the reference of section 8) and
one-shot classified at a fixed threshold b chosen on the validation (a stronger reference).

--valMode chooses how trainEpochs picks the best epoch: bleaching (default), fixed (validation
at the fixed threshold b, section 7) or none (no early stopping, all --epochs are used).

Usage:
  python examples/compare_epochs.py
  python examples/compare_epochs.py --dataset overlap --mix 0.3 --valMode none --epochs 5 --verbose
"""
import argparse
import gzip
import math
import multiprocessing
import os
import struct
from random import Random

import wisardpkg as wp

B_GRID = [1, 2]
RHO_GRID = [0.1, 0.15, 0.25, 0.35,0.5, 0.6]
TAU_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
B_FIXED_GRID = [1, 2, 3, 5, 10, 15, 20, 25, 30]  # only for the fixed threshold baseline


# ---------------------------------------------------------------- data

def noise_dataset(n, rng):
    """2 classes, each bit drawn independently (0.35 vs 0.65 on each half): spread noise."""
    X, y = [], []
    for i in range(n):
        label = "a" if i % 2 == 0 else "b"
        center = 0.35 if label == "a" else 0.65
        X.append([1 if rng.random() < (center if j < 15 else 1 - center) else 0 for j in range(30)])
        y.append(label)
    return X, y


def overlap_dataset(n, rng, classes=3, size=64, flip=0.1, mix=0.2):
    """One prototype per class; with probability `mix` a sample of class c is built from the
    prototype of another class, so the classes overlap and some addresses become impure."""
    prototypes = [[rng.randint(0, 1) for _ in range(size)] for _ in range(classes)]
    X, y = [], []
    for i in range(n):
        c = i % classes
        source = c
        if rng.random() < mix:
            source = rng.choice([k for k in range(classes) if k != c])
        X.append([bit ^ (1 if rng.random() < flip else 0) for bit in prototypes[source]])
        y.append("c%d" % c)
    return X, y


def read_idx(path):
    """IDX file of the original MNIST distribution (optionally gzipped): (shape, flat bytes)"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    zero, dtype, ndim = struct.unpack(">HBB", data[:4])
    if zero != 0 or dtype != 0x08:
        raise ValueError("%s is not an unsigned byte IDX file" % path)
    shape = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return shape, data[4 + 4 * ndim:]


def mnist_dataset(directory, threshold=128):
    """The 70000 MNIST images (official train + test, split again by the protocol), binarized
    with pixel >= threshold."""
    X, y = [], []
    for prefix in ("train", "t10k"):
        def find(kind, dims):
            name = "%s-%s-idx%d-ubyte" % (prefix, kind, dims)
            for candidate in (name, name + ".gz"):
                if os.path.exists(os.path.join(directory, candidate)):
                    return os.path.join(directory, candidate)
            raise IOError("%s(.gz) not found in %s" % (name, directory))
        (count, rows, cols), pixels = read_idx(find("images", 3))
        (labels_count,), labels = read_idx(find("labels", 1))
        if labels_count != count:
            raise ValueError("MNIST images and labels have different sizes")
        size = rows * cols
        for i in range(count):
            X.append([1 if p >= threshold else 0 for p in pixels[i * size:(i + 1) * size]])
            y.append(str(labels[i]))
    return X, y


def stratified_subsample(X, y, n, rng):
    idx = stratified_split(list(range(len(y))), y, rng, fractions=(n / float(len(y)), 1 - n / float(len(y))))[0][0]
    return [X[i] for i in idx], [y[i] for i in idx]


def stratified_split(X, y, rng, fractions=(0.70, 0.15, 0.15)):
    parts = [([], []) for _ in fractions]
    for label in sorted(set(y)):
        idx = [i for i in range(len(y)) if y[i] == label]
        rng.shuffle(idx)
        start = 0
        for k, fraction in enumerate(fractions):
            end = len(idx) if k == len(fractions) - 1 else start + int(round(fraction * len(idx)))
            for i in idx[start:end]:
                parts[k][0].append(X[i])
                parts[k][1].append(y[i])
            start = end
    return parts


# ---------------------------------------------------------------- models

def confidence(values):
    """C = (R_w - R_l) / R_w of section 2; 0 on ties (Bleaching::getConfidence is not used
    because it skips tied values and reports a high confidence on ties)."""
    top = sorted(values, reverse=True) + [0]
    return (top[0] - top[1]) / float(top[0]) if top[0] > 0 else 0.0


def predict(model, X):
    """classify (iterative bleaching); the confidence is measured on the votes where the
    bleaching stopped (the degrees are the votes divided by their sum, so the ratio is the same)"""
    out = model.classify(X)
    return [o["class"] for o in out], [confidence([c["degree"] for c in o["classesDegrees"]]) for o in out]


def predict_fixed(model, X, b):
    """classification at a fixed threshold b, without bleaching; ties go to the smallest label,
    as in the epoch training"""
    labels, confidences = [], []
    for x in X:
        R = model.responses(x, b)
        best = max(R.values())
        labels.append(min(k for k, v in R.items() if v == best))
        confidences.append(confidence(list(R.values())))
    return labels, confidences


def evaluate(prediction, expected):
    labels, confidences = prediction
    hits = [p == t for p, t in zip(labels, expected)]
    mean = lambda v: sum(v) / float(len(v)) if v else float("nan")
    return dict(acc=mean([1.0 if h else 0.0 for h in hits]), conf=mean(confidences),
                conf_hit=mean([c for c, h in zip(confidences, hits) if h]),
                conf_miss=mean([c for c, h in zip(confidences, hits) if not h]))


def new_model(addressSize, indexes):
    return wp.Wisard(addressSize, indexes=indexes, returnClassesDegrees=True)


def run_seed(X, y, seed, args):
    rng = Random(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, rng)
    indexes = list(range(len(X[0])))
    rng.shuffle(indexes)  # same mapping for baseline and proposal

    baseline = new_model(args.address, indexes)
    baseline.train(Xtr, ytr)
    b_fixed = max(B_FIXED_GRID, key=lambda v: (evaluate(predict_fixed(baseline, Xva, v), yva)["acc"], -v))

    early_stopping = args.valMode != "none"
    chosen, scores = None, []
    for rho0 in RHO_GRID:
        for tau in TAU_GRID:
            for b in B_GRID:
                model = new_model(args.address, indexes)
                options = dict(b=b, rho0=rho0, tau=tau, epochs=args.epochs, patience=args.patience, seed=seed)
                if early_stopping:
                    options["valMode"] = args.valMode
                report = model.trainEpochs(Xtr, ytr, Xva if early_stopping else [], yva if early_stopping else [],
                                           **options)
                # the configuration is chosen with the inference that will be used (bleaching)
                score = evaluate(predict(model, Xva), yva)["acc"]
                scores.append(score)
                # weakest correction first in the grid order, so a strict > keeps it on ties
                if chosen is None or score > chosen["val"]:
                    chosen = dict(val=score, b=b, rho0=rho0, tau=tau, model=model, report=report)

    result = dict(
        seed=seed, b=chosen["b"], b_fixed=b_fixed,
        baseline_val=evaluate(predict(baseline, Xva), yva),
        baseline_test=evaluate(predict(baseline, Xte), yte),
        fixed_val=evaluate(predict_fixed(baseline, Xva, b_fixed), yva),
        fixed_test=evaluate(predict_fixed(baseline, Xte, b_fixed), yte),
        epochs_val=evaluate(predict(chosen["model"], Xva), yva),
        epochs_test=evaluate(predict(chosen["model"], Xte), yte),
        rho0=chosen["rho0"], tau=chosen["tau"],
        tied=sum(s == chosen["val"] for s in scores), configs=len(scores),
        best_epoch=chosen["report"]["bestEpoch"],
        ran=len(chosen["report"]["perEpoch"]),
    )
    if args.verbose:
        print("  seed %d, per epoch of the chosen configuration:" % seed)
        for row in chosen["report"]["perEpoch"]:
            print("    epoch %(epoch)2d  rho %(rho).3f  bT %(bT)3d  accTrain %(accTrain).3f  "
                  "accVal %(accVal).3f  penError %(nPenError)4d  penMargin %(nPenMargin)4d" % row)
    return result


_mnist = {}


def load_data(args, seed):
    data_rng = Random(1000 + seed)
    if args.dataset == "noise":
        return noise_dataset(args.n or 600, data_rng)
    if args.dataset == "overlap":
        return overlap_dataset(args.n or 600, data_rng, mix=args.mix)
    if "data" not in _mnist:  # read once per process
        _mnist["data"] = mnist_dataset(args.mnistDir, args.threshold)
    X, y = _mnist["data"]
    return (X, y) if args.n is None else stratified_subsample(X, y, args.n, data_rng)


def run_job(job):
    seed, args = job
    X, y = load_data(args, seed)
    return run_seed(X, y, seed, args)


# ---------------------------------------------------------------- report

def mean_std(values):
    m = sum(values) / len(values)
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return m, s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["noise", "overlap", "mnist"], default="noise")
    parser.add_argument("--n", type=int, default=None,
                        help="number of examples (synthetic: 600 by default; mnist: stratified subsample of the 70000)")
    parser.add_argument("--mnistDir", default="mnist", help="folder with the MNIST IDX files (.gz or not)")
    parser.add_argument("--threshold", type=int, default=128, help="MNIST binarization: pixel >= threshold")
    parser.add_argument("--mix", type=float, default=0.2, help="overlap between classes (overlap dataset)")
    parser.add_argument("--address", type=int, default=3, help="address size")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--valMode", choices=["bleaching", "fixed", "none"], default="bleaching",
                        help="how trainEpochs picks the best epoch (none: no early stopping)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--jobs", type=int, default=1, help="seeds run in parallel processes")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    jobs = [(seed, args) for seed in args.seeds]
    if args.jobs > 1:
        pool = multiprocessing.Pool(min(args.jobs, len(jobs)))
        results = pool.map(run_job, jobs)
        pool.close()
    else:
        results = [run_job(job) for job in jobs]

    for r in results:
        print("seed %d | one-shot+bleaching test acc %.3f conf %.3f | one-shot b=%-2d test acc %.3f conf %.3f | "
              "epochs test acc %.3f conf %.3f (val %.3f, b=%d rho0=%s tau=%s, %d/%d configs tied, best epoch %d of %d)" % (
                  r["seed"], r["baseline_test"]["acc"], r["baseline_test"]["conf"], r["b_fixed"],
                  r["fixed_test"]["acc"], r["fixed_test"]["conf"], r["epochs_test"]["acc"], r["epochs_test"]["conf"],
                  r["epochs_val"]["acc"], r["b"], r["rho0"], r["tau"], r["tied"], r["configs"],
                  r["best_epoch"], r["ran"]))

    models = (("epochs + bleaching", "epochs"), ("one-shot + bleaching", "baseline"),
              ("one-shot, fixed b from validation", "fixed"))
    print("\n%d seeds (%s dataset, valMode=%s); confidence C = (R_w - R_l)/R_w on the votes "
          "where each model decides" % (len(results), args.dataset, args.valMode))
    for split, suffix in (("validation (used to choose the configurations)", "_val"), ("test", "_test")):
        print("\n  %s" % split)
        print("  %-34s %-15s %-15s %-12s %-12s" % ("model", "accuracy", "confidence", "conf (hits)", "conf (errors)"))
        for name, key in models:
            rows = [r[key + suffix] for r in results]
            acc, conf = mean_std([v["acc"] for v in rows]), mean_std([v["conf"] for v in rows])
            print("  %-34s %.3f +- %.3f  %.3f +- %.3f  %.3f        %.3f" % (
                name, acc[0], acc[1], conf[0], conf[1],
                mean_std([v["conf_hit"] for v in rows])[0], mean_std([v["conf_miss"] for v in rows])[0]))

    prop = [r["epochs_test"]["acc"] for r in results]
    print("\n  paired test accuracy difference of epochs + bleaching:")
    for name, key in models[1:]:
        diff = [p - r[key + "_test"]["acc"] for p, r in zip(prop, results)]
        m, s = mean_std(diff)
        line = "    vs %-34s %+.3f +- %.3f  (wins %d, ties %d, losses %d)" % (
            name + ":", m, s, sum(d > 0 for d in diff), sum(d == 0 for d in diff), sum(d < 0 for d in diff))
        if len(diff) > 1 and s > 0:
            line += "  t = %.2f" % (m / (s / math.sqrt(len(diff))))
        print(line)
    if len(results) > 1:
        print("    (paired t with %d degrees of freedom; |t| > 2.776 is significant at 5%% for 5 seeds)"
              % (len(results) - 1))


if __name__ == "__main__":
    main()
