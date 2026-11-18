"""
Equivalence check of trainEpochs across implementations: runs a fixed set of cases (options,
bleaching policies, input types, mappings, a class that appears late, a pre-trained model) and
records the report of every epoch, the counters of the delivered model and its classification.

  python test/epoch_equivalence.py --save test/epoch_reference.json   # with the old build
  python test/epoch_equivalence.py --check test/epoch_reference.json  # with the new build

--check also runs every case again with threads=4 (validation and address precomputation in
parallel) and expects the same result. A case that fails prints its first difference.
"""
import argparse
import json
import sys
from random import Random

import wisardpkg as wp


def data(n, size, classes, seed, late=None, nval=200):
    """n training and nval validation examples of size bits; class k has its own bit
    probabilities. late: label of a class whose training examples are only in the last third
    (its discriminator is created during epoch 0)"""
    rng = Random(seed)
    centers = [[rng.uniform(0.25, 0.75) for _ in range(size)] for _ in classes]

    def sample(k):
        return [1 if rng.random() < centers[k][j] else 0 for j in range(size)]

    X, y = [], []
    for i in range(n):
        k = i % len(classes)
        if late is not None and classes[k] == late and i < 2 * n // 3:
            k = (k + 1) % len(classes)
        X.append(sample(k))
        y.append(classes[k])
    Xv = [sample(i % len(classes)) for i in range(nval)]
    yv = [classes[i % len(classes)] for i in range(nval)]
    return X, y, Xv, yv


def mapping(size, address, seed, labels, shared=True):
    rng = Random(seed)
    out = {}
    for label in labels:
        order = list(range(size))
        rng.shuffle(order)
        rams = -(-size // address)
        order += [rng.randrange(size) for _ in range(rams * address - size)]
        out[label] = [order[r * address:(r + 1) * address] for r in range(rams)]
        if shared:
            return {other: out[label] for other in labels}
    return out


def to_dataset(X, y):
    ds = wp.DataSet()
    for x, label in zip(X, y):
        ds.add(wp.BinInput(x), label)
    return ds


def counters(wsd):
    out = {}
    for label, d in json.loads(wsd.json())["classes"].items():
        handle = wp.RAMDataHandle(d["data"])
        out[label] = {"count": d["count"],
                      "rams": [sorted(dict(handle.get(r)).items()) for r in range(len(d["mapping"]))]}
    return out


CLASSES2 = ["a", "b"]
CLASSES3 = ["x", "y", "z"]

CASES = [
    # name, data args, Wisard options, trainEpochs options
    ("default", dict(n=600, size=48, classes=CLASSES2, seed=1), dict(address=6), dict()),
    ("grid-like", dict(n=600, size=48, classes=CLASSES2, seed=2), dict(address=6),
     dict(b=1, rho0=0.9, tau=0.2, epochs=12, patience=5, valMode="bleaching", seed=3)),
    ("b2", dict(n=600, size=48, classes=CLASSES2, seed=3), dict(address=4),
     dict(b=2, rho0=0.5, tau=0.1, epochs=8, patience=3, seed=1)),
    ("dab", dict(n=600, size=48, classes=CLASSES3, seed=4), dict(address=6, bleaching="dab"),
     dict(rho0=0.7, tau=0.2, epochs=10, patience=4, valMode="bleaching", seed=2)),
    ("rdab", dict(n=600, size=48, classes=CLASSES3, seed=5), dict(address=6, bleaching="rdab"),
     dict(rho0=0.7, tau=0.2, epochs=10, patience=4, valMode="bleaching", seed=2)),
    ("lowest-count", dict(n=500, size=40, classes=CLASSES2, seed=6), dict(address=5),
     dict(selectError="lowest_count", selectMargin="lowest_count", kappa=3, tau=0.3, epochs=6)),
    ("select-all", dict(n=500, size=40, classes=CLASSES2, seed=7), dict(address=5),
     dict(selectMargin="all", tau=0.3, modulate=True, decay="geometric", gamma=0.7, epochs=6)),
    ("stochastic", dict(n=500, size=40, classes=CLASSES3, seed=8), dict(address=5),
     dict(counterMode="int_stochastic", rho0=0.6, tau=0.2, epochs=6, seed=9)),
    ("ignore-zero", dict(n=500, size=40, classes=CLASSES2, seed=9), dict(address=4, ignoreZero=True),
     dict(rho0=0.5, tau=0.2, epochs=6)),
    ("no-validation", dict(n=500, size=40, classes=CLASSES2, seed=10), dict(address=5),
     dict(rho0=0.5, tau=0.2, epochs=5), "noval"),
    ("late-class", dict(n=600, size=48, classes=CLASSES3, seed=11, late="z"), dict(address=6),
     dict(rho0=0.5, tau=0.2, epochs=6)),
    ("own-mappings", dict(n=600, size=48, classes=CLASSES3, seed=12), dict(address=6, shared=False),
     dict(rho0=0.5, tau=0.2, epochs=6, valMode="bleaching")),
    ("pretrained", dict(n=600, size=48, classes=CLASSES2, seed=13), dict(address=6),
     dict(rho0=0.5, tau=0.2, epochs=6), "pretrain"),
    ("address-10", dict(n=800, size=60, classes=CLASSES2, seed=14), dict(address=10),
     dict(rho0=0.5, tau=0.2, epochs=6)),
    ("address-20", dict(n=400, size=60, classes=CLASSES2, seed=16), dict(address=20),
     dict(rho0=0.5, tau=0.2, epochs=5)),
    ("lists", dict(n=500, size=40, classes=CLASSES3, seed=15), dict(address=5),
     dict(rho0=0.5, tau=0.2, epochs=6, valMode="bleaching"), "lists"),
]


def run(case, extra):
    name, dargs, wargs, targs = case[:4]
    flags = case[4] if len(case) > 4 else ""
    X, y, Xv, yv = data(**dargs)
    wargs = dict(wargs)
    address, shared = wargs.pop("address"), wargs.pop("shared", True)
    labels = sorted(set(y))
    wsd = wp.Wisard(address, mapping=mapping(dargs["size"], address, dargs["seed"], labels, shared), **wargs)
    if flags == "pretrain":
        wsd.train(X[:100], y[:100])
    if flags == "noval":
        Xv, yv = [], []
    if flags == "lists":
        report = wsd.trainEpochs(X, y, Xv, yv, **dict(targs, **extra))
        predicted = wsd.classify(Xv or X[:50])
    else:
        val = to_dataset(Xv, yv) if Xv else wp.DataSet()
        report = wsd.trainEpochs(to_dataset(X, y), val, **dict(targs, **extra))
        predicted = wsd.classify(to_dataset(Xv, yv) if Xv else to_dataset(X[:50], y[:50]))
    return {"report": report, "model": counters(wsd), "classify": predicted}


def first_difference(a, b, path=""):
    if type(a) != type(b):
        return "%s: %r != %r" % (path, a, b)
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return "%s/%s: only in one" % (path, k)
            d = first_difference(a[k], b[k], "%s/%s" % (path, k))
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return "%s: length %d != %d" % (path, len(a), len(b))
        for i, (u, v) in enumerate(zip(a, b)):
            d = first_difference(u, v, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    return None if a == b else "%s: %r != %r" % (path, a, b)


TIMES = ("secondsTrain", "secondsVal")  # wall times of every epoch, not compared


def normalized(result):
    result = json.loads(json.dumps(result))  # tuples become lists, as in the saved file
    for row in result["report"]["perEpoch"]:
        for key in TIMES:
            row.pop(key, None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save")
    group.add_argument("--check")
    args = parser.parse_args()

    if args.save:
        results = {case[0]: normalized(run(case, {})) for case in CASES}
        with open(args.save, "w") as f:
            json.dump(results, f)
        print("saved %d cases in %s" % (len(results), args.save))
        return

    with open(args.check) as f:
        reference = json.load(f)
    failed = 0
    for case in CASES:
        for extra in ({}, {"threads": 4}):
            label = case[0] + (" threads=4" if extra else "")
            d = first_difference(reference[case[0]], normalized(run(case, extra)))
            print("%-26s %s" % (label, "OK" if d is None else "DIFFERENT " + d))
            failed += d is not None
    print("all %d runs identical" % (2 * len(CASES)) if not failed else "%d runs differ" % failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
