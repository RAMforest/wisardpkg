from unittest import TestCase, SkipTest
from random import Random
import json
import math
import os
import sys
import wisardpkg as wp

INFO = dict(returnConfidence=True, returnBleachingInfo=True)
# (confidenceTolerance, freqTolerance, dabRemoval): tolerance 0 decides at the first step
RDAB_CONFIGS = [(tol, freq, removal)
                for tol in (0.0, 0.01, 0.05, 0.2, 0.5)
                for freq in (0.0, 0.05, 0.1, 0.3, 0.5, 1.0)
                for removal in ("inclusive", "exclusive")]
EXPERIMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")


# ---------------------------------------------------------------- reference implementations
# votes: {label: [counter read in each RAM]}, as Wisard.votes returns; labels in std::map order

def biggest_candidate(candidates):
    label, biggest = "", 0
    for k in sorted(candidates):
        if candidates[k] >= biggest:
            label, biggest = k, candidates[k]
    return label


def iterative(votes, activated=True, confidence=1):
    """Bleaching::makeConfidenceless + getBiggestCandidate + getConfidence"""
    b, steps = 1, 0
    while True:
        threshold, steps = b, steps + 1
        labels = {k: sum(v >= b for v in vs) for k, vs in votes.items()}
        if not activated:
            break
        b += 1
        biggest, ambiguity = 0, False
        for k in sorted(labels):
            if labels[k] > biggest:
                biggest, ambiguity = labels[k], False
            elif biggest - labels[k] < confidence:
                ambiguity = True
        if not (ambiguity and biggest > 1):
            break
    label = biggest_candidate(labels)
    top = labels.get(label, 0)
    second = max([v for v in labels.values() if v < top] + [0])
    return dict(label=label, confidence=(top - second) / top if top else float("nan"), steps=steps, threshold=threshold)


def top2(values):
    """(label, first, second); ties go to the last label"""
    label, v1, v2 = "", 0.0, 0.0
    for k in sorted(values):
        if values[k] >= v1:
            label, v1, v2 = k, values[k], v1
        elif values[k] > v2:
            v2 = values[k]
    return label, v1, v2


def dab(votes, sizes=None, tol=0.01, freq=None, inclusive=True):
    """dab-dev.md, part 1: dab, or rdab when freq is given"""
    keep, previous, step = 1, None, 0
    while True:
        step += 1
        counts = {k: sum(v >= keep for v in vs) for k, vs in votes.items()}
        label, r1, r2 = top2(counts)
        if r1 == 0:
            if step == 1:
                return dict(label=label, confidence=0.0, steps=1, threshold=1)
            return dict(previous, steps=step)
        current = dict(label=label, confidence=(r1 - r2) / r1, steps=step, threshold=keep)
        if current["confidence"] >= tol:
            return current
        if freq is not None:
            f = {k: (sum(v for v in vs if v >= keep) / sizes[k] if sizes.get(k) else 0.0) for k, vs in votes.items()}
            flabel, f1, f2 = top2(f)
            cf = (f1 - f2) / f1 if f1 > 0 else 0.0
            if cf >= freq:
                return dict(current, label=flabel, confidence=cf)
        tied = [k for k in votes if counts[k] > 0 and (r1 - counts[k]) / r1 < tol]
        tau_b = max(min(v for v in votes[k] if v >= keep) for k in tied)
        nxt = tau_b + 1 if inclusive else tau_b
        if nxt <= keep:
            return current
        previous, keep = current, nxt


# ---------------------------------------------------------------- helpers

def with_counters(spec, **options):
    """4 RAMs (address 2, fixed mapping); on x = [0]*8 the counters of label are spec[label][0];
    spec[label][1] extra examples touch no counter of x and only grow the class size"""
    X, y = [], []
    for label, (counters, extra) in spec.items():
        for ram, k in enumerate(counters):
            sample = [1] * 8
            sample[2 * ram] = sample[2 * ram + 1] = 0
            X += [sample] * k
            y += [label] * k
        X += [[1] * 8] * extra
        y += [label] * extra
    wsd = wp.Wisard(2, indexes=list(range(8)), **options)
    wsd.train(X, y)
    return wsd, [0] * 8


def random_model(seed, address=3, n=60, entry=24):
    rng = Random(seed)
    centers = {label: [rng.random() for _ in range(entry)] for label in "abc"}

    def sample(label):
        return [1 if rng.random() < 0.5 * centers[label][j] + 0.25 else 0 for j in range(entry)]

    X = [sample("abc"[i % 3]) for i in range(n)]
    y = ["abc"[i % 3] for i in range(n)]
    wsd = wp.Wisard(address, indexes=list(range(entry)))
    wsd.train(X, y)
    return wsd, [sample("abc"[i % 3]) for i in range(150)]


def configured(wsd, **options):
    """the same model with other classify options (through json)"""
    return wp.Wisard(json.dumps(dict(json.loads(wsd.json()), **options)))


def class_sizes(wsd):
    return {label: d["count"] for label, d in json.loads(wsd.json())["classes"].items()}


def assert_decision(test, out, expected):
    """the classify output (with INFO) against a reference decision"""
    test.assertEqual(out["class"], expected["label"])
    test.assertEqual(out["steps"], expected["steps"])
    test.assertEqual(out["finalThreshold"], expected["threshold"])
    if math.isnan(expected["confidence"]):
        test.assertTrue(math.isnan(out["confidence"]))
    else:
        test.assertAlmostEqual(out["confidence"], expected["confidence"], places=6)


class WisardBleachingTestCase(TestCase):

    def assertDecision(self, out, expected):
        assert_decision(self, out, expected)

    # 5a: iterative is the bleaching of the package (also checked against the build before the change)
    def test_iterative_matches_reference(self):
        for seed in range(4):
            for options in (dict(), dict(confidence=2), dict(bleachingActivated=False)):
                wsd, X = random_model(seed)
                outs = configured(wsd, **dict(INFO, **options)).classify(X)
                for x, out in zip(X, outs):
                    self.assertDecision(out, iterative(wsd.votes(x), options.get("bleachingActivated", True), options.get("confidence", 1)))

    def test_dab_and_rdab_match_reference(self):
        for seed in range(4):
            wsd, X = random_model(seed)
            sizes = class_sizes(wsd)
            votes = [wsd.votes(x) for x in X]
            for removal in ("inclusive", "exclusive"):
                for tol in (0.01, 0.2):
                    outs = configured(wsd, bleaching="dab", confidenceTolerance=tol, dabRemoval=removal, **INFO).classify(X)
                    for v, out in zip(votes, outs):
                        self.assertDecision(out, dab(v, tol=tol, inclusive=removal == "inclusive"))
                    for freq in (0.05, 0.3):
                        outs = configured(wsd, bleaching="rdab", confidenceTolerance=tol, freqTolerance=freq,
                                          dabRemoval=removal, **INFO).classify(X)
                        for v, out in zip(votes, outs):
                            self.assertDecision(out, dab(v, sizes, tol, freq, removal == "inclusive"))

    # 5b: Algorithm 1 by hand. A = {4, 4, 1, 0}, B = {2, 2, 2, 0}: 3 x 3 votes at the first step
    def test_algorithm1_by_hand(self):
        wsd, x = with_counters({"A": ([4, 4, 1, 0], 0), "B": ([2, 2, 2, 0], 0)})
        self.assertEqual(wsd.votes(x), {"A": [4, 4, 1, 0], "B": [2, 2, 2, 0]})
        # iterative: b = 2 -> A 2, B 3
        out = configured(wsd, **INFO).classify([x])[0]
        self.assertEqual((out["class"], out["steps"], out["finalThreshold"]), ("B", 2, 2))
        # dab: tau_b = max(min A, min B) = max(1, 2) = 2; removing <= 2 leaves A {4, 4}, B {}
        out = configured(wsd, bleaching="dab", **INFO).classify([x])[0]
        self.assertEqual((out["class"], out["steps"], out["finalThreshold"], out["confidence"]), ("A", 2, 3, 1.0))
        # removing only < 2 leaves A {4, 4}, B {2, 2, 2}: c = 1/3
        out = configured(wsd, bleaching="dab", dabRemoval="exclusive", **INFO).classify([x])[0]
        self.assertEqual((out["class"], out["steps"], out["finalThreshold"]), ("B", 2, 2))
        self.assertAlmostEqual(out["confidence"], 1 / 3, places=6)

    def test_dab_returns_last_votes_when_everything_is_removed(self):
        # 2 x 2 votes of 2; removing <= 2 empties both: the first step decides (tie: last label)
        wsd, x = with_counters({"A": ([2, 2, 0, 0], 0), "B": ([2, 2, 0, 0], 0)})
        out = configured(wsd, bleaching="dab", **INFO).classify([x])[0]
        self.assertEqual((out["class"], out["steps"], out["finalThreshold"], out["confidence"]), ("B", 2, 1, 0.0))

    def test_rdab_uses_class_size(self):
        # same votes as above, but A has 9 more examples: f_A = 9/18, f_B = 6/6 -> c_f = 0.5
        wsd, x = with_counters({"A": ([4, 4, 1, 0], 9), "B": ([2, 2, 2, 0], 0)})
        self.assertEqual(class_sizes(wsd), {"A": 18, "B": 6})
        out = configured(wsd, bleaching="rdab", freqTolerance=0.1, **INFO).classify([x])[0]
        self.assertEqual((out["class"], out["steps"], out["finalThreshold"]), ("B", 1, 1))
        self.assertAlmostEqual(out["confidence"], 0.5, places=6)
        self.assertEqual(configured(wsd, bleaching="dab").classify([x]), ["A"])

    def test_rdab_configurations_match_reference(self):
        for seed in range(2):
            for address in (2, 3, 6):
                wsd, X = random_model(seed, address=address)
                sizes = class_sizes(wsd)
                votes = [wsd.votes(x) for x in X]
                for tol, freq, removal in RDAB_CONFIGS:
                    outs = configured(wsd, bleaching="rdab", confidenceTolerance=tol, freqTolerance=freq,
                                      dabRemoval=removal, **INFO).classify(X)
                    for v, out in zip(votes, outs):
                        self.assertDecision(out, dab(v, sizes, tol, freq, removal == "inclusive"))

    def test_rdab_frequency_zero_decides_at_first_step(self):
        # c_f >= 0 always: when c does not decide, the frequency does, without a new threshold
        wsd, X = random_model(1)
        outs = configured(wsd, bleaching="rdab", freqTolerance=0.0, **INFO).classify(X)
        self.assertEqual({out["steps"] for out in outs}, {1})
        self.assertEqual({out["finalThreshold"] for out in outs}, {1})

    def test_rdab_imbalanced_classes(self):
        # the same votes (A = {4, 4, 1, 0}, B = {2, 2, 2, 0}) with other class sizes:
        #   A 9 x B 6:  f_A = f_B = 1, the frequency does not decide -> the dab decision, A
        #   A 18 x B 6: f_A = 9/18, f_B = 6/6 -> B by the frequency
        expected = {0: "A", 9: "B"}
        for extra_a, label in expected.items():
            wsd, x = with_counters({"A": ([4, 4, 1, 0], extra_a), "B": ([2, 2, 2, 0], 0)})
            out = configured(wsd, bleaching="rdab", freqTolerance=0.1, **INFO).classify([x])[0]
            self.assertEqual(out["class"], label)
        # with A = 9 and B = 6 + 12 examples: f_A = 9/9, f_B = 6/18 -> A by the frequency
        wsd, x = with_counters({"A": ([4, 4, 1, 0], 0), "B": ([2, 2, 2, 0], 12)})
        out = configured(wsd, bleaching="rdab", freqTolerance=0.1, **INFO).classify([x])[0]
        self.assertEqual((out["class"], out["steps"]), ("A", 1))
        self.assertAlmostEqual(out["confidence"], 1 - (6 / 18), places=6)
        # a high freqTolerance falls back to the thresholds (the dab decision)
        out = configured(wsd, bleaching="rdab", freqTolerance=0.9, **INFO).classify([x])[0]
        self.assertEqual((out["class"], out["steps"], out["finalThreshold"]), ("A", 2, 3))

    # 5c: without a tie at the first step the three policies agree
    def test_policies_agree_without_tie(self):
        for seed in range(4):
            wsd, X = random_model(seed)
            confident = [x for x in X if dab(wsd.votes(x))["steps"] == 1 and dab(wsd.votes(x))["confidence"] >= 0.01]
            self.assertTrue(confident)
            plain = [configured(wsd, bleaching=mode).classify(confident) for mode in ("iterative", "dab", "rdab")]
            self.assertEqual(plain[0], plain[1])
            self.assertEqual(plain[0], plain[2])

    # 5d: c_f >= 1 needs every other class without votes, which c already decides
    def test_rdab_with_freq_tolerance_one_is_dab(self):
        for seed in range(4):
            wsd, X = random_model(seed)
            self.assertEqual(configured(wsd, bleaching="rdab", freqTolerance=1.0, **INFO).classify(X),
                             configured(wsd, bleaching="dab", **INFO).classify(X))

    def test_options_json_and_errors(self):
        wsd, X = random_model(0)
        model = configured(wsd, bleaching="rdab", confidenceTolerance=0.05, freqTolerance=0.2, dabRemoval="exclusive")
        config = json.loads(model.json())
        self.assertEqual((config["bleaching"], config["confidenceTolerance"], config["freqTolerance"], config["dabRemoval"]),
                         ("rdab", 0.05, 0.2, "exclusive"))
        self.assertEqual(wp.Wisard(model.json()).classify(X), model.classify(X))
        self.assertEqual(wp.Wisard(3, bleaching="dab").json().count('"bleaching":"dab"'), 1)
        for options in (dict(bleaching="other"), dict(dabRemoval="other"), dict(confidenceTolerance=1.5), dict(freqTolerance=-0.1)):
            with self.assertRaises(Exception):
                wp.Wisard(3, **options)

    # Level A: an epoch-trained model is an ordinary WiSARD; its class sizes are the counts
    def test_epoch_model(self):
        rng = Random(3)
        X = [[rng.randint(0, 1) for _ in range(24)] for _ in range(90)]
        y = ["abc"[i % 3] for i in range(90)]
        wsd = wp.Wisard(3, indexes=list(range(24)))
        wsd.trainEpochs(X[:60], y[:60], X[60:], y[60:], b=1, rho0=0.5, tau=0.1, epochs=6, seed=0)
        sizes = class_sizes(wsd)
        self.assertEqual(sizes, {"a": 20, "b": 20, "c": 20})
        outs = configured(wsd, bleaching="rdab", **INFO).classify(X)
        for x, out in zip(X, outs):
            self.assertDecision(out, dab(wsd.votes(x), sizes, 0.01, 0.1))

    # valMode="bleaching" validates every epoch with the bleaching policy of the model
    def test_epoch_validation_uses_the_policy(self):
        # with one epoch the delivered model is the trained one: accVal is the classify accuracy
        differ = False
        for seed in range(6):
            _, X = random_model(seed)
            y = ["abc"[i % 3] for i in range(len(X))]
            accs = {}
            for mode in ("iterative", "dab", "rdab"):
                model = wp.Wisard(3, indexes=list(range(24)), bleaching=mode)
                report = model.trainEpochs(X[:90], y[:90], X[90:], y[90:], b=1, rho0=0.5, tau=0.1,
                                           epochs=1, valMode="bleaching", seed=seed)
                accs[mode] = sum(p == t for p, t in zip(model.classify(X[90:]), y[90:])) / 60
                self.assertAlmostEqual(report["perEpoch"][0]["accVal"], accs[mode], places=12)
            differ = differ or len(set(accs.values())) > 1
        self.assertTrue(differ)  # otherwise the test could not tell the policies apart


class SegmentEpochsBleachingTestCase(TestCase):
    """segment (benchmark split, seed 1, thermometer 8 bits, address 16) trained by epochs and
    classified with iterative, dab and rdab; needs experiments/data and scikit-learn/pandas"""

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(os.path.join(EXPERIMENTS, "data", "openml")):
            raise SkipTest("experiments/data/openml not found")
        sys.path.insert(0, EXPERIMENTS)
        try:
            import datasets
            import run_grid as rg
            from encoders import Encoder, to_dataset
        except ImportError as e:
            raise SkipTest("experiments dependencies missing: %s" % e)
        ds = datasets.load("segment", os.path.join(EXPERIMENTS, "data"))
        (Xtr, ytr), (Xva, yva), (Xte, yte) = rg.split(ds, 1)
        encoder = Encoder("thermometer", 8).fit(Xtr, ds.nominal)
        train, val, cls.test = (to_dataset(encoder.transform(X), y, encoder.size)
                                for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte)))
        cls.labels = list(yte)
        cls.model = wp.Wisard(16, mapping=rg.mapping(encoder.size, 16, 1, sorted(set(ytr))))
        cls.model.trainEpochs(train, val, b=1, rho0=0.5, tau=0.1, epochs=20, patience=3, valMode="bleaching", seed=1)
        cls.sizes = class_sizes(cls.model)
        cls.votes = [cls.model.votes(cls.test.get(i)) for i in range(cls.test.size())]
        cls.train_sizes = {label: list(ytr).count(label) for label in set(ytr)}

    def classify(self, **options):
        return configured(self.model, **dict(INFO, **options)).classify(self.test)

    def accuracy(self, outs):
        return sum(out["class"] == label for out, label in zip(outs, self.labels)) / len(self.labels)

    def assertMatches(self, outs, reference):
        for v, out in zip(self.votes, outs):
            assert_decision(self, out, reference(v))

    def test_class_sizes_are_the_training_sizes(self):
        self.assertEqual(self.sizes, self.train_sizes)

    def test_iterative(self):
        outs = self.classify(bleaching="iterative")
        self.assertMatches(outs, iterative)
        self.assertGreater(self.accuracy(outs), 0.95)

    def test_dab(self):
        for removal in ("inclusive", "exclusive"):
            outs = self.classify(bleaching="dab", dabRemoval=removal)
            self.assertMatches(outs, lambda v: dab(v, inclusive=removal == "inclusive"))
            self.assertGreater(self.accuracy(outs), 0.95)

    def test_rdab(self):
        self.assertGreater(self.accuracy(self.classify(bleaching="rdab")), 0.95)  # confidenceTolerance 0.01, freqTolerance 0.1
        # every configuration decides as the reference; the accuracy is not checked, since a high
        # confidenceTolerance with a low freqTolerance hands most decisions to the frequency at
        # the first step (about 0.75 on segment with 0.5 x 0.05)
        for tol, freq, removal in RDAB_CONFIGS:
            outs = self.classify(bleaching="rdab", confidenceTolerance=tol, freqTolerance=freq, dabRemoval=removal)
            self.assertMatches(outs, lambda v: dab(v, self.sizes, tol, freq, removal == "inclusive"))
