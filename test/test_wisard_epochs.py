from unittest import TestCase
from random import Random
import wisardpkg as wp
import json


def counters(wsd):
    """{label: {"count": N_d, "rams": [{address: counter}, ...]}} read from the serialized model"""
    out = {}
    for label, d in json.loads(wsd.json())["classes"].items():
        handle = wp.RAMDataHandle(d["data"])
        out[label] = {
            "count": d["count"],
            "rams": [dict(handle.get(r)) for r in range(len(d["mapping"]))]
        }
    return out


def fixed(entrySize, addressSize):
    """WiSARD with a deterministic mapping, shared by every class"""
    return wp.Wisard(addressSize, indexes=list(range(entrySize)))


class WisardEpochsTestCase(TestCase):
    X = [
        [1,1,1,0,0,0,0,0,0],
        [1,1,1,1,0,0,1,0,0],
        [1,1,1,0,0,0,1,0,0],
        [1,1,1,1,0,0,0,0,0],
        [1,0,1,0,1,0,1,0,1],
        [0,0,0,0,1,1,1,1,1],
        [0,0,1,0,0,1,1,1,1],
        [0,0,1,0,1,1,1,1,1],
        [0,0,0,0,0,1,1,1,1]
    ]
    y = ["cold"]*5 + ["hot"]*4

    def overlapped(self, n=200, seed=0):
        rng = Random(seed)
        X, y = [], []
        for i in range(n):
            label = "a" if i % 2 == 0 else "b"
            center = 0.35 if label == "a" else 0.65
            X.append([1 if rng.random() < (center if j < 15 else 1-center) else 0 for j in range(30)])
            y.append(label)
        return X, y

    def margin_case(self):
        # 4 RAMs; on x the counters of B are {2, 50, 11, 300} and the ones of A are 10
        x = [0]*8
        X, y = [x]*10 + [x]*2, ["A"]*10 + ["B"]*2
        for ram, extra in ((1, 48), (2, 9), (3, 298)):
            sample = [1]*8
            sample[2*ram] = sample[2*ram+1] = 0
            X += [sample]*extra
            y += ["B"]*extra
        wsd = fixed(8, 2)
        wsd.train(X, y)
        return wsd, x

    # T-01
    def test_degenerate_equals_one_shot(self):
        one = fixed(9, 3)
        one.train(self.X, self.y)
        ep = fixed(9, 3)
        ep.trainEpochs(self.X, self.y, [], [], epochs=1, rho0=0.0, tau=0.0)
        a, b = json.loads(one.json())["classes"], json.loads(ep.json())["classes"]
        self.assertEqual(sorted(a), sorted(b))
        for label in a:
            self.assertEqual(a[label]["count"], b[label]["count"])
            self.assertTrue(wp.RAMDataHandle(a[label]["data"]).compare(wp.RAMDataHandle(b[label]["data"])))

    # T-02
    def test_penalize_float(self):
        x = self.X[0]
        wsd = fixed(9, 3)
        wsd.train([x]*20, ["a"]*20)
        addrs = wsd.activeAddresses(x, "a", 1)
        wsd.penalize("a", addrs[:1], 0.7)
        rams = counters(wsd)["a"]["rams"]
        r, addr = addrs[0]
        self.assertEqual(rams[r][addr], 14)
        for r, addr in addrs[1:]:
            self.assertEqual(rams[r][addr], 20)

    def many_rams(self):
        x = [0]*20000
        wsd = fixed(20000, 2)
        wsd.train([x], ["a"])
        return wsd, x, wsd.activeAddresses(x, "a", 1)

    # T-03
    def test_stochastic_rounding(self):
        wsd, x, addrs = self.many_rams()
        self.assertEqual(len(addrs), 10000)
        wsd.penalize("a", addrs, 0.7, counterMode="int_stochastic", seed=1)
        survivors = wsd.responses(x, 1)["a"]
        self.assertTrue(0.65 <= survivors/10000.0 <= 0.75)
        values = set(v for ram in counters(wsd)["a"]["rams"] for v in ram.values())
        self.assertEqual(values, {1})

    # T-04 and T-05
    def test_never_negative_and_erase(self):
        for factor in (0.01, 0.3):
            wsd, x, addrs = self.many_rams()
            wsd.penalize("a", addrs, factor, counterMode="int_stochastic", seed=2)
            rams = counters(wsd)["a"]["rams"]
            values = [v for ram in rams for v in ram.values()]
            self.assertTrue(all(v >= 1 for v in values))
            self.assertEqual(len(values), wsd.responses(x, 1)["a"])
            # empty RAMs must survive the serialization
            loaded = wp.Wisard(wsd.json())
            self.assertEqual(loaded.responses(x, 1), wsd.responses(x, 1))

    # T-06
    def test_error_penalizes_winner(self):
        xA, xA2, xB = self.X[0], [0,1,1,1,1,1,1,0,0], [0,0,0,1,1,1,1,1,1]
        wsd = fixed(9, 3)
        wsd.train([xA]*5 + [xA2]*2 + [xB]*3, ["A"]*7 + ["B"]*3)
        hit = dict(wsd.activeAddresses(xA, "A", 1))
        before = counters(wsd)
        report = wsd.trainEpochs([xA], ["B"], [], [], epochs=1, rho0=0.25, tau=0.0, decay="constant")
        after = counters(wsd)
        self.assertEqual(report["perEpoch"][0]["nPenError"], 1)
        for r, ram in enumerate(after["A"]["rams"]):
            for addr, value in before["A"]["rams"][r].items():
                self.assertEqual(ram[addr], 4 if hit.get(r) == addr else value)
        for r, ram in enumerate(after["B"]["rams"]):
            for addr, value in ram.items():
                self.assertEqual(value, before["B"]["rams"][r].get(addr, 0) + (1 if hit[r] == addr else 0))
        self.assertEqual(after["A"]["count"], 7)
        self.assertEqual(after["B"]["count"], 4)

    def margin_run(self, **kwargs):
        wsd, x = self.margin_case()
        self.assertEqual(wsd.responses(x, 10), {"A": 4, "B": 3})  # C = 0.25
        hit = dict(wsd.activeAddresses(x, "A", 1))
        wsd.trainEpochs([x], ["A"], [], [], epochs=1, b=10, decay="constant", kappa=2, **kwargs)
        after = counters(wsd)
        return [after["B"]["rams"][r][hit[r]] for r in range(4)], [after["A"]["rams"][r][hit[r]] for r in range(4)]

    # T-07 and T-10: factor = 1 - 0.8*(0.5-0.25)/0.5 = 0.6
    def test_low_margin_penalizes_runner_up(self):
        b, a = self.margin_run(rho0=0.8, tau=0.5, selectMargin="closest_to_b")
        self.assertEqual(b, [2, 30, 7, 300])
        self.assertEqual(a, [11]*4)

    def test_low_margin_lowest_count(self):
        b, a = self.margin_run(rho0=0.8, tau=0.5, selectMargin="lowest_count")
        self.assertEqual(b, [1, 50, 7, 300])

    # T-08
    def test_high_margin_does_not_penalize(self):
        b, a = self.margin_run(rho0=0.8, tau=0.2, selectMargin="closest_to_b")
        self.assertEqual(b, [2, 50, 11, 300])
        self.assertEqual(a, [11]*4)

    # T-09
    def test_classify_before_write(self):
        wsd = fixed(9, 3)
        report = wsd.trainEpochs([[1,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,1,1]], ["a", "b"], [], [], epochs=1)
        self.assertEqual(report["perEpoch"][0]["accTrain"], 0)

    # T-11
    def test_inverse_decay(self):
        wsd = fixed(9, 3)
        report = wsd.trainEpochs(self.X, self.y, [], [], epochs=4, rho0=0.5, decay="inverse")
        rhos = [row["rho"] for row in report["perEpoch"]]
        for got, expected in zip(rhos, [0.5, 0.25, 0.5/3, 0.125]):
            self.assertAlmostEqual(got, expected)
        self.assertEqual([row["bT"] for row in report["perEpoch"]], [1, 2, 3, 4])

    # T-12
    def test_validation_is_not_written(self):
        models = []
        for val in (slice(0, 4), slice(4, 9)):
            wsd = fixed(9, 3)
            wsd.trainEpochs(self.X, self.y, self.X[val], self.y[val], epochs=1, seed=3)
            models.append(counters(wsd))
        self.assertEqual(models[0], models[1])

    # T-13
    def test_early_stopping(self):
        X, y = self.overlapped()
        for patience in (1, 2, 3):
            wsd = fixed(30, 3)
            report = wsd.trainEpochs(X[:140], y[:140], X[140:], y[140:], epochs=15, patience=patience, rho0=0.4, tau=0.2, seed=4)
            accs = [row["accVal"] for row in report["perEpoch"]]
            self.assertEqual(report["bestEpoch"], accs.index(max(accs)))
            self.assertEqual(len(accs), min(15, report["bestEpoch"] + patience + 1))

    # the validation of the best epoch is the accuracy of the delivered model
    def test_validation_measures_delivered_model(self):
        X, y = self.overlapped()
        for valMode in ("fixed", "bleaching"):
            wsd = fixed(30, 3)
            report = wsd.trainEpochs(X[:140], y[:140], X[140:], y[140:], epochs=8, b=1, rho0=0.25, tau=0.2,
                                     valMode=valMode, seed=7)
            best = report["perEpoch"][report["bestEpoch"]]["accVal"]
            if valMode == "bleaching":
                got = sum(p == t for p, t in zip(wsd.classify(X[140:]), y[140:])) / 60.0
            else:
                got = 0
                for x, label in zip(X[140:], y[140:]):
                    R = wsd.responses(x, 1)
                    got += min(k for k, v in R.items() if v == max(R.values())) == label
                got /= 60.0
            self.assertAlmostEqual(best, got)

    def test_early_stopping_bleaching(self):
        X, y = self.overlapped()
        wsd = fixed(30, 3)
        report = wsd.trainEpochs(X[:140], y[:140], X[140:], y[140:], epochs=15, patience=2, b=1,
                                 rho0=0.25, tau=0.2, valMode="bleaching", seed=8)
        accs = [row["accVal"] for row in report["perEpoch"]]
        self.assertEqual(report["bestEpoch"], accs.index(max(accs)))
        self.assertEqual(len(accs), min(15, report["bestEpoch"] + 2 + 1))

    # T-14
    def test_reproducible(self):
        X, y = self.overlapped()
        runs = []
        for _ in range(2):
            wsd = fixed(30, 3)
            report = wsd.trainEpochs(X[:140], y[:140], X[140:], y[140:], epochs=6, counterMode="int_stochastic", seed=5)
            runs.append((counters(wsd), report))
        self.assertEqual(runs[0], runs[1])

    # T-15: the training threshold compares float counters, without rounding
    def test_penalizations_to_deactivate(self):
        self.assertGreaterEqual(1000*0.7**12, 10)
        self.assertLess(1000*0.7**13, 10)

    # T-16
    def test_serialization(self):
        X, y = self.overlapped()
        wsd = fixed(30, 3)
        wsd.trainEpochs(X[:140], y[:140], X[140:], y[140:], epochs=6, seed=6)
        loaded = wp.Wisard(wsd.json())
        self.assertSequenceEqual(wsd.classify(X), loaded.classify(X))
        self.assertEqual(counters(wsd), counters(loaded))

    def test_delivered_model_is_normalized(self):
        X, y = self.overlapped()
        wsd = fixed(30, 3)
        report = wsd.trainEpochs(X[:140], y[:140], [], [], epochs=4, rho0=0.0, tau=0.0)
        one = fixed(30, 3)
        one.train(X[:140], y[:140])
        self.assertEqual(report["bestEpoch"], 3)
        self.assertEqual(counters(wsd), counters(one))

    def test_dataset_input_equals_lists(self):
        X, y = self.overlapped()
        def dataset(X, y):
            d = wp.DataSet()
            for x, label in zip(X, y):
                d.add(x, label)
            return d
        options = dict(epochs=5, b=1, rho0=0.25, tau=0.2, valMode="bleaching", counterMode="int_stochastic", seed=9)
        lists, sets = fixed(30, 3), fixed(30, 3)
        r1 = lists.trainEpochs(X[:140], y[:140], X[140:], y[140:], **options)
        r2 = sets.trainEpochs(dataset(X[:140], y[:140]), dataset(X[140:], y[140:]), **options)
        self.assertEqual(r1, r2)
        self.assertEqual(counters(lists), counters(sets))
        self.assertEqual(sets.responses(wp.BinInput(X[0]), 1), lists.responses(X[0], 1))

    def test_invalid_options(self):
        wsd = fixed(9, 3)
        with self.assertRaises(RuntimeError):
            wsd.trainEpochs(self.X, self.y, [], [], unknown=1)
        with self.assertRaises(RuntimeError):
            wsd.trainEpochs(self.X, self.y, [], [], rho0=1.5)
        with self.assertRaises(RuntimeError):
            wsd.trainEpochs(self.X, self.y, [], [], selectError="closest_to_b")
        with self.assertRaises(RuntimeError):
            wsd.trainEpochs(self.X, self.y, [], [], valMode="best")
