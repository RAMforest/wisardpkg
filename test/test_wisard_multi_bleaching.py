from unittest import TestCase
import json
import math
import wisardpkg as wp

from test_wisard_bleaching import INFO, configured, random_model, with_counters, dab, iterative

OUT = dict(INFO, returnClassesDegrees=True)
BASES = ("iterative", "dab", "bbleaching")


# ---------------------------------------------------------------- reference implementations
# votes: {label: [counter read in each RAM]}, as Wisard.votes returns; labels in std::map order.
# Every base returns (candidates at the deciding threshold, that threshold, vote counts made).

def count(votes, b):
    return {k: sum(v >= b for v in votes[k]) for k in sorted(votes)}


def top2(values):
    """(label, first, second); ties go to the last label"""
    label, v1, v2 = "", 0, 0
    for k in sorted(values):
        if values[k] >= v1:
            label, v1, v2 = k, values[k], v1
        elif values[k] > v2:
            v2 = values[k]
    return label, v1, v2


def confidence(candidates):
    """(r1 - r2)/r1, 0 on ties and without votes"""
    _, r1, r2 = top2(candidates)
    return (r1 - r2) / r1 if r1 else 0.0


def iterative_from(votes, start, margin=1):
    b, steps = start, 0
    while True:
        threshold, steps = b, steps + 1
        labels = count(votes, b)
        b += 1
        biggest, ambiguity = 0, False
        for k in sorted(labels):
            if labels[k] > biggest:
                biggest, ambiguity = labels[k], False
            elif biggest - labels[k] < margin:
                ambiguity = True
        if not (ambiguity and biggest > 1):
            return labels, threshold, steps


def dab_from(votes, start, tol=0.01, inclusive=True):
    keep, previous, step = start, None, 0
    while True:
        step += 1
        counts = count(votes, keep)
        _, r1, r2 = top2(counts)
        if r1 == 0:
            return (counts, keep, 1) if step == 1 else (previous[0], previous[1], step)
        if (r1 - r2) / r1 >= tol:
            return counts, keep, step
        tied = [k for k in votes if counts[k] > 0 and (r1 - counts[k]) / r1 < tol]
        tau_b = max(min(v for v in votes[k] if v >= keep) for k in tied)
        nxt = tau_b + 1 if inclusive else tau_b
        if nxt <= keep:
            return counts, keep, step
        previous, keep = (counts, keep), nxt


def bbleaching_from(votes, start, tie=0.0, reference_cut=True):
    """binary search with the geometric mean on [start, max(B_k - 1, start)]"""
    lo = start
    hi = max(max(max(v) if v else 0 for v in votes.values()) - 1, 1, start)
    accept_lo, require = start == 1, start == 1 or reference_cut
    best, steps = (count(votes, lo), lo), 1
    reference = top2(best[0])[1]
    if accept_lo:
        if confidence(best[0]) > tie:
            return best[0], best[1], steps
        lo += 1
    while lo <= hi:
        mid = min(hi, max(lo, int(math.floor(math.sqrt(lo * hi)))))
        probe, steps = count(votes, mid), steps + 1
        tied = confidence(probe) <= tie
        if not tied and (not require or top2(probe)[1] == reference):
            return probe, mid, steps
        if confidence(probe) > confidence(best[0]):
            best = (probe, mid)
        if tied:
            lo = mid + 1
        else:
            hi = mid - 1
    return best[0], best[1], steps


def base_from(mode, **options):
    if mode == "iterative":
        return lambda votes, start: iterative_from(votes, start)
    if mode == "dab":
        return lambda votes, start: dab_from(votes, start, options.get("confidenceTolerance", 0.01),
                                             options.get("dabRemoval", "inclusive") == "inclusive")
    return lambda votes, start: bbleaching_from(votes, start, options.get("tieTolerance", 0.0), options.get("nextCut", "ref") == "ref")


def multi(votes, base, threshold=0.1, max_rounds=64):
    scores = {k: 0 for k in votes}
    last = max(max(v) if v else 0 for v in votes.values())
    start, steps, cuts, confidences = 1, 0, [], []
    for rnd in range(max_rounds):
        if rnd > 0 and start > last:
            break
        candidates, cut, n = base(votes, start)
        for k, v in candidates.items():
            scores[k] += v
        steps += n
        cuts.append(cut)
        confidences.append(confidence(candidates))
        if confidences[-1] > threshold:
            break
        start = cut + 1
    return dict(label=top2(scores)[0], confidence=confidence(scores), steps=steps, threshold=cuts[-1],
                thresholds=cuts, confidences=confidences, scores=scores)


def degrees(scores):
    total = float(sum(scores.values())) or 1.0
    return {k: v / total for k, v in scores.items()}


def six_rams(spec, **options):
    """6 RAMs (address 2, fixed mapping); on x = [0]*12 the counters of label are spec[label]"""
    X, y = [], []
    for label, counters in spec.items():
        for ram, k in enumerate(counters):
            sample = [1] * 12
            sample[2 * ram] = sample[2 * ram + 1] = 0
            X += [sample] * k
            y += [label] * k
        if not any(counters):
            X, y = X + [[1] * 12], y + [label]
    wsd = wp.Wisard(2, indexes=list(range(12)), **options)
    wsd.train(X, y)
    return wsd, [0] * 12


# the example of the RFC: iterative ties at b = 1 and b = 2 and decides A at b = 3
RFC = {"A": [6, 5, 3, 2, 2, 1], "B": [9, 8, 2, 2, 2, 1], "C": [3, 1, 1, 0, 0, 0]}


class WisardMultiBleachingTestCase(TestCase):

    def assertDecision(self, out, expected):
        self.assertEqual(out["class"], expected["label"])
        self.assertEqual(out["steps"], expected["steps"])
        self.assertEqual(out["finalThreshold"], expected["threshold"])
        self.assertEqual(out["thresholds"], expected["thresholds"])
        self.assertAlmostEqual(out["confidence"], expected["confidence"], places=6)
        for got, want in zip(out["confidences"], expected["confidences"]):
            self.assertAlmostEqual(got, want, places=12)
        want = degrees(expected["scores"])
        for row in out["classesDegrees"]:
            self.assertAlmostEqual(row["degree"], want[row["class"]], places=6)

    def models(self):
        """small counters (20 examples per class) and large ones (400 per class)"""
        for seed in range(3):
            yield random_model(seed)
            yield random_model(seed, address=2, n=1200, entry=32)

    def base_options(self):
        yield "iterative", dict()
        for removal in ("inclusive", "exclusive"):
            for tol in (0.01, 0.2):
                yield "dab", dict(confidenceTolerance=tol, dabRemoval=removal)
        for tie in (0.0, 0.1):
            for cut in ("ref", "untie"):
                yield "bbleaching", dict(tieTolerance=tie, nextCut=cut)

    # ------------------------------------------------------------ bbleaching alone
    def test_bbleaching_matches_reference(self):
        for wsd, X in self.models():
            for tie in (0.0, 0.05, 0.2):
                outs = configured(wsd, bleaching="bbleaching", tieTolerance=tie, **OUT).classify(X)
                for x, out in zip(X, outs):
                    candidates, cut, steps = bbleaching_from(wsd.votes(x), 1, tie)
                    self.assertDecision(out, dict(label=top2(candidates)[0], confidence=confidence(candidates), steps=steps,
                                                  threshold=cut, thresholds=[cut], confidences=[confidence(candidates)], scores=candidates))

    # ------------------------------------------------------------ the layer over every base
    def test_multi_matches_reference(self):
        several = {mode: 0 for mode in BASES}
        for wsd, X in self.models():
            votes = [wsd.votes(x) for x in X]
            for mode, options in self.base_options():
                for threshold in (0.0, 0.05, 0.3, 1.0):
                    for rounds in (64, 3):
                        outs = configured(wsd, bleaching=mode, multiBleaching=True, confidenceThreshold=threshold,
                                          maxRounds=rounds, **dict(OUT, **options)).classify(X)
                        for v, out in zip(votes, outs):
                            self.assertDecision(out, multi(v, base_from(mode, **options), threshold, rounds))
                            several[mode] += len(out["thresholds"]) > 1
        for mode in BASES:
            self.assertTrue(several[mode], mode)  # otherwise only the first round was exercised

    # round 1 is the base as it is; with one round the decision is the one of the base
    def test_first_round_is_the_base(self):
        one = {mode: 0 for mode in BASES}
        for wsd, X in self.models():
            for mode in BASES:
                base = configured(wsd, bleaching=mode, **OUT).classify(X)
                many = configured(wsd, bleaching=mode, multiBleaching=True, confidenceThreshold=0.2, **OUT).classify(X)
                for b, m in zip(base, many):
                    self.assertEqual(b["finalThreshold"], m["thresholds"][0])
                    if len(m["thresholds"]) == 1:
                        one[mode] += 1
                        self.assertEqual((b["class"], b["steps"], b["classesDegrees"]), (m["class"], m["steps"], m["classesDegrees"]))
                    else:
                        self.assertLessEqual(m["confidences"][0], 0.2)
        for mode in BASES:
            self.assertTrue(one[mode], mode)  # otherwise the single-round case was not exercised

    # the base references of this file, from threshold 1, are the references of test_wisard_bleaching
    def test_base_references_from_one(self):
        for wsd, X in self.models():
            for x in X:
                votes = wsd.votes(x)
                want = iterative(votes)
                self.assertEqual(iterative_from(votes, 1)[1:], (want["threshold"], want["steps"]))
                for tol in (0.01, 0.2):
                    for inclusive in (True, False):
                        want = dab(votes, tol=tol, inclusive=inclusive)
                        self.assertEqual(dab_from(votes, 1, tol, inclusive)[1:], (want["threshold"], want["steps"]))

    def test_rounds_grow_and_stop_at_the_threshold(self):
        for wsd, X in self.models():
            for mode in BASES:
                for threshold in (0.1, 0.5):
                    outs = configured(wsd, bleaching=mode, multiBleaching=True, confidenceThreshold=threshold, **OUT).classify(X)
                    for x, out in zip(X, outs):
                        cuts, confs = out["thresholds"], out["confidences"]
                        self.assertEqual(cuts, sorted(set(cuts)))
                        self.assertTrue(all(c <= threshold for c in confs[:-1]))  # only the last round may pass it
                        if confs[-1] <= threshold and len(cuts) < 64:  # stopped for lack of counters, not for confidence or maxRounds
                            self.assertGreaterEqual(cuts[-1], max(max(v) for v in wsd.votes(x).values()))

    # the score is the sum of the votes at the deciding thresholds
    def test_score_is_the_sum_over_the_rounds(self):
        for wsd, X in self.models():
            for mode in BASES:
                outs = configured(wsd, bleaching=mode, multiBleaching=True, confidenceThreshold=1.0, **OUT).classify(X)
                for x, out in zip(X, outs):
                    votes = wsd.votes(x)
                    want = degrees({k: sum(sum(v >= b for b in out["thresholds"]) for v in vs) for k, vs in votes.items()})
                    for row in out["classesDegrees"]:
                        self.assertAlmostEqual(row["degree"], want[row["class"]], places=6)

    # ------------------------------------------------------------ by hand
    def test_rfc_example(self):
        wsd, x = six_rams(RFC, **OUT)
        out = wsd.classify([x])[0]
        self.assertEqual((out["class"], out["finalThreshold"], out["steps"]), ("A", 3, 3))
        self.assertEqual(wsd.responses(x, 3), {"A": 3, "B": 2, "C": 1})

        wsd.setBleaching(multiBleaching=True, confidenceThreshold=0.5)
        out = wsd.classify([x])[0]
        self.assertEqual((out["class"], out["thresholds"], out["steps"]), ("B", [3, 6, 7], 7))
        for got, want in zip(out["confidences"], (1 / 3.0, 0.5, 1.0)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual({r["class"]: round(r["degree"] * 11) for r in out["classesDegrees"]}, {"A": 4, "B": 6, "C": 1})

        wsd.setBleaching(confidenceThreshold=0.3)  # round 1 is already above it: the base decision
        out = wsd.classify([x])[0]
        self.assertEqual((out["class"], out["thresholds"], out["steps"]), ("A", [3], 3))

    # summing the visited thresholds of the iterative search (1, 2, 3) would not change it: the
    # leaders are tied before the decision. The rounds after the decision are what changes it.
    def test_rfc_example_other_bases(self):
        wsd, x = six_rams(RFC, bleaching="dab", **OUT)
        votes = wsd.votes(x)
        for mode, options in self.base_options():
            for threshold in (0.2, 0.5, 1.0):
                wsd.setBleaching(bleaching=mode, multiBleaching=True, confidenceThreshold=threshold, **options)
                self.assertDecision(wsd.classify([x])[0], multi(votes, base_from(mode, **options), threshold))

    # the class with few examples wins one round and loses the sum: its counters are small, so it
    # only votes at the low thresholds
    def test_summing_rounds_favours_large_counters(self):
        for mode in BASES:
            wsd, x = with_counters({"big": ([40, 40, 40, 0], 0), "small": ([2, 2, 2, 2], 0)}, bleaching=mode, **OUT)
            self.assertEqual(wsd.classify([x])[0]["class"], "small")
            wsd.setBleaching(multiBleaching=True, confidenceThreshold=1.0)
            out = wsd.classify([x])[0]
            self.assertDecision(out, multi(wsd.votes(x), base_from(mode), 1.0))
            self.assertEqual(out["class"], "big")

    # ------------------------------------------------------------ options
    def test_set_bleaching_is_the_configured_model(self):
        wsd, X = random_model(1, address=2, n=1200, entry=32)
        model = configured(wsd, **OUT)
        for options in (dict(bleaching="bbleaching", multiBleaching=True, confidenceThreshold=0.3, nextCut="untie"),
                        dict(bleaching="dab", multiBleaching=False, confidenceTolerance=0.2), dict(bleaching="iterative", multiBleaching=False),
                        dict(bleaching="iterative", multiBleaching=True, confidenceThreshold=0.5, maxRounds=4)):
            model.setBleaching(**options)
            self.assertEqual(model.classify(X), configured(wsd, **dict(OUT, **options)).classify(X))
        before = model.classify(X)
        for options in (dict(bleaching="other"), dict(bleaching="rdab"), dict(nextCut="other"), dict(confidenceThreshold=2.0), dict(unknown=1)):
            with self.assertRaises(Exception):
                model.setBleaching(**options)
            self.assertEqual(model.classify(X), before)  # a rejected change leaves the model as it was

    def test_options_json_and_errors(self):
        wsd, X = random_model(0)
        model = configured(wsd, bleaching="dab", multiBleaching=True, confidenceThreshold=0.25, maxRounds=7)
        config = json.loads(model.json())
        self.assertEqual((config["bleaching"], config["multiBleaching"], config["confidenceThreshold"], config["maxRounds"]),
                         ("dab", True, 0.25, 7))
        self.assertEqual(wp.Wisard(model.json()).classify(X), model.classify(X))
        self.assertFalse(json.loads(wp.Wisard(3).json())["multiBleaching"])
        for options in (dict(nextCut="other"), dict(confidenceThreshold=1.5), dict(tieTolerance=1.0), dict(maxRounds=0),
                        dict(bleaching="rdab", multiBleaching=True), dict(bleaching="multi")):
            with self.assertRaises(Exception):
                wp.Wisard(3, **options)

    def test_no_bleaching_and_unseen_input(self):
        wsd, X = random_model(0)
        plain = configured(wsd, bleachingActivated=False).classify(X)
        for mode in BASES:
            self.assertEqual(configured(wsd, bleaching=mode, multiBleaching=True, bleachingActivated=False).classify(X), plain)
            unseen, x = six_rams({"A": [0] * 6, "B": [0] * 6}, bleaching=mode, multiBleaching=True, **OUT)
            out = unseen.classify([x])[0]  # nothing votes: one round, no confidence
            self.assertEqual((out["thresholds"], out["confidence"], out["confidences"]), ([1], 0.0, [0.0]))

    # valMode="bleaching" validates every epoch with the policy of the model, the layer included
    def test_epoch_validation_uses_the_policy(self):
        _, X = random_model(2)
        y = ["abc"[i % 3] for i in range(len(X))]
        for mode in BASES:
            model = wp.Wisard(3, indexes=list(range(24)), bleaching=mode, multiBleaching=True, confidenceThreshold=0.3)
            report = model.trainEpochs(X[:90], y[:90], X[90:], y[90:], b=1, rho0=0.5, tau=0.1, epochs=1, valMode="bleaching", seed=0)
            acc = sum(p == t for p, t in zip(model.classify(X[90:]), y[90:])) / 60
            self.assertAlmostEqual(report["perEpoch"][0]["accVal"], acc, places=12)
