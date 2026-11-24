"""
Binarization of the raw features, fitted on the training part only.

  thermometer:  `bits` thresholds per feature, evenly spaced between the training min and max;
  distributive: `bits` thresholds per feature at the training quantiles k/(bits+1), so every
                level holds about the same number of training examples.

Bit k of a feature is 1 when value > threshold k. Nominal features are one-hot encoded with
the categories seen in training, whatever the number of bits.

With binary_as_bit=True (the multi-bleaching benchmark), the kind of a feature also depends on how
many values it has in training: a feature with two values, nominal or numeric, is ONE bit (1 for
the larger value); a feature with one value gives no bit; the thermometer is only for numeric
features with three or more values and the one-hot only for nominal ones with three or more
categories. The default (False) keeps the encoding of the epoch benchmark. The output is packed (8 bits per
byte, most significant first), the layout BinInput uses, so it goes into wisardpkg without
unpacking.
"""
import base64

import numpy as np

import wisardpkg as wp

ENCODINGS = ("thermometer", "distributive")


class Encoder(object):
    def __init__(self, kind, bits, binary_as_bit=False):
        if kind not in ENCODINGS:
            raise ValueError("unknown encoding %s" % kind)
        self.kind, self.bits, self.binary_as_bit = kind, bits, binary_as_bit

    def fit(self, X, nominal):
        self.binary, self.low = np.zeros(0, dtype=int), np.zeros(0)
        if self.binary_as_bit:
            values = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
            self.binary = np.where(values == 2)[0]
            self.low = X[:, self.binary].astype(np.float64).min(axis=0)  # the bit is 1 above the smaller value
            nominal = nominal & (values >= 3)
            self.numeric = np.where(~nominal & (values >= 3))[0]
        else:
            self.numeric = np.where(~nominal)[0]
        self.nominal = np.where(nominal)[0]
        levels = np.arange(1, self.bits + 1) / float(self.bits + 1)
        thresholds = np.empty((len(self.numeric), self.bits))
        for start in range(0, len(self.numeric), 256):  # by column blocks, to bound the memory
            block = X[:, self.numeric[start:start + 256]].astype(np.float64)
            if self.kind == "thermometer":
                low, high = block.min(axis=0), block.max(axis=0)
                thresholds[start:start + 256] = low[:, None] + (high - low)[:, None] * levels[None, :]
            else:
                thresholds[start:start + 256] = np.quantile(block, levels, axis=0).T
        self.thresholds = thresholds
        self.categories = [np.unique(X[:, j]) for j in self.nominal]
        return self

    @property
    def size(self):
        return self.thresholds.size + len(self.binary) + sum(len(c) for c in self.categories)

    def transform(self, X, chunk=2048):
        """packed bits, shape (n, ceil(size/8))"""
        out = np.empty((len(X), (self.size + 7) // 8), dtype=np.uint8)
        for start in range(0, len(X), chunk):
            rows = X[start:start + chunk]
            parts = [(rows[:, self.numeric].astype(np.float64)[:, :, None] > self.thresholds[None, :, :])
                     .reshape(len(rows), -1)]
            if len(self.binary):
                parts.append(rows[:, self.binary].astype(np.float64) > self.low[None, :])
            for j, categories in zip(self.nominal, self.categories):
                parts.append(rows[:, j][:, None] == categories[None, :])
            out[start:start + chunk] = np.packbits(np.concatenate(parts, axis=1), axis=1)
        return out


def to_dataset(packed, labels, size):
    """wisardpkg.DataSet from packed rows (BinInput's base64 form: a byte with the number of
    padding bits, then the packed bits)"""
    padding = bytes([(8 - size % 8) % 8])
    data = wp.DataSet()
    for row, label in zip(packed, labels):
        data.add(wp.BinInput(base64.b64encode(padding + row.tobytes()).decode("ascii")), str(label))
    return data
