"""
Datasets of the epoch training benchmark. Every loader returns a Dataset with the raw
features (before any binarization), so the encoders can be fitted on the training part only.

Sources:
  - tabular datasets: OpenML (fetch_openml, cached in <dataDir>/openml);
  - MNIST and FashionMNIST: the original IDX files;
  - CIFAR-10: the binary version (cifar-10-binary.tar.gz), read straight from the archive.
    The "python" version is a pickle and is not used.
"""
import gzip
import os
import struct
import tarfile
import urllib.request

import numpy as np

# name -> OpenML data_id (version 1 unless noted)
OPENML = {
    "iris": 61,
    "phoneme": 1489,
    "skin-seg": 1502,
    "higgs": 23512,       # 98050 rows, the 9 rows with missing values are dropped
    "australian": 40981,
    "nomao": 1486,
    "segment": 958,       # binarized version of segment
    "miniboone": 41150,
    "christine": 41142,
    "jasmine": 41143,
    "sylvine": 41146,
    "blood": 1464,
    "jsc": 42468,         # hls4ml_lhc_jets_hlf, 830000 rows, 5 classes
}

IMAGES = ("mnist", "fashion-mnist", "cifar10")
NAMES = tuple(OPENML) + IMAGES

URLS = {
    "mnist": "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "fashion-mnist": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
}


class Dataset(object):
    """X: numeric matrix (n, features), float for tabular and uint8 for images; y: array of
    str labels; nominal: boolean mask of the
    nominal features (encoded one-hot); official test split in X_test / y_test when it exists."""

    def __init__(self, name, X, y, nominal=None, X_test=None, y_test=None):
        self.name = name
        self.X, self.y = X, np.asarray(y).astype(str)
        self.nominal = np.zeros(X.shape[1], dtype=bool) if nominal is None else nominal
        self.X_test = X_test
        self.y_test = None if y_test is None else np.asarray(y_test).astype(str)

    @property
    def has_official_test(self):
        return self.X_test is not None


def load(name, data_dir):
    if name in OPENML:
        return load_openml(name, data_dir)
    if name in ("mnist", "fashion-mnist"):
        return load_idx_images(name, data_dir)
    if name == "cifar10":
        return load_cifar10(data_dir)
    raise ValueError("unknown dataset %s (choose from %s)" % (name, ", ".join(NAMES)))


def load_openml(name, data_dir):
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=OPENML[name], as_frame=True, parser="auto",
                         data_home=os.path.join(data_dir, "openml"))
    frame, target = bunch.data, bunch.target
    keep = ~(frame.isna().any(axis=1) | target.isna()).to_numpy()
    frame, target = frame[keep], target[keep]
    nominal = np.array([str(t) in ("category", "object", "bool") for t in frame.dtypes])
    X = np.empty(frame.shape, dtype=np.float64)
    for j, column in enumerate(frame.columns):
        values = frame[column]
        # nominal values become category codes; the encoder turns them into one-hot bits
        X[:, j] = values.astype("category").cat.codes.to_numpy() if nominal[j] else values.to_numpy(dtype=np.float64)
    return Dataset(name, X, target.astype(str).to_numpy(), nominal)


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def read_idx(path):
    """IDX file (gzipped or not) as a numpy array of unsigned bytes"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    zero, dtype, ndim = struct.unpack(">HBB", data[:4])
    if zero != 0 or dtype != 0x08:
        raise ValueError("%s is not an unsigned byte IDX file" % path)
    shape = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_idx_images(name, data_dir):
    folder = os.path.join(data_dir, name)
    parts = {}
    for prefix in ("train", "t10k"):
        arrays = []
        for kind, dims in (("images", 3), ("labels", 1)):
            filename = "%s-%s-idx%d-ubyte.gz" % (prefix, kind, dims)
            arrays.append(read_idx(_download(URLS[name] + filename, os.path.join(folder, filename))))
        images, labels = arrays
        parts[prefix] = (images.reshape(len(images), -1), labels.astype(str))  # pixels stay uint8
    return Dataset(name, parts["train"][0], parts["train"][1], X_test=parts["t10k"][0], y_test=parts["t10k"][1])


def load_cifar10(data_dir):
    path = _download(URLS["cifar10"], os.path.join(data_dir, "cifar10", "cifar-10-binary.tar.gz"))
    record = 1 + 3072  # label byte + 32x32x3 pixels (channel planes)
    parts = {"train": [], "test": []}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():  # read in memory, nothing is extracted to disk
            base = os.path.basename(member.name)
            if not member.isfile() or not base.endswith(".bin") or base == "batches.meta.txt":
                continue
            raw = np.frombuffer(tar.extractfile(member).read(), dtype=np.uint8).reshape(-1, record)
            parts["test" if base == "test_batch.bin" else "train"].append(raw)
    train, test = np.concatenate(parts["train"]), np.concatenate(parts["test"])
    if len(train) != 50000 or len(test) != 10000:
        raise ValueError("unexpected CIFAR-10 sizes %d / %d" % (len(train), len(test)))
    return Dataset("cifar10", np.ascontiguousarray(train[:, 1:]), train[:, 0].astype(str),
                   X_test=np.ascontiguousarray(test[:, 1:]), y_test=test[:, 0].astype(str))
