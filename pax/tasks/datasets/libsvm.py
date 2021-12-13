# %%
import hashlib
import os
import urllib.request
from typing import NamedTuple

import numpy as np
import pax.tasks.registry as registry
import sklearn.datasets
import torch
from pax.tasks.datasets.utils import PyTorchDataset

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CACHE_DIR = os.getenv("LIBSVM_DATASET_DIR", os.path.join(os.getenv("HOME"), "libsvm"))


class LibSVMDataset(torch.utils.data.Dataset):
    def __init__(self, url, data_root=CACHE_DIR, download=False, md5=None, dimensionality=None, classes=None):
        self.url = url
        self.data_root = data_root
        self._dimensionality = dimensionality

        self.filename = os.path.basename(url)
        self.dataset_type = os.path.basename(os.path.dirname(url))

        if not os.path.isfile(self.local_filename):
            if download:
                print(f"Downloading {url}")
                self._download()
                if md5 is not None:  # verify the downloaded file
                    assert self.hash() == md5
            else:
                raise RuntimeError(
                    "Dataset not found or corrupted. You can use download=True to download it."
                )
        elif md5 is not None:
            assert self.hash() == md5
            print("Files already downloaded and verified")
        else:
            print("Files already downloaded")

        is_multilabel = self.dataset_type == "multilabel"
        self.data, y = sklearn.datasets.load_svmlight_file(
            self.local_filename, multilabel=is_multilabel
        )

        sparsity = self.data.nnz / (self.data.shape[0] * self.data.shape[1])
        if sparsity > 0.1:
            self.data = self.data.todense().astype(np.float32)
            self._is_sparse = False
        else:
            self._is_sparse = True

        # convert labels to [0, 1, ...]
        if classes is None:
            classes = np.unique(y)
        self.classes = np.sort(classes)
        self.targets = torch.zeros(len(y), dtype=torch.int64)
        for i, label in enumerate(self.classes):
            self.targets[y == label] = i

        self.class_to_idx = {cl: idx for idx, cl in enumerate(self.classes)}

        super().__init__()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_features(self):
        return self.data.shape[1]

    def hash(self):
        md5 = hashlib.md5()
        with open(self.local_filename, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)
        return md5.hexdigest()

    def __getitem__(self, idx):
        if self._is_sparse:
            x = torch.from_numpy(self.data[idx].todense().astype(np.float32)).flatten()
        else:
            x = torch.from_numpy(self.data[idx]).flatten()
        y = self.targets[idx]

        # We may have to pad with zeros
        if self._dimensionality is not None:
            if len(x) < self._dimensionality:
                x = torch.cat([x, torch.zeros([self._dimensionality - len(x)], dtype=x.dtype, device=x.device)])
            elif len(x) > self._dimensionality:
                raise RuntimeError("Dimensionality is set wrong.")

        return x, y


    def __len__(self):
        return len(self.targets)

    @property
    def local_filename(self):
        return os.path.join(self.data_root, self.dataset_type, self.filename)

    def _download(self):
        os.makedirs(os.path.dirname(self.local_filename), exist_ok=True)
        urllib.request.urlretrieve(self.url, filename=self.local_filename)


class IJCNN1(LibSVMDataset):
    def __init__(self, split, download=False, data_root=CACHE_DIR):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2"
            md5 = "9889c2e9d957dca5304ed2d285f1be6d"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2"
            md5 = "66433ab8089acee9e56dc61ac89a2fe2"
        elif split == "val":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.val.bz2"
            md5 = "9940e6f83e00623a5ca993f189ab18d9"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, md5=md5, download=download, data_root=data_root)


class CovTypeBinary(LibSVMDataset):
    def __init__(self, download=False, scale=True, data_root=CACHE_DIR):
        if scale:
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2"
            md5 = "d95f45e15c284005c2c7a4c82e4be102"
        else:
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2"
            md5 = "0d3439b314ce13e2f8b903b12bb3ea20"
        super().__init__(url=url, md5=md5, download=download, data_root=data_root)


class HIGGS(LibSVMDataset):
    def __init__(self, split, download=False, scale=True, data_root=CACHE_DIR):
        super().__init__(
            url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2",
            data_root=data_root,
            download=download,
            md5="be7a1089c595e1ffca4988c8549a871c",
        )
        if split == "train":
            self.data = self.data[:-500_000]
            self.targets = self.targets[:-500_000]
        elif split == "test":
            self.data = self.data[-500_000:]
            self.targets = self.targets[-500_000:]
        elif split == "all":
            pass
        else:
            raise RuntimeError(f"Unavailable split {split}")


class MNIST(LibSVMDataset):
    def __init__(self, split, scale=True, download=False, data_root=CACHE_DIR):
        if split == "train" and scale:
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2"
            md5 = "e8c48fc61df3cf808e7f35204431b376"
        elif split == "test" and scale:
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2"
            md5 = "56453a3c427a8b32632db2875abc891d"
        elif split == "train" and not scale:
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2"
            md5 = "1eca9ecabee216ae06b3e2f811f4cfd7"
        elif split == "test" and not scale:
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2"
            md5 = "a4aafe182113f147e3068d37760ece9d"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, md5=md5, download=download, data_root=data_root, dimensionality=28*28, classes=np.arange(10))


class RCV1MultiClass(LibSVMDataset):
    def __init__(self, split, download=False, data_root=CACHE_DIR):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/rcv1_train.multiclass.bz2"
            md5 = "b0ce08cd1a4c9e15c887c20acfb0eade"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/rcv1_test.multiclass.bz2"
            md5 = "68a377cfff6f4a82edac1975b148afd3"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, md5=md5, download=download, data_root=data_root, classes=np.arange(53))


class RCV1Binary(LibSVMDataset):
    def __init__(self, split, download=False, data_root=CACHE_DIR):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2"
            md5 = "1aeda848408e621468c0fe6944d9382f"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2"
            md5 = "d6e3ab397758fb5c036d9cced52aedae"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, md5=md5, download=download, data_root=data_root)


class TrainTest(NamedTuple):
    train: PyTorchDataset
    test: PyTorchDataset


class TrainValidTest(NamedTuple):
    train: PyTorchDataset
    valid: PyTorchDataset
    test: PyTorchDataset


def ijcnn1(device: torch.DeviceObjType = DEFAULT_DEVICE,):
    return TrainValidTest(
        train=PyTorchDataset(
            IJCNN1("train", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 1},
        ),
        valid=PyTorchDataset(
            IJCNN1("val", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": False,
                "batch_size": 1000,
            },
        ),
        test=PyTorchDataset(
            IJCNN1("test", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": False,
                "batch_size": 1000,
            },
        ),
    )


registry.dataset.register("libsvm.ijcnn1", ijcnn1)


def covtype_binary(device: torch.DeviceObjType = DEFAULT_DEVICE):
    dataset = PyTorchDataset(
        CovTypeBinary(download=True, data_root=registry.config["data_root"]),
        device,
        iterator_defaults={"shuffle": True, "drop_last": True},
    )

    generator = torch.Generator()
    generator.manual_seed(0)

    num_train = int(len(dataset) * 2 / 3)
    num_test = len(dataset) - num_train

    train, test = torch.utils.data.random_split(dataset, [num_train, num_test], generator)

    return TrainTest(
        train=PyTorchDataset(
            train,
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 1},
        ),
        test=PyTorchDataset(
            test,
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": False,
                "batch_size": 1000,
            },
        ),
    )


registry.dataset.register("libsvm.covtype-binary", covtype_binary)


def rcv1_binary(device: torch.DeviceObjType = DEFAULT_DEVICE,):
    return TrainTest(
        train=PyTorchDataset(
            RCV1Binary("train", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 1},
        ),
        test=PyTorchDataset(
            RCV1Binary("test", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": False,
                "batch_size": 1000,
            },
        ),
    )


registry.dataset.register("libsvm.rcv1-binary", rcv1_binary)


def higgs(device: torch.DeviceObjType = DEFAULT_DEVICE,):
    return TrainTest(
        train=PyTorchDataset(
            HIGGS("train", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 1},
        ),
        test=PyTorchDataset(
            HIGGS("test", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": False,
                "batch_size": 1000,
            },
        ),
    )


registry.dataset.register("libsvm.higgs", higgs)


def mnist(device: torch.DeviceObjType = DEFAULT_DEVICE,):
    return TrainTest(
        train=PyTorchDataset(
            MNIST("train", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 1},
        ),
        test=PyTorchDataset(
            MNIST("test", download=True, data_root=registry.config["data_root"]),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": False,
                "batch_size": 1000,
            },
        ),
    )


registry.dataset.register("libsvm.mnist", mnist)


def rcv1_multiclass(device: torch.DeviceObjType = DEFAULT_DEVICE,):
    return TrainTest(
        train=PyTorchDataset(
            RCV1MultiClass(
                "train", download=True, data_root=registry.config["data_root"]
            ),
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 1},
        ),
        test=PyTorchDataset(
            RCV1MultiClass(
                "test", download=True, data_root=registry.config["data_root"]
            ),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": False,
                "batch_size": 1000,
            },
        ),
    )


registry.dataset.register("libsvm.rcv1-multiclass", rcv1_multiclass)
