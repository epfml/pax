import itertools
import math
from typing import Iterable, List, Tuple

import torch
from pax.tasks.datasets.api import Batch, Dataset
from torch.utils.data import DataLoader


class SupervisedBatch(Batch):
    def __init__(self, x, y, progress):
        self.x = x
        self.y = y
        self.progress = progress

    def __len__(self) -> int:
        return len(self.x)

    def to(self, device):
        return SupervisedBatch(self.x.to(device), self.y.to(device), self.progress)


class PyTorchDataset(Dataset):
    def __init__(self, dataset, device, prepare_batch=None, iterator_defaults={}):
        self._set = dataset
        self._device = device
        self._iterator_defaults = {
            "shuffle": True,
            "drop_last": True,
            "num_workers": 1,
            "pin_memory": self._device.type == "cuda",
            **iterator_defaults,
        }
        if prepare_batch is not None:
            self.prepare_batch = prepare_batch

    def __len__(self):
        return len(self._set)

    def subsample(self, size, generator) -> "PyTorchDataset":
        indices = torch.randperm(len(self._set), generator=generator)[:size]
        dataset = torch.utils.data.Subset(self._set, indices)
        return PyTorchDataset(
            dataset,
            self._device,
            prepare_batch=self.prepare_batch,
            iterator_defaults=self._iterator_defaults
        )

    @property
    def num_classes(self):
        return len(self._set.classes)

    def prepare_batch(self, batch, progress):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 1:
            x = batch[0]
            y = torch.empty([], device=x.device)
        else:
            raise ValueError("Weird number of entries in the batch")
        return SupervisedBatch(x, y, progress).to(self._device)

    def iterator(
        self, batch_size: int = None, repeat = False, **iterator_args
    ) -> Iterable[Tuple[float, Batch]]:
        iterator_args = {**self._iterator_defaults, **iterator_args}
        if batch_size is not None:
            iterator_args["batch_size"] = batch_size
        if iterator_args["drop_last"]:
            nn = int(len(self) / iterator_args["batch_size"])
        else:
            nn = int(math.ceil(len(self) / iterator_args["batch_size"]))

        loader = DataLoader(self._set, **iterator_args)

        step = 0
        for _ in itertools.count() if repeat else [0]:
            for i, batch in enumerate(loader):
                epoch_fractional = float(step) / nn
                yield self.prepare_batch(batch, epoch_fractional)
                step += 1
