import torch
import pax.tasks.registry as registry
from pax.tasks.datasets.utils import PyTorchDataset
from torch.utils.data.dataset import Subset

import deepobs.pytorch.datasets as datasets

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepObsDataset:
    train: PyTorchDataset
    test: PyTorchDataset
    train_eval: PyTorchDataset

    def __init__(self, dataset, device):
        self._dataset = dataset(batch_size=1)

        self.train = PyTorchDataset(
            _extract_dataset(self._dataset._train_dataloader),
            device,
            iterator_defaults={
                "shuffle": True,
                "drop_last": self._dataset._train_dataloader.drop_last,
            },
        )
        self.valid = PyTorchDataset(
            _extract_dataset(self._dataset._valid_dataloader),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": self._dataset._valid_dataloader.drop_last,
            },
        )
        self.test = PyTorchDataset(
            _extract_dataset(self._dataset._test_dataloader),
            device,
            iterator_defaults={
                "shuffle": False,
                "drop_last": self._dataset._test_dataloader.drop_last,
            },
        )

        del self._dataset._train_dataloader
        del self._dataset._train_eval_dataloader
        del self._dataset._valid_dataloader
        del self._dataset._test_dataloader


def _extract_dataset(deepobs_loader):
    dataset = deepobs_loader.dataset
    sampler = deepobs_loader.sampler
    if isinstance(sampler, torch.utils.data.sampler.SubsetRandomSampler):
        return Subset(dataset, sampler.indices)
    elif isinstance(sampler, torch.utils.data.sampler.SequentialSampler):
        return dataset
    elif isinstance(sampler, torch.utils.data.sampler.RandomSampler):
        return dataset
    else:
        raise ValueError(f"Unkown sampler type {type(sampler)}")


def _wrap(dataset):
    def f(device: torch.DeviceObjType = DEFAULT_DEVICE):
        return DeepObsDataset(dataset, device)

    return f


registry.dataset.register("deepobs.cifar10", _wrap(datasets.cifar10))
registry.dataset.register("deepobs.cifar100", _wrap(datasets.cifar100))
registry.dataset.register("deepobs.fmnist", _wrap(datasets.fmnist))
registry.dataset.register("deepobs.mnist", _wrap(datasets.mnist))
registry.dataset.register("deepobs.quadratic", _wrap(datasets.quadratic))
registry.dataset.register("deepobs.svhn", _wrap(datasets.svhn))
registry.dataset.register("deepobs.tolstoi", _wrap(datasets.tolstoi))
# registry.dataset.register("deepobs.imagenet", _wrap(datasets.imagenet))
