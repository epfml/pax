from typing import NamedTuple

import pax.tasks.registry as registry
import torch
from pax.tasks.datasets.utils import PyTorchDataset

import torchvision

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainTest(NamedTuple):
    train: PyTorchDataset
    test: PyTorchDataset


class TrainValidTest(NamedTuple):
    train: PyTorchDataset
    valid: PyTorchDataset
    test: PyTorchDataset


cifar_transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

cifar_transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)


def cifar10(
    train_transform=cifar_transform_train,
    test_transform=cifar_transform_test,
    device: torch.DeviceObjType = DEFAULT_DEVICE,
):
    return TrainTest(
        train=PyTorchDataset(
            torchvision.datasets.CIFAR10(
                registry.config["data_root"],
                train=True,
                transform=train_transform,
                download=True,
            ),
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 128},
        ),
        test=PyTorchDataset(
            torchvision.datasets.CIFAR10(
                registry.config["data_root"],
                train=False,
                transform=test_transform,
                download=True,
            ),
            device,
            iterator_defaults={"shuffle": False, "drop_last": False, "batch_size": 1000},
        ),
    )


registry.dataset.register("torchvision.cifar10", cifar10)


def cifar100(
    train_transform=cifar_transform_train,
    test_transform=cifar_transform_test,
    device: torch.DeviceObjType = DEFAULT_DEVICE,
):
    return TrainTest(
        train=PyTorchDataset(
            torchvision.datasets.CIFAR100(
                registry.config["data_root"],
                train=True,
                transform=train_transform,
                download=True,
            ),
            device,
            iterator_defaults={"shuffle": True, "drop_last": True, "batch_size": 128},
        ),
        test=PyTorchDataset(
            torchvision.datasets.CIFAR100(
                registry.config["data_root"],
                train=False,
                transform=test_transform,
                download=True,
            ),
            device,
            iterator_defaults={"shuffle": False, "drop_last": False, "batch_size": 1000},
        ),
    )


registry.dataset.register("torchvision.cifar100", cifar100)
