import importlib
from typing import List

import pax.tasks.registry as registry
from pax.tasks.tasks.api import Task


def list() -> List[str]:
    return registry.task.list()


def list_datasets() -> List[str]:
    return registry.dataset.list()


def list_models() -> List[str]:
    return registry.model.list()


def get(task_name: str, *args, **kwargs) -> Task:
    return registry.task(task_name)(*args, **kwargs)


def get_dataset(dataset_name: str, *args, **kwargs) -> Task:
    return registry.dataset(dataset_name)(*args, **kwargs)


def get_model(model_name: str, *args, **kwargs) -> Task:
    return registry.model(model_name)(*args, **kwargs)


def configure(key, value):
    registry.config[key] = value


def list_config() -> dict:
    return {key: registry.config[key] for key in registry.config.list()}


def try_import(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        pass


try_import("pax.tasks.datasets.deepobs")
try_import("pax.tasks.datasets.torchvision")
try_import("pax.tasks.models.basic")
try_import("pax.tasks.models.deepobs")
try_import("pax.tasks.models.torchvision")
try_import("pax.tasks.models.timm")
try_import("pax.tasks.models.resnet20")
try_import("pax.tasks.tasks.deepobs")
try_import("pax.tasks.tasks.classification")
try_import("pax.tasks.datasets.libsvm")
