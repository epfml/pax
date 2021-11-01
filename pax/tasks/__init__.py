from typing import List
import importlib

import pax.tasks.registry as registry
from pax.tasks.tasks.api import Task


def list() -> List[str]:
    return registry.task.list()


def list_datasets() -> List[str]:
    return registry.dataset.list()


def list_models() -> List[str]:
    return registry.model.list()


def get(task_name: str) -> Task:
    return registry.task(task_name)


def get_dataset(dataset_name: str) -> Task:
    return registry.dataset(dataset_name)


def get_model(model_name: str) -> Task:
    return registry.model(model_name)


def try_import(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        pass

try_import("pax.tasks.datasets.deepobs")
try_import("pax.tasks.models.deepobs")
try_import("pax.tasks.models.torchvision")
try_import("pax.tasks.models.timm")
try_import("pax.tasks.tasks.deepobs")
