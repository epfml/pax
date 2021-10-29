from typing import List

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


import pax.tasks.datasets.deepobs
import pax.tasks.models.deepobs
import pax.tasks.models.timm
import pax.tasks.tasks.deepobs
