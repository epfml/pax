from typing import Mapping

import pax.tasks.registry as registry
import regex as re
import torch
from pax.tasks.datasets.api import Batch
from pax.tasks.models.api import Buffers, Model, Params, Tuple
from pax.tasks.tasks.api import Task


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassificationTask(Task):
    def __init__(self, device=DEFAULT_DEVICE):
        self._device = device

    def init(self, seed: int = 0) -> Tuple[Params, Buffers]:
        return self.model.init(seed)

    @property
    def name(self) -> str:
        return type(self).__name__

    def loss(
        self,
        params: Params,
        batch: Batch,
        buffers: Buffers = None,
        is_training: bool = True,
    ) -> Tuple[float, Buffers]:
        prediction, buffers = self.model.forward(
            params, batch.x, buffers=buffers, is_training=is_training
        )
        loss = torch.nn.functional.cross_entropy(prediction, batch.y)

        regularization = 0
        for name, param in params.items():
            wd = self._weight_decay_for_param(name)
            if wd > 0:
                regularization += _squared_norm(param) * wd

        return loss + regularization, buffers

    def evaluate_batch(
        self,
        params: Params,
        batch: Batch,
        buffers: Buffers = None,
        is_training: bool = False,
    ) -> Mapping[str, torch.Tensor]:
        with torch.no_grad():
            output, buffers = self.model.forward(
                params, batch.x, buffers=buffers, is_training=is_training
            )
            loss = torch.nn.functional.cross_entropy(output, batch.y)
            predictions = torch.argmax(output, 1)
            accuracy = predictions.eq(batch.y).float().mean()
            predicted_probs = torch.nn.functional.softmax(output, dim=-1)
            soft_accuracy = predicted_probs[torch.arange(len(batch), device=predicted_probs.device), batch.y].mean()
            return {"loss": loss, "accuracy": accuracy, "soft_accuracy": soft_accuracy}

    def _weight_decay_for_param(self, param_name: str) -> float:
        return 0.0


class Cifar10(ClassificationTask):
    config = {
        "weight_decay": 1e-4,
        "eval_batch_size": 1000,
        "learning_rate": 0.05,
        "momentum": 0.9,
        "optimizer": "SGD",
    }

    def __init__(self, model: str = "resnet20", device=DEFAULT_DEVICE):
        data = registry.dataset("torchvision.cifar10")(device=device)
        self.train = data.train
        self.test = data.test
        self.model: Model = registry.model(model)(num_outputs=10, device=device)
        super().__init__(device)

    def _weight_decay_for_param(self, param_name: str) -> float:
        if _parameter_type(param_name) != "batch_norm":
            return self.config["weight_decay"]
        else:
            return 0.0


registry.task.register("cifar10", Cifar10)


class Cifar100(ClassificationTask):
    config = {
        "weight_decay": 1e-4,
        "eval_batch_size": 1000,
        "learning_rate": 0.05,
        "momentum": 0.9,
        "optimizer": "SGD",
    }

    def __init__(self, model: str = "resnet20", device=DEFAULT_DEVICE):
        data = registry.dataset("torchvision.cifar100")(device=device)
        self.train = data.train
        self.test = data.test
        self.model: Model = registry.model(model)(num_outputs=100, device=device)
        super().__init__(device)

    def _weight_decay_for_param(self, param_name: str) -> float:
        if _parameter_type(param_name) != "batch_norm":
            return self.config["weight_decay"]
        else:
            return 0.0


registry.task.register("cifar100", Cifar100)


def _squared_norm(tensor):
    return torch.sum(tensor ** 2)


def _parameter_type(parameter_name):
    if "conv" in parameter_name and "weight" in parameter_name:
        return "convolution"
    elif re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name):
        return "batch_norm"
    else:
        return "other"


class LogisticRegression(ClassificationTask):
    config = {"eval_batch_size": 1000}

    def __init__(self, dataset: str, weight_decay=0, num_classes=None, device=DEFAULT_DEVICE):
        data = registry.dataset(dataset)(device=device)
        self.train = data.train
        self.test = data.test
        self.weight_decay = weight_decay
        example_batch = next(iter(self.train.iterator(batch_size=1)))

        if num_classes is None:
            num_classes = max(data.train.num_classes, data.test.num_classes)

        self.model: Model = registry.model("linear")(in_features=example_batch.x.shape[-1], out_features=num_classes, device=device)

        super().__init__(device)

    def _weight_decay_for_param(self, param_name: str) -> float:
        return self.weight_decay


registry.task.register("logistic-regression", LogisticRegression)
