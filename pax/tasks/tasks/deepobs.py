from typing import Mapping

import numpy as np
import pax.tasks.registry as registry
import torch
from pax.tasks.datasets.api import Batch, Dataset
from pax.tasks.models.api import Buffers, Model, Params, Tuple
from pax.tasks.tasks.api import Task

from deepobs.pytorch.testproblems.quadratic_deep import random_rotation
from deepobs.pytorch.testproblems.testproblems_utils import \
    vae_loss_function_factory

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepObsTask(Task):
    def __init__(self, device=DEFAULT_DEVICE):
        self._device = device
        self.train = self._data.train
        self.test = self._data.test
        self.valid = self._data.valid

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
            accuracy = torch.argmax(output, 1).eq(batch.y).float().mean()
            return {"loss": loss, "accuracy": accuracy}

    def _weight_decay_for_param(self, param_name: str) -> float:
        return 0.0


class Cifar100_3C3D(DeepObsTask):
    config = {"weight_decay": 0.002, "eval_batch_size": 1000}

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.cifar100")(device)
        self.model: Model = registry.model("deepobs.cifar10_3c3d")(
            num_outputs=100, device=device
        )
        super().__init__(device)

    def _weight_decay_for_param(self, param_name: str) -> float:
        if "bias" in param_name:
            return 0.0
        else:
            return self.config["weight_decay"]


registry.task.register("deepobs.cifar100_3c3d", Cifar100_3C3D)


class Cifar100_allcnnc(DeepObsTask):
    config = {
        "weight_decay": 5e-4,
        "eval_batch_size": 1000,
        "batch_size": 256,
        "num_epochs": 350,
        "momentum": 0.9,
        "learning_rate": 0.05,
        "lr_decay_points": [(200, 10), (250, 10), (300, 10)],
    }

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.cifar100")(device)
        self.model: Model = registry.model("deepobs.cifar100_allcnnc")(
            num_outputs=100, device=device
        )
        super().__init__(device)

    def _weight_decay_for_param(self, param_name: str) -> float:
        if "bias" in param_name:
            return 0.0
        else:
            return self.config["weight_decay"]


registry.task.register("deepobs.cifar100_allcnnc", Cifar100_allcnnc)


class Cifar10_3C3D(DeepObsTask):
    config = {
        "weight_decay": 0.002,
        "eval_batch_size": 1000,
        "batch_size": 128,
        "num_epochs": 100,
        "momentum": 0.0,
        "learning_rate": 0.01,
    }

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.cifar10")(device)
        self.model: Model = registry.model("deepobs.cifar10_3c3d")(
            num_outputs=10, device=device
        )
        super().__init__(device)

    def _weight_decay_for_param(self, param_name: str) -> float:
        if "bias" in param_name:
            return 0.0
        else:
            return self.config["weight_decay"]


registry.task.register("deepobs.cifar10_3c3d", Cifar10_3C3D)


class FMNIST_2C2D(DeepObsTask):
    config = {"eval_batch_size": 1000, "batch_size": 128, "num_epochs": 100}

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.fmnist")(device)
        self.model: Model = registry.model("deepobs.mnist_2c2d")(
            num_outputs=10, device=device
        )
        super().__init__(device)


registry.task.register("deepobs.fmnist_2c2d", FMNIST_2C2D)


class MNIST_2C2D(DeepObsTask):
    config = {"eval_batch_size": 1000}

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.mnist")(device)
        self.model: Model = registry.model("deepobs.mnist_2c2d")(
            num_outputs=10, device=device
        )
        super().__init__(device)


registry.task.register("deepobs.mnist_2c2d", MNIST_2C2D)


class FMNIST_MLP(DeepObsTask):
    config = {"eval_batch_size": 1000}

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.fmnist")(device)
        self.model: Model = registry.model("deepobs.mlp")(num_outputs=10, device=device)
        super().__init__(device)


registry.task.register("deepobs.fmnist_mlp", FMNIST_MLP)


class MNIST_logreg(DeepObsTask):
    config = {"eval_batch_size": 1000}

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.mnist")(device)
        self.model: Model = registry.model("deepobs.mnist_logreg")(
            num_outputs=10, device=device
        )
        super().__init__(device)


registry.task.register("deepobs.mnist_logreg", MNIST_logreg)


class MNIST_MLP(DeepObsTask):
    config = {"eval_batch_size": 1000}

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.mnist")(device)
        self.model: Model = registry.model("deepobs.mlp")(num_outputs=10, device=device)
        super().__init__(device)


registry.task.register("deepobs.mnist_mlp", MNIST_MLP)


class MNIST_VAE(DeepObsTask):
    config = {"eval_batch_size": 1000, "batch_size": 64, "num_epochs": 50}

    def __init__(self, device=DEFAULT_DEVICE, _data=None):
        if _data is None:
            self._data = registry.dataset("deepobs.mnist")(device)
        else:
            self._data = _data
        self.model: Model = registry.model("deepobs.vae")(n_latent=8, device=device)
        self._vae_loss = lambda outputs, x: vae_loss_function_factory()(
            outputs[0], x, outputs[1], outputs[2]
        )
        super().__init__(device)

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
        loss = self._vae_loss(prediction, batch.x)
        return loss, buffers

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
            loss = self._vae_loss(output, batch.x)
            return {"loss": loss}


registry.task.register("deepobs.mnist_vae", MNIST_VAE)


class FMNIST_VAE(MNIST_VAE):
    config = {"eval_batch_size": 1000, "batch_size": 64, "num_epochs": 100}

    def __init__(self, device=DEFAULT_DEVICE):
        super.__init__(device, _data=registry.dataset("deepobs.fmnist")(device))


registry.task.register("deepobs.fmnist_vae", FMNIST_VAE)


class QuadraticDeep(DeepObsTask):
    config = {"eval_batch_size": 1000, "batch_size": 128, "num_epochs": 100}

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.quadratic")(device)

        rng = np.random.RandomState(42)
        eigenvalues = np.concatenate(
            (rng.uniform(0.0, 1.0, 90), rng.uniform(30.0, 60.0, 10)), axis=0
        )
        D = np.diag(eigenvalues)
        R = random_rotation(D.shape[0])
        hessian = np.matmul(np.transpose(R), np.matmul(D, R))
        hessian = torch.from_numpy(hessian).to(device, torch.float32)
        self.model: Model = registry.model("deepobs.quadratic_deep")(
            device, 100, hessian
        )
        super().__init__(device)

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
        return torch.mean(prediction), buffers

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
            return {"loss": torch.mean(output)}


registry.task.register("deepobs.quadratic_deep", QuadraticDeep)


class SVHN_WRN164(DeepObsTask):
    config = {
        "weight_decay": 5e-4,
        "eval_batch_size": 1000,
        "batch_size": 128,
        "num_epochs": 160,
        "momentum": 0.9,
        "learning_rate": 0.01,
        "lr_decay_points": [(80, 10), (120, 10)],
    }

    def __init__(self, device=DEFAULT_DEVICE):
        self._data = registry.dataset("deepobs.svhn")(device)
        self.model: Model = registry.model("deepobs.wrn")(
            num_outputs=10, num_residual_blocks=2, widening_factor=4, device=device
        )
        super().__init__(device)

    def _weight_decay_for_param(self, param_name: str) -> float:
        if ("weight" in param_name) and (
            ("dense" in param_name) or ("conv" in param_name)
        ):
            return self.config["weight_decay"]
        else:
            return 0.0


registry.task.register("deepobs.svhn_wrn164", SVHN_WRN164)


def _squared_norm(tensor):
    return torch.sum(tensor ** 2)

