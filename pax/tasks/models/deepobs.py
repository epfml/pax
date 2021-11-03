from typing import Any, Tuple

import pax
import pax.tasks.registry as registry
import torch
from pax.tasks.models.api import Buffers, Model, Params
from functools import partial

from deepobs.pytorch.testproblems import testproblems_modules


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepObsModel(Model):
    def __init__(self, module_class, device=DEFAULT_DEVICE, *args, **kwargs):
        self._args = args
        self._device = device
        self._module_class = module_class
        self._kwargs = kwargs
        self.init(seed=0)

    def init(self, seed: int = 0) -> Tuple[Params, Buffers]:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.module = self._module_class(*self._args, **self._kwargs).to(self._device)
            self.module.to(self._device)
        self._forward = pax.functional_module(self.module, preserve_original=False)
        return pax.get_params(self.module), pax.get_buffers(self.module)

    def forward(self, params, x, buffers=None, is_training=True) -> Tuple[Any, Buffers]:
        return self._forward(params, x, buffers=buffers, is_training=is_training)

registry.model.register("deepobs.char_rnn", partial(DeepObsModel, testproblems_modules.net_char_rnn))
registry.model.register("deepobs.cifar10_3c3d", partial(DeepObsModel, testproblems_modules.net_cifar10_3c3d))
registry.model.register("deepobs.cifar100_allcnnc", partial(DeepObsModel, testproblems_modules.net_cifar100_allcnnc))
registry.model.register("deepobs.mlp", partial(DeepObsModel, testproblems_modules.net_mlp))
registry.model.register("deepobs.mnist_2c2d", partial(DeepObsModel, testproblems_modules.net_mnist_2c2d))
registry.model.register("deepobs.mnist_logreg", partial(DeepObsModel, testproblems_modules.net_mnist_logreg))
registry.model.register("deepobs.quadratic_deep", partial(DeepObsModel, testproblems_modules.net_quadratic_deep))
registry.model.register("deepobs.vae", partial(DeepObsModel, testproblems_modules.net_vae))
registry.model.register("deepobs.vgg", partial(DeepObsModel, testproblems_modules.net_vgg))
registry.model.register("deepobs.wrn", partial(DeepObsModel, testproblems_modules.net_wrn))
