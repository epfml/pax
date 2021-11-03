from functools import partial
from typing import Any, Tuple

import pax
import pax.tasks.registry as registry
import torch
from pax.tasks.models.api import Buffers, Model, Params

import torchvision

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchVisionModel(Model):
    def __init__(self, module, device=DEFAULT_DEVICE, *args, **kwargs):
        self._args = args
        self._device = device
        self._module = module
        self._kwargs = kwargs

        if "num_outputs" in self._kwargs:
            self._kwargs["num_classes"] = self._kwargs.pop("num_outputs")

        self.init(seed=0)

    def init(self, seed: int = 0) -> Tuple[Params, Buffers]:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.module = self._module(*self._args, **self._kwargs)
            self.module.to(self._device)
        self._forward = pax.functional_module(self.module, preserve_original=False)
        return pax.get_params(self.module), pax.get_buffers(self.module)

    def forward(self, params, x, buffers=None, is_training=True) -> Tuple[Any, Buffers]:
        return self._forward(params, x, buffers=buffers, is_training=is_training)


for name in dir(torchvision.models):
    m = getattr(torchvision.models, name)
    if str(type(m)) == "<class 'function'>":
        registry.model.register(f"torchvision.{name}", partial(TorchVisionModel, m))

for name in dir(torchvision.models.segmentation):
    m = getattr(torchvision.models.segmentation, name)
    if str(type(m)) == "<class 'function'>":
        registry.model.register(f"torchvision.segmentation.{name}", partial(TorchVisionModel, m))

for name in dir(torchvision.models.detection):
    m = getattr(torchvision.models.detection, name)
    if str(type(m)) == "<class 'function'>":
        registry.model.register(f"torchvision.detection.{name}", partial(TorchVisionModel, m))
