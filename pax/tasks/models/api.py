from typing import Callable, OrderedDict, Tuple, Any
import torch

Params = OrderedDict[str, torch.Tensor]
Buffers = OrderedDict[str, torch.Tensor]

class Model:
    module: torch.nn.Module

    def init(self, seed: int = 0) -> Tuple[Params, Buffers]:
        pass

    def forward(self, params, x, buffers=None, is_training=True) -> Tuple[Any, Buffers]:
        pass
