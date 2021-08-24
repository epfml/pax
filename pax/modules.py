import torch
from copy import deepcopy


def functional_module(module: torch.nn.Module, preserve_original: bool = True):
    if preserve_original:
        module = deepcopy(module)

    def forward(params, *args, buffers=None, is_training=True, **kwargs):
        if is_training:
            module.train()
        else:
            module.eval()

        for param, value in zip(module.parameters(), params):
            param.data = value

        if buffers is not None:
            for buffer, value in zip(module.buffers(), buffers):
                buffer.data = value.detach().clone()

        out = module(*args, **kwargs)

        if buffers is not None:
            new_buffers = [b.data for b in module.buffers()]
            return out, new_buffers
        else:
            return out

    return forward
