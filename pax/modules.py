from collections import OrderedDict
from copy import deepcopy

import torch


def functional_module(module: torch.nn.Module, preserve_original: bool = True):
    if preserve_original:
        original_device = next(module.parameters()).device
        module = deepcopy(module).to(original_device)

    def forward(params, *args, buffers=None, is_training=True, **kwargs):
        if is_training:
            module.train()
        else:
            module.eval()

        for name, _ in list(module.named_parameters()):
            path = name.split(".")
            mod = module
            for p in path[:-1]:
                mod = getattr(mod, p)
            mod._parameters[path[-1]] = params[name]
        
        if buffers is not None:
            for name, buffer in module.named_buffers():
                buffer.data = buffers[name].detach().clone()

        out = module(*args, **kwargs)

        if buffers is not None:
            new_buffers = OrderedDict((name, bfr.data) for name, bfr in module.named_buffers())
            return out, new_buffers
        else:
            return out

    return forward


def get_params(module: torch.nn.Module):
    return OrderedDict((name, param.detach()) for name, param in module.named_parameters())


def get_buffers(module: torch.nn.Module):
    return OrderedDict(module.named_buffers())
