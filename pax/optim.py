from typing import Any, Callable, NamedTuple, Tuple, Optional
from copy import copy, deepcopy

import torch

import pax
from pax import tree_leaves


Params = Any
Gradients = Any
Updates = Any


class OptState(NamedTuple):
    base_optimizer: torch.optim.Optimizer


class FunctionalOptimizer(NamedTuple):
    init: Callable[[Params], OptState]
    step: Callable[[Params, Gradients, OptState], Tuple[Params, OptState]]
    compute_update: Callable[[Params, Gradients, OptState], Tuple[Updates, OptState]]


def functional_optimizer(base_optimizer_class, *args, **kwargs):
    def init(params: Params) -> OptState:
        managed_params = [p.detach() for p in tree_leaves(params)]
        optimizer = base_optimizer_class(managed_params, *args, **kwargs)

        return OptState(optimizer)

    def step(
        params: Params, grads: Gradients, opt_state: OptState, **kwargs
    ) -> Tuple[Params, OptState]:
        """kwargs can be used to override things like the learning rate for this step only"""
        base_optimizer = opt_state.base_optimizer
        managed_params = base_optimizer.param_groups[0]["params"]

        flat_params, treedef = pax.tree_flatten(params)

        # set params and gradients on the model
        for mp, p, g in zip(managed_params, flat_params, treedef.flatten_up_to(grads)):
            mp.data = p.clone()
            mp.grad = g
        
        # copy the optimizer to make sure that the update doesn't affect the input state
        new_base_optimizer = copy(base_optimizer)
        new_base_optimizer.state = pax.tree_map(deepcopy, base_optimizer.state)

        # temporarily override parameters like the learning rate passed as kwargs
        original_values = {}
        for key, value in kwargs.items():
            assert key in new_base_optimizer.param_groups[0]
            original_values[key] = new_base_optimizer.param_groups[0][key]
            new_base_optimizer.param_groups[0][key] = value

        new_base_optimizer.step()

        # reset parameters like the learning rate passed as kwargs
        for key, value in kwargs.items():
            new_base_optimizer.param_groups[0][key] = original_values[key]

        new_params = treedef.unflatten(pax.tree_map(lambda t: t.data, managed_params))

        return new_params, OptState(new_base_optimizer)

    def compute_update(params: Params, grads: Gradients, opt_state: OptState):
        new_params, new_opt_state = step(params, grads, opt_state)
        updates = pax.tree_map(lambda a, b: a - b, new_params, params)
        return updates, new_opt_state

    return FunctionalOptimizer(init, step, compute_update)


class StandaloneScheduler():
    """Wrap a PyTorch scheduler so it's not tightly connected to an optimizer"""
    def __init__(self, scheduler_class: torch.optim.lr_scheduler._LRScheduler, *args, initial_lr: float=1.0, **kwargs):
        self.dummy_optimizer = torch.optim.SGD([torch.tensor([])], lr=initial_lr)
        self.scheduler = scheduler_class(self.dummy_optimizer, *args, **kwargs)
        self.dummy_optimizer.step()  # to avoid warnings
    
    @property
    def lr(self):
        return self.dummy_optimizer.param_groups[0]["lr"]

    # forward all unknown methods to the scheduler
    def __getattr__(self, name):
        return getattr(self.scheduler, name)


def functional_schedule(scheduler_class: torch.optim.lr_scheduler._LRScheduler, *args, initial_lr: float = 1.0, **kwargs):
    def get_lr(step: int):
        scheduler = StandaloneScheduler(scheduler_class, *args, initial_lr=initial_lr, **kwargs)
        for _ in range(step):  # we could make this a bit more efficient with a cache, if it turns out to be a bottleneck
            scheduler.step()
        return scheduler.lr

    return get_lr
