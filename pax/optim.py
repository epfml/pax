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
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]


class FunctionalOptimizer(NamedTuple):
    init: Callable[[Params], OptState]
    step: Callable[[Params, Gradients, OptState], Tuple[Params, OptState]]
    scheduler_step: Callable[[OptState], OptState]
    update: Callable[[Params, Gradients, OptState], Tuple[Updates, OptState]]


def functional_optimizer(base_optimizer_class, *args, scheduler_class=None, **kwargs):
    def init(params: Params) -> OptState:
        managed_params = [p.detach() for p in tree_leaves(params)]
        optimizer = base_optimizer_class(managed_params, *args, **kwargs)

        if scheduler_class is not None:
            scheduler = scheduler_class(optimizer)
        else:
            scheduler = None

        return OptState(optimizer, scheduler)

    def step(
        params: Params, grads: Gradients, opt_state: OptState
    ) -> Tuple[Params, OptState]:
        base_optimizer, scheduler = opt_state
        managed_params = base_optimizer.param_groups[0]["params"]

        flat_params, treedef = pax.tree_flatten(params)

        for mp, p, g in zip(managed_params, flat_params, treedef.flatten_up_to(grads)):
            mp.data = p.clone()
            mp.grad = g

        new_base_optimizer = copy(base_optimizer)
        new_base_optimizer.state = pax.tree_map(deepcopy, base_optimizer.state)

        new_scheduler = copy(scheduler)
        if new_scheduler is not None:
            new_scheduler.optimizer = new_base_optimizer

        new_base_optimizer.step()

        new_params = treedef.unflatten(pax.tree_map(lambda t: t.data, managed_params))

        return new_params, OptState(new_base_optimizer, new_scheduler)

    def scheduler_step(opt_state: OptState) -> OptState:
        base_optimizer, scheduler = opt_state

        if scheduler is None:
            return opt_state

        new_optimizer = copy(base_optimizer)
        new_optimizer.param_groups = copy(new_optimizer.param_groups)
        new_optimizer.param_groups[0] = copy(new_optimizer.param_groups[0])

        new_scheduler = deepcopy(scheduler)
        new_scheduler.optimizer = new_optimizer

        new_scheduler.step()

        return OptState(new_optimizer, new_scheduler)

    def update(params: Params, grads: Gradients, opt_state: OptState):
        new_params, new_opt_state = step(params, grads, opt_state)
        updates = pax.tree_map(lambda a, b: a - b, new_params, params)
        return updates, new_opt_state

    return FunctionalOptimizer(init, step, scheduler_step, update)
