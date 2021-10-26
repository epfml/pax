from pax.tree_util import tree_map, tree_flatten, tree_leaves
import inspect
import torch
from typing import Callable, Union, Sequence, Tuple, Any
from pax.utils import CallStack
from functools import wraps

_gradient_call_stack = CallStack()

def value_and_grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> Callable[..., Tuple[Any, Any]]:
    assert allow_int == False, "not implemented"
    assert holomorphic == False, "not implemented"

    argnums_is_sequence = not (type(argnums) is int)
    argnums = [argnums] if type(argnums) is int else argnums

    @_gradient_call_stack.register
    @wraps(fun)
    def value_and_grad_f(*args, **kwargs):
        if max(argnums) >= len(args):
            msg = (
                "differentiating with respect to argnums={} requires at least "
                "{} positional arguments to be passed by the caller, but got only "
                "{} positional arguments."
            )
            raise TypeError(msg.format(argnums, max(argnums) + 1, len(args)))

        _check_callable(fun)

        # We will modify the requires_grad attributes, but want to reset them later
        original_requires_grad = tree_map(lambda t: torch.is_tensor(t) and t.requires_grad, args)

        try:
            new_args = []
            for num, arg in enumerate(args):
                if num in argnums:
                    new_arg = tree_map(
                        lambda x: _force_tensor(x).requires_grad_(), arg
                    )
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)

            with torch.enable_grad():
                if not has_aux:
                    ans = fun(*new_args, **kwargs)
                else:
                    ans, aux = fun(*new_args, **kwargs)

            args_to_pass, treedef = tree_flatten([new_args[a] for a in argnums])
            g = torch.autograd.grad(
                ans, args_to_pass, allow_unused=allow_unused, create_graph=create_graph or len(_gradient_call_stack) > 1
            )
            g = treedef.unflatten(g)
            if not argnums_is_sequence:
                g = g[0]
            if not has_aux:
                return ans.detach(), g
            else:
                return (ans.detach(), aux), g
        finally:
            # Restore original requires_grad attributes
            for t, requires_grad in zip(tree_leaves(args), tree_leaves(original_requires_grad)):
                if torch.is_tensor(t) and t.requires_grad != requires_grad:
                    t.requires_grad_(requires_grad)

    return value_and_grad_f


def grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> Callable:
    value_and_grad_f = value_and_grad(
        fun,
        argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )

    def grad_f(*args, **kwargs):
        _, g = value_and_grad_f(*args, **kwargs)
        return g

    def grad_f_aux(*args, **kwargs):
        (_, aux), g = value_and_grad_f(*args, **kwargs)
        return g, aux

    return grad_f_aux if has_aux else grad_f


def _force_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


def _check_callable(fun):
    """Copied from jax"""
    if not callable(fun):
        raise TypeError(f"Expected a callable value, got {fun}")
    if inspect.isgeneratorfunction(fun):
        raise TypeError(f"Expected a function, got a generator function: {fun}")
