# Would be nice to replace this to remove the dependency on JAX
from typing import Any, Callable, List, Tuple

import torch
from jax.tree_util import (
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_multimap,
    tree_unflatten,
)


def ravel_pytree(pytree) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    """Ravel (i.e. flatten) a pytree of arrays down to a 1D array.

    Args:
        pytree: a pytree of arrays and scalars to ravel.

    Returns:
        A pair where the first element is a 1D array representing the flattened and
        concatenated leaf values, with dtype determined by promoting the dtypes of
        leaf values, and the second element is a callable for unflattening a 1D
        vector of the same length back to a pytree of of the same structure as the
        input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
        a convention a 1D empty array of dtype float32 is returned in the first
        component of the output.

    Transcribed from https://jax.readthedocs.io/en/latest/_modules/jax/_src/flatten_util.html#ravel_pytree
    """
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    unravel_pytree = lambda flat: tree_unflatten(treedef, unravel_list(flat))
    return flat, unravel_pytree


def _ravel_list(lst: List[torch.Tensor]):
    if not lst:
        return torch.tensor([], torch.float32), lambda _: []
    from_dtypes = [l.dtype for l in lst]
    to_dtype = from_dtypes[0]
    assert all(t == to_dtype for t in from_dtypes)
    sizes = [x.numel() for x in lst]
    shapes = [x.shape for x in lst]
    indices = torch.cumsum(torch.tensor(sizes), 0)

    def unravel(arr):
        chunks = torch.split(arr, indices[:-1])
        return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]

    raveled = torch.cat([torch.ravel(e) for e in lst])
    return raveled, unravel
