import math
import numpy as np
import jax
import jax.numpy as jnp

from collections.abc import Iterator
from geometric_bayesian.utils.types import Matrix, PyTree, List, Vector, PyTreeDef, Optional, Int, Callable


class DataLoader:
    """Simple dataloader."""

    def __init__(self, X, y, batch_size, *, shuffle=True) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = X.shape[0]
        self.indices = np.arange(self.dataset_size)
        self.rng = np.random.default_rng(seed=0)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_size:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx

        return self.X[batch_indices], self.y[batch_indices]


def array_to_pytree(
    arr: Vector | Matrix,
    tree: tuple[List, PyTreeDef] | PyTree
) -> PyTree:
    if isinstance(tree, tuple):  # and list(map(type, tree)) == [List, PyTreeDef]
        shapes, tree_def = tree
    elif isinstance(tree, PyTree):
        leaves, tree_def = jax.tree.flatten(tree)
        shapes = [leaf.shape for leaf in leaves]
    else:
        msg = "`tree` must be a tuple [List, PyTreeDef] or a PyTree structure."
        raise TypeError(msg)
    arr_split = jnp.split(arr, jnp.cumsum(jnp.array([math.prod(sh) for sh in shapes])[:-1]), axis=-1)
    return jax.tree.unflatten(
        tree_def,
        [a.reshape(-1, *sh) if isinstance(arr, Matrix) else a.reshape(sh) for a, sh in zip(arr_split, shapes, strict=True)],
    )


def pytree_to_array(
        tree: PyTree,
        axis: Optional[int] = None
) -> Vector | Matrix:
    leaves, _ = jax.tree.flatten(tree)
    return jnp.concatenate([leaf.ravel() for leaf in leaves]) if axis is None else jnp.hstack([leaf.reshape(leaf.shape[axis], -1) for leaf in leaves])


def pytree_size(
        tree: PyTree
) -> Int:
    return sum(x.size for x in jax.tree.leaves(tree))


def wrap_pytree_function(
    f: Callable,
    tree: tuple[List, PyTreeDef] | PyTree
):
    def fn_wrap(*args):
        return pytree_to_array(f(*[array_to_pytree(arg, tree) for arg in args]))
    return fn_wrap
