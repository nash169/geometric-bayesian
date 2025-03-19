#!/usr/bin/env python
# encoding: utf-8

import math
import jax
import jax.numpy as jnp
from geometric_bayesian.utils.types import Matrix, PyTree, List, Vector, PyTreeDef, Optional, Int


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
