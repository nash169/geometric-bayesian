import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import inspect

from collections.abc import Generator
from geometric_bayesian.utils.types import (
    Matrix,
    PyTree,
    List,
    Vector,
    PyTreeDef,
    Optional,
    Int,
    Callable,
)


def cumsum(seq: Generator) -> list[int]:
    """Compute the cumulative sum of a sequence.

    This function takes a sequence of integers and returns a list of cumulative
    sums.

    Args:
        seq: A generator or sequence of integers.

    Returns:
        A list where each element is the cumulative sum up to that point
        in the input sequence.
    """
    total = 0
    return [total := total + ele for ele in seq]


def array_to_pytree(
    arr: Vector | Matrix, tree: tuple[List, PyTreeDef] | PyTree
) -> PyTree:
    if isinstance(tree, tuple):  # and list(map(type, tree)) == [List, PyTreeDef]
        shapes, tree_def = tree
    elif isinstance(tree, PyTree):
        leaves, tree_def = jax.tree.flatten(tree)
        shapes = [leaf.shape for leaf in leaves]
    else:
        msg = "`tree` must be a tuple [List, PyTreeDef] or a PyTree structure."
        raise TypeError(msg)
    arr_split = jnp.split(arr, cumsum(math.prod(sh) for sh in shapes)[:-1], axis=-1)
    return jax.tree.unflatten(
        tree_def,
        [
            a.reshape(-1, *sh) if isinstance(arr, Matrix) else a.reshape(sh)
            for a, sh in zip(arr_split, shapes, strict=True)
        ],
    )


def pytree_to_array(tree: PyTree, axis: Optional[int] = None) -> Vector | Matrix:
    leaves, _ = jax.tree.flatten(tree)
    return (
        jnp.concatenate([leaf.ravel() for leaf in leaves])
        if axis is None
        else jnp.hstack([leaf.reshape(leaf.shape[axis], -1) for leaf in leaves])
    )


def pytree_size(tree: PyTree) -> Int:
    return sum(x.size for x in jax.tree.leaves(tree))


def wrap_pytree_function(f: Callable, tree: tuple[List, PyTreeDef] | PyTree):
    def fn_wrap(*args):
        return pytree_to_array(f(*[array_to_pytree(arg, tree) for arg in args]))

    return fn_wrap


def wrap_array_function(f: Callable, tree: tuple[List, PyTreeDef] | PyTree):
    def fn_wrap(*args):
        return array_to_pytree(f(*[pytree_to_array(arg) for arg in args]), tree)

    return fn_wrap


def random_psd(dim):
    import jax
    import jax.numpy as jnp

    rng_key = jax.random.key(0)
    mat = jax.random.uniform(rng_key, shape=dim)
    mat = mat + mat.T + 100 * jnp.eye(dim)
    return mat


def make_moons(n: int = 300, noise: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = rng.uniform(0.0, math.pi, size=n1)
    t2 = rng.uniform(0.0, math.pi, size=n2)
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    x2 = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(n1), np.ones(n2)], axis=0)
    X = X + noise * rng.normal(size=X.shape)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def make_clusters(n: int = 30, m: int = 30, noise: float = 0.15, seed: int = 0):
    X_1 = jax.random.multivariate_normal(jax.random.key(seed), jnp.array([-1, -1]), 0.2 * jnp.eye(2), shape=(n,))
    y_1 = jnp.zeros(n)
    X_2 = jax.random.multivariate_normal(jax.random.key(0), jnp.array([1, 1]), 0.2 * jnp.eye(2), shape=(m,))
    y_2 = jnp.ones(m)

    X = jnp.concatenate((X_1, X_2), axis=0)
    y = jnp.concatenate((y_1, y_2))

    return X, y


def make_sinusoid(n: int = 30, noise: float = 0.15, seed: int = 0):
    X_1 = jax.random.multivariate_normal(jax.random.key(seed), jnp.array([-1, -1]), 0.2 * jnp.eye(2), shape=(n,))
    y_1 = jnp.zeros(n)
    X_2 = jax.random.multivariate_normal(jax.random.key(0), jnp.array([1, 1]), 0.2 * jnp.eye(2), shape=(m,))
    y_2 = jnp.ones(m)

    X = jnp.concatenate((X_1, X_2), axis=0)
    y = jnp.concatenate((y_1, y_2))

    return X, y


def random_like(rng_key, x):
    from geometric_bayesian.operators import PSDOperator
    if isinstance(x, jax.Array):
        return jax.random.uniform(jax.random.split(rng_key)[1], shape=x.shape)
    elif isinstance(x, PSDOperator):
        return PSDOperator(rng_key=jax.random.split(rng_key)[1], op_size=x.size()[0])


def gradient_check(f, df, x, v):
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))

    grad = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v))

    t = jnp.log(t)
    grad = jnp.log(grad)

    w = [25, 35]
    slope = jnp.mean((grad[w[0] + 1:w[1]] - grad[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("First order Taylor expansion slope:", slope, "- It should be approximately equal to 2.0")

    return True if jnp.abs(slope - 2.0) <= 1e-1 else False, grad, t


def hessian_check(f, df, ddf, x, v):
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))

    hess = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v) - 0.5 * jnp.pow(t, 2) * (ddf(x, v) @ v))

    t = jnp.log(t)
    hess = jnp.log(hess)

    w = [30, 40]
    slope = jnp.mean((hess[w[0] + 1:w[1]] - hess[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("Second order Taylor expansion slope:", slope, "- It should be approximately equal to 3.0")

    return True if jnp.abs(slope - 3.0) <= 1e-1 else False, hess, t

# if __name__ == "__main__":
#     def fun(x):
#         return jnp.sum(x**3)
#
#     def grad_fun(x, v):
#         return (3 * x**2) @ v
#
#     def hess_fun(x, v):
#         return (6 * jnp.diag(x)) @ v
#
#     ok_grad, grads, ts = gradient_check(fun, grad_fun, in_dim=3, out_dim=1)
#     ok_hess, hesss, ts = hessian_check(fun, grad_fun, hess_fun, in_dim=1, out_dim=1)
#
#     fig, ax = plt.subplots()
#     ax.scatter(ts, hesss)
#     ax.scatter(ts[25].item(), hesss[25].item(), c='r')
#     plt.show()
