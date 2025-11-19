from chex import Array
import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Callable, Optional, PyTree, Tensor
from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.utils.helper import pytree_to_array, array_to_pytree


def hvp(
    f: Callable
) -> Callable:
    def fn(x, v):
        def vjp(x): return jax.lax.map(jax.vjp(f, x)[1], jnp.eye(len(f(x))))
        return jax.jvp(vjp, (x,), (v,))[1]
    return fn


def hvp_2(
    f: Callable
) -> Callable:
    def fn(x, v):
        return jax.jvp(jax.jacrev(f), (x,), (v,))[1]
    return fn

# def hvp(func: Callable, primals, tangents):
#     return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]


def pullback(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    def fn(x, v):
        _, jvp = jax.linearize(f, x)
        hjv = jvp(v) if h is None else h(jvp(v))
        return jax.linear_transpose(jvp, v)(hjv)[0]
    return fn


def inner_jacobian(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    def fn(x, v):
        _, jvp = jax.linearize(f, x)
        hjv = jvp(v) if h is None else h(jvp(v))
        return jax.linear_transpose(jvp, x)(hjv)[0]
    return fn


def outer_jacobian(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    def fn(x, v):
        _, jvp = jax.linearize(f, x)
        jtv = jax.linear_transpose(jvp, x)(v)[0]
        if h is not None:
            leaves, tree_def = jax.tree.flatten(jtv)
            shapes = [leaf.shape for leaf in leaves]
            jtv = array_to_pytree(h(pytree_to_array(jtv)), (shapes, tree_def))
        return jvp(jtv)
    return fn


def gram(
    k: Callable,
    x: Tensor
) -> Callable:
    n = x.shape[0]
    k_diag = jax.vmap(lambda i: k(x[i], x[i]))(jnp.arange(n))
    i_off, j_off = jnp.triu_indices(n, k=1)
    k_off = jax.vmap(lambda i, j: k(x[i], x[j]))(i_off, j_off)

    def mv(v):
        y = k_diag * v
        y = y.at[i_off].add(k_off * v[j_off])
        y = y.at[j_off].add(k_off * v[i_off])
        return y

    return mv
