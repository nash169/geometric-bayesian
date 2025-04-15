import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Callable, Optional
from geometric_bayesian.operators.linear_operator import LinearOperator


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


def pullback(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    def fn(x, v):
        _, jvp = jax.linearize(f, x)
        hjv = jvp(v) if h is None else h(jvp(v))
        return jax.linear_transpose(jvp, v)(hjv)[0]
    return fn
