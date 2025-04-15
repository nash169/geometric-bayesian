#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Callable, Optional
from geometric_bayesian.operators.linear_operator import LinearOperator


def pullmetric(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    r"""
    Calculate the manifold metric via pullback of the embedding space metric

    Args:
        f: Embedding map
        h: Ambient (embedding space) metric

    Returns:
        Metric vector product
    """
    def pullmetric_fn(x, v):
        jvp = jax.linearize(f, x)[1]
        hjv = h(jvp(v)) if h is not None else jvp(v)
        return jax.linear_transpose(jvp, v)(hjv)[0]
    return pullmetric_fn


def christoffel_fk(
    f: Callable,
    h: Optional[LinearOperator] = None
):
    r"""
    Calculate the christoffel symbols of the first kind (T_kij) given the embedding and the ambient metric

    Args:
        f: Embedding map
        h: Ambient (embedding space) metric

    Returns:
        Christoffel symbols first kind
    """
    def vjp(x): return jax.vjp(f, x)[1]
    def grad(x): return jax.lax.map(vjp(x), jnp.eye(len(f(x))))
    def hvp(x, v): return jax.jvp(grad, (x,), (v,))[1][0]
    return lambda x, v: vjp(x)(hvp(x, v)@v)[0]


def christoffel_sk(
    f: Callable,
    h: Optional[LinearOperator] = None
):
    r"""
    Calculate the christoffel symbols of the second kind (T_kij) given the embedding and the ambient metric

    Args:
        f: Embedding map
        h: Ambient (embedding space) metric

    Returns:
        Christoffel symbols second kind
    """
    def jvp(x): return jax.linearize(f, x)[1]
    def vjp(x): return jax.linear_transpose(jvp(x), x)
    def grad(x): return jax.lax.map(vjp(x), jnp.eye(len(f(x))))
    def hvp(x, v): return jax.jvp(grad, (x,), (v,))[1][0]

    def hJv(x, v): return h(jvp(x)(v)) if h is not None else jvp(x)(v)
    def g(x, v): return vjp(x)(hJv(x, v))[0]
    return lambda x, v: jax.scipy.sparse.linalg.cg(lambda v: g(x, v), vjp(x)(hvp(x, v)@v)[0])[0]
