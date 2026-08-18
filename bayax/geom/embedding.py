#!/usr/bin/env python
# encoding: utf-8

import jax

from bayax.utils.types import Callable, Optional
from bayax.operators.linear_operator import LinearOperator


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


def christoffel_fk(f: Callable, h: Optional[LinearOperator] = None):
    r"""
    Calculate the christoffel symbols of the first kind (T_kij) given the embedding
    and the ambient metric

    Args:
        f: Embedding map
        h: Ambient (embedding space) metric

    Returns:
        Christoffel symbols first kind
    """

    def fn(x, v, u):
        jvp = jax.linearize(f, x)[1]
        hess = jax.jvp(lambda x: jax.jvp(f, (x,), (u,))[1], (x,), (v,))[1]
        return jax.linear_transpose(jvp, v)(h(hess) if h is not None else hess)[0]

    return fn


def christoffel_sk(
    f: Callable,
    h: Optional[LinearOperator] = None,
    g: Optional[Callable] = None,
    g_inv: Optional[Callable] = None
):
    r"""
    Calculate the christoffel symbols of the second kind (T^k_ij) given the
    embedding and the ambient metric

    Args:
        f: Embedding map
        h: Ambient (embedding space) metric

    Returns:
        Christoffel symbols second kind
    """
    fk = christoffel_fk(f, h)

    if g_inv is None:
        g_op = pullmetric(f, h) if g is None else g

        def fn(x, v, u):
            return jax.scipy.sparse.linalg.cg(lambda v: g_op(x, v), fk(x, v, u))[0]
    else:
        def fn(x, v, u):
            return g_inv(x, fk(x, v, u))

    return fn
