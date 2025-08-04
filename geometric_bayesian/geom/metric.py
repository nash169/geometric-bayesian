#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Callable, Optional
from geometric_bayesian.operators.linear_operator import LinearOperator


def christoffel_fk(
    g: Callable,
) -> Callable:
    r"""
    Calculate the christoffel symbols of the first kind (T_kij) given the metric

    Args:
        g: Metric

    Returns:
        Christoffel symbols first kind
    """
    def fn(x, v, u):
        dg = jax.jacfwd(lambda x: jax.lax.map(lambda v: g(x, v), jnp.eye(len(x))))(x)
        return jnp.einsum('mji,j->mi', dg, v) @ u - 0.5 * jnp.einsum('ijm,j->mi', dg, v) @ u
        # return 0.5 * (jnp.einsum('mij,j->mi', dg, v) @ u + jnp.einsum('mji,j->mi', dg, v) @ u - jnp.einsum('ijm,j->mi', dg, v) @ u)

    return fn


def christoffel_sk(
    g: Callable,
    g_inv: Optional[Callable] = None
) -> Callable:
    r"""
    Calculate the christoffel symbols of the second kind (T_kij) given metric

    Args:
        g: Metric

    Returns:
        Christoffel symbols second kind
    """
    def fn(x, v, u):
        if g_inv is None:
            return jax.scipy.sparse.linalg.cg(lambda v: g(x, v), christoffel_fk(g)(x, v, u))[0]
        else:
            return g_inv(x, christoffel_fk(g)(x, v, u))

    return fn
