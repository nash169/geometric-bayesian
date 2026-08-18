#!/usr/bin/env python
# encoding: utf-8

import jax

from bayax.utils.types import Callable, Optional
from bayax.operators import PSDOperator


def brownian(
        cov: PSDOperator,
) -> Callable:
    def fn(t, x, u):
        return None, cov

    return fn


def brownian_geometric(
        metric: Callable,
        christoffels: Optional[Callable] = None,
) -> Callable:
    def fn(t, x, u):
        g = metric(x)

        if christoffels is not None:
            drift = -0.5 * g.inv(jax.vmap(lambda u: christoffels(x, u, u), in_axes=(1,))(g.inv_sqrt).sum(axis=0))
        else:
            drift = None

        return drift, g.inv

    return fn
