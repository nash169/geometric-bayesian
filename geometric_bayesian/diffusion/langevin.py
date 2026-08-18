#!/usr/bin/env python
# encoding: utf-8

import jax

from bayax.utils.types import Callable, Optional
from bayax.operators import PSDOperator


def langevin(
        grad_energy: Callable,
        cov: PSDOperator,
) -> Callable:
    def fn(t, x, u):
        return -0.5 * grad_energy(x), cov

    return fn


def langevin_geometric(
        grad_energy: Callable,
        metric: Callable,
        christoffels: Optional[Callable] = None,
) -> Callable:
    def fn(t, x, u):
        g = metric(x)

        if christoffels is not None:
            drift = -0.5 * g.inv(grad_energy(x) + jax.vmap(lambda u: christoffels(x, u, u), in_axes=(1,))(g.inv_sqrt).sum(axis=0))
        else:
            drift = -0.5 * g.inv(grad_energy(x))

        return drift, g.inv

    return fn
