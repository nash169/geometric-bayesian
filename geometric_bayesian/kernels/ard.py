#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from geometric_bayesian.utils.types import Vector


def ard(
    x: Vector,
    y: Vector,
    l: Vector
):
    return jnp.exp(-.5 * jnp.sum(jnp.pow(x - y, 2) / l))
