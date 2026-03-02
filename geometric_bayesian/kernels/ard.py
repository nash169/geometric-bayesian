#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from geometric_bayesian.utils.types import Vector, Scalar


def ard(
    x: Vector,
    y: Vector,
    l: Vector,
    tau: Scalar = 1.0
):
    return jnp.exp(-.5 * jnp.sum(jnp.pow(x - y, 2) / l))
