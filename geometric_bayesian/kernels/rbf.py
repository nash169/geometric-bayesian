#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from geometric_bayesian.utils.types import Vector, Float, Optional, Scalar


def rbf(
    x: Scalar | Vector,
    y: Scalar | Vector,
    l: Scalar,
    tau: Scalar = 1.0,
):
    return jnp.exp(tau) * jnp.exp(-.5 * jnp.sum(jnp.pow(x - y, 2)) / jnp.square(jnp.exp(l)))
