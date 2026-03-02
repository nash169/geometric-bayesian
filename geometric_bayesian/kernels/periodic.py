#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from geometric_bayesian.utils.types import Vector, Float, Optional, Scalar


def periodic(
    x: Scalar | Vector,
    y: Scalar | Vector,
    l: Scalar,
    p: Scalar,
    tau: Scalar = 1.0,
):
    return jnp.exp(tau) * jnp.exp(-2.0 * jnp.sum(jnp.pow(jnp.sin(jnp.pi * jnp.abs(x - y) / p), 2)) / jnp.square(jnp.exp(l)))
