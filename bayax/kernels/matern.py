#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from bayax.utils.types import Vector, Float, Optional, Scalar


def matern12(
    x: Scalar | Vector,
    y: Scalar | Vector,
    l: Scalar,
    tau: Scalar = 1.0,
):
    return jnp.exp(tau) * jnp.exp(-jnp.linalg.norm(x - y) / jnp.exp(l))


def matern32(
    x: Scalar | Vector,
    y: Scalar | Vector,
    l: Scalar,
    tau: Scalar = 1.0,
):
    beta = jnp.sqrt(3) / jnp.exp(l) * jnp.linalg.norm(x - y)
    return jnp.exp(tau) * (1 + beta) * jnp.exp(-beta)


def matern52(
    x: Scalar | Vector,
    y: Scalar | Vector,
    l: Scalar,
    tau: Scalar = 1.0,
):
    beta = jnp.sqrt(5) / jnp.exp(l) * jnp.linalg.norm(x - y)
    return jnp.exp(tau) * (1 + beta + beta**2 / 3) * jnp.exp(-beta)
