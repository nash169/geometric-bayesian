#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from bayax.utils.types import Callable, Optional


def mcmc(
    f: Callable,
    p: Callable,
    q: Callable,
    dt: float = 0.01,
    T: float = 1.0,
    u: Optional[Callable] = None,
    seed: int = 0,
):
    key = jr.key(seed)

    def step(carry, i):
        t = i * dt
        x, q_curr, key = carry
        key, qkey, akey = jax.random.split(key, 3)

        # sample proposal
        xn = q_curr.sample(key=qkey)

        # proposal density
        q_next = q(f, t, xn, u, dt)

        # acceptance probability
        accept = jnp.log(jax.random.uniform(akey, dtype=x.dtype)) < jnp.minimum(0.0, p(xn) + q_next(x) - p(x) - q_curr(xn))
        carry = jax.lax.cond(
            accept,
            lambda _: (xn, q_next, key),
            lambda _: (x, q_curr, key),
            operand=None,
        )

        return carry, (carry[0], accept)

    def run(x):
        xs, accepts = jax.lax.scan(
            step,
            (x, q(f, 0.0, x, u, dt), key),
            jnp.arange(int(round(T / dt))),
        )[1]
        return (
            jnp.concatenate((jnp.expand_dims(x, axis=0), xs), axis=0),
            jnp.concatenate((jnp.array([True]), accepts), axis=0),
        )

    return run
