#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from geometric_bayesian.utils.types import Callable, Optional, Array, Float, Key
from geometric_bayesian.integrate.ef import ef


def integrate(
    f: Callable,
    u: Optional[Callable | Array] = None,
    dt: Optional[Float] = 0.01,
    T: Optional[Float] = 1.0,
    integrator: Optional[Callable] = ef,
    key: Optional[Key] = None
):
    assert integrator is not None and dt is not None and T is not None

    def step(carry, i):
        t = i * dt
        x, key = carry
        if key is not None:
            key, subkey = jax.random.split(key)
        x = integrator(f=f if key is None else lambda t, x, u: f(t=t, x=x, u=u, key=subkey), t=t, x=carry[0], u=u, dt=dt)
        return (x, key), (x, )

    return lambda x: jax.lax.scan(step, (x, key), jnp.arange(int(T / dt)))[1]

# def euler_forward(f, x, v, dt):
#     vn = v + dt*f(x, v)
#     xn = x + dt*v
#     return xn, vn
#
# def integrate(
#     f: Callable,
#     dt: Optional[float] = 0.01,
#     T: Optional[float] = 1.0,
#     integrator: Optional[Callable] = euler_forward,
# ):
#     assert integrator is not None and dt is not None and T is not None
#
#     def step(carry, _):
#         x, v = integrator(f=f, x=carry[0], v=carry[1], dt=0.01)
#         return (x, v), (x, v)
#
#     return lambda x, v: jax.lax.scan(step, (x, v), None, length=int(T / dt))[1]
