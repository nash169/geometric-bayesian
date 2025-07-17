#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from geometric_bayesian.utils.types import Callable, Optional, Array, Float
from geometric_bayesian.integrate.ef import ef


def integrate(
    f: Callable,
    u: Optional[Callable | Array] = None,
    dt: Optional[Float] = 0.01,
    T: Optional[Float] = 1.0,
    integrator: Optional[Callable] = ef,
):
    assert integrator is not None and dt is not None and T is not None

    def step(carry, i):
        t = i * dt
        x = integrator(f=f, t=t, x=carry[0], u=u, dt=dt)
        return (x, t), (x, )

    return lambda x: jax.lax.scan(step, (x, 0.), jnp.arange(int(T / dt)))[1]

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
