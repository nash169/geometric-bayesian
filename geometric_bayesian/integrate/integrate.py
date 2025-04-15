#!/usr/bin/env python
# encoding: utf-8

import jax
from geometric_bayesian.utils.types import Callable, Optional
from geometric_bayesian.integrate.euler_forward import euler_forward


def integrate(
    f: Callable,
    dt: Optional[float] = 0.01,
    T: Optional[float] = 1.0,
    integrator: Optional[Callable] = euler_forward,
):
    assert integrator is not None and dt is not None and T is not None

    def step(carry, _):
        x, v = integrator(f=f, x=carry[0], v=carry[1], dt=0.01)
        return (x, v), (x, v)

    return lambda x, v: jax.lax.scan(step, (x, v), None, length=int(T/dt))[1]
