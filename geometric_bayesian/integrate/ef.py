#!/usr/bin/env python
# encoding: utf-8

def ef(f, t, x, u, dt):
    drift, diffusion = f(t, x, u)
    if drift is not None:
        x += dt * drift
    if diffusion is not None:
        x += diffusion
    return x
