#!/usr/bin/env python
# encoding: utf-8


def em(f, t, x, u, dt):
    drift, diffusion = f(t, x, u)
    return x + dt * drift + diffusion if drift is not None else x + diffusion
