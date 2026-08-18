#!/usr/bin/env python
# encoding: utf-8

def ode23(f, t, x, u, dt):
    return x + dt * f(t + dt / 2, x + dt / 2 * f(t, x, u), u)
