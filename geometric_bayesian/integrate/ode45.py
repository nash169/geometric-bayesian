#!/usr/bin/env python
# encoding: utf-8

def ode45(f, t, x, u, dt):
    f1 = f(t, x, u)
    f2 = f(t + dt / 2, x + dt / 2 * f1, u)
    f3 = f(t + dt / 2, x + dt / 2 * f2, u)
    f4 = f(t + dt / 2, x + dt / 2 * f3, u)
    return x + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
