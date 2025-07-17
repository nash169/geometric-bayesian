#!/usr/bin/env python
# encoding: utf-8

def ef(f, t, x, u, dt):
    return x + dt * f(t, x, u)
