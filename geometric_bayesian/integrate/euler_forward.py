#!/usr/bin/env python
# encoding: utf-8

def euler_forward(f, x, v, dt):
    vn = v + dt*f(x, v)
    xn = x + dt*v
    return xn, vn
