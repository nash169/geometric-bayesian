#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)


def gradient_check(f, df, in_dim, out_dim=1):
    rng_key = jax.random.key(0)
    x = jax.random.uniform(rng_key, (in_dim,))
    v = jax.random.uniform(rng_key, (in_dim,))
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))
    print(x)
    print(v)

    grad = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v))

    t = jnp.log(t)
    grad = jnp.log(grad)

    w = [25, 35]
    slope = jnp.mean((grad[w[0] + 1:w[1]] - grad[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("First order Taylor expansion slope:", slope, "- It should be approximately equal to 2.0")

    return True if jnp.abs(slope - 2.0) <= 1e-3 else False, grad, t


def hessian_check(f, df, ddf, in_dim, out_dim=1):
    rng_key = jax.random.key(0)
    x = jax.random.uniform(rng_key, (in_dim,))
    v = jax.random.uniform(rng_key, (in_dim,))
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))

    hess = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v) - 0.5 * jnp.pow(t, 2) * (ddf(x, v) @ v))

    t = jnp.log(t)
    hess = jnp.log(hess)

    w = [25, 35]
    slope = jnp.mean((hess[w[0] + 1:w[1]] - hess[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("Second order Taylor expansion slope:", slope, "- It should be approximately equal to 3.0")

    return True if jnp.abs(slope - 3.0) <= 1e-3 else False, hess, t


if __name__ == "__main__":
    def fun(x):
        return jnp.sum(x**3)

    def grad_fun(x, v):
        return (3 * x**2) @ v

    def hess_fun(x, v):
        return (6 * jnp.diag(x)) @ v

    ok_grad, grads, ts = gradient_check(fun, grad_fun, in_dim=3, out_dim=1)
    ok_hess, hesss, ts = hessian_check(fun, grad_fun, hess_fun, in_dim=1, out_dim=1)

    fig, ax = plt.subplots()
    ax.scatter(ts, hesss)
    ax.scatter(ts[25].item(), hesss[25].item(), c='r')
    plt.show()
