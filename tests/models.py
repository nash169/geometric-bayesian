# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
#!/usr/bin/env python
# encoding: utf-8

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

REPO_ROOT = os.path.abspath(os.getcwd())
if os.path.basename(REPO_ROOT) == "examples":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))

from geometric_bayesian.kernels import rbf
from geometric_bayesian.models import MLP, GP
from geometric_bayesian.utils.helper import array_to_pytree, pytree_to_array, gradient_check, hessian_check, random_like
from geometric_bayesian.utils.types import Vector, Float, Optional, Scalar


# %%
def _nnx_grad_vec(loss_fn, graph_def, params_tree, x, y):
    def grad_vec(p_vec):
        model = nnx.merge(graph_def, array_to_pytree(p_vec, params_tree))
        _, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        return pytree_to_array(grads)

    return grad_vec


# %%
def _run_case(name, loss_fn, model, x, y):
    graph_def, params_tree = nnx.split(model)
    params_vec = pytree_to_array(params_tree)
    print(params_vec)

    def f(p_vec):
        model = nnx.merge(graph_def, array_to_pytree(p_vec, params_tree))
        return loss_fn(model, x, y)

    grad_vec = _nnx_grad_vec(loss_fn, graph_def, params_tree, x, y)

    def df(p, v):
        return jnp.dot(grad_vec(p), v)

    def ddf(p, v):
        return jax.jvp(grad_vec, (p,), (v.astype(p.dtype),))[1]

    key_p0, key_v = jr.split(jr.key(100))
    p0, v = random_like(key_p0, params_vec), random_like(key_v, params_vec)

    ok_grad, _, _ = gradient_check(f, df, p0, v)
    ok_hess, _, _ = hessian_check(f, df, ddf, p0, v)
    print(f"{name} nnx gradient_check: {ok_grad}")
    print(f"{name} nnx hessian_check: {ok_hess}")


# %%
mlp = MLP(layers=[3, 4, 1], seed=0)
x_mlp = jr.uniform(jr.key(0), (3,))
y_mlp = jr.uniform(jr.key(1), (1,))

def mlp_loss(m, x, y):
    pred = m(x)
    return jnp.sum((pred - y) ** 2)

_run_case("MLP", mlp_loss, mlp, x_mlp, y_mlp)

gp = GP(dim=3, kernel=rbf, mean=MLP(layers=[3, 4, 1], seed=0), seed=0)
x_gp = jr.uniform(jr.key(2), (3, 3))
y_gp = jr.uniform(jr.key(3), (3,))

def gp_loss(m, x, y):
    return m(x, y)

_run_case("GP", gp_loss, gp, x_gp, y_gp)

# %%
