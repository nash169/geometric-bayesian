#!/usr/bin/env python
# encoding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt

gb_root = os.path.abspath(
    os.path.join(os.getcwd(), "/Users/bernardo/Repos/geometric-bayesian")
)
if not os.path.isdir(gb_root):
    gb_root = "/home/ubuntu/geometric-bayesian"
sys.path.insert(0, gb_root)
from geometric_bayesian.models.gp import GP
from geometric_bayesian.kernels import rbf
from geometric_bayesian.utils.train import TrainCfg, train
from geometric_bayesian.utils.plot import plot, scatter
from geometric_bayesian.utils.helper import make_sinusoid
from geometric_bayesian.operators import PSDOperator

# %%
sinus_factor=5.0
X_train, y_train, _, _, X_test, y_test = make_sinusoid(
    n_train=150, n_valid=50, n_test=100,
    interval=(-2.0,2.0),
    noise=0.15,
    sinus_factor=sinus_factor
)

# %%
model = GP(dim=1, kernel=rbf)
model.params = jnp.log(jnp.array([0.2,0.2,1.0e-2]))
print(model.params)

# %%
fig = scatter(jnp.vstack((X_train, y_train)).T, color="blue", alpha=0.6, label="Training data")
fig = scatter(jnp.vstack((X_test, y_test)).T, fig=fig, color="green", alpha=0.6, label="Test data")
fig = plot(lambda x: jnp.sin(sinus_factor*x), fig=fig, range=[-5,5], color="black", linestyle="--", label="True function")
fig = plot(
    jax.vmap(model.posterior_mu(X_train,y_train)), 
    fn_between=jax.vmap(lambda x: 2*jnp.sqrt(model.posterior_cov(X_train)(x))), 
    fig=fig, range=[-5,5], color="red", label="Prediction"
)


# %%
def loss_fn(m, x, y):
    return -m(x, y) / x.shape[0]

cfg = TrainCfg(opt=optax.adam(1e-2), steps=1000, batch_size=X_train.shape[0])
loss_val = train(model, X_train, y_train, loss_fn, cfg)
print(model.params)

# %%
fig = scatter(jnp.vstack((X_train, y_train)).T, color="blue", alpha=0.6, label="Training data")
fig = scatter(jnp.vstack((X_test, y_test)).T, fig=fig, color="green", alpha=0.6, label="Test data")
fig = plot(lambda x: jnp.sin(sinus_factor*x), fig=fig, range=[-5,5], color="black", linestyle="--", label="True function")
fig = plot(
    jax.vmap(model.posterior_mu(X_train,y_train)), 
    fn_between=jax.vmap(lambda x: 2*jnp.sqrt(model.posterior_cov(X_train)(x))), 
    fig=fig, range=[-5,5], color="red", label="Prediction"
)

# %%
