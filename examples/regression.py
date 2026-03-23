# ---
# jupyter:
#   jupytext:
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
# %load_ext autoreload
# %autoreload 2
    
import os
import sys
import time
import glob
import gc

import jax
import jax.numpy as jnp
from flax import nnx
from jax import grad, jit, vmap, random
import optax
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join('../')))
from geometric_bayesian.models import MLP
from geometric_bayesian.utils.train import DataLoader, OptCfg, TrainCfg, train
from geometric_bayesian.utils.helper import make_sinusoid
from geometric_bayesian.utils.plot import plot, scatter

# %%
sinus_factor=5.0
X_train, y_train, _, _, X_test, y_test = make_sinusoid(
    n_train=150, n_valid=50, n_test=100,
    interval=(-2.0,2.0),
    noise=0.15,
    sinus_factor=sinus_factor
)

# %%
model = MLP(
    layers=[1, 8, 8, 1],
    nl = nnx.tanh,
    use_bias=True,
)
num_params = model.size

# %%
fig = scatter(jnp.vstack((X_train, y_train)).T, color="blue", alpha=0.6, label="Training data")
fig = scatter(jnp.vstack((X_test, y_test)).T, fig=fig, color="green", alpha=0.6, label="Test data")
fig = plot(lambda x: jnp.sin(sinus_factor*x), fig=fig, range=[-2,2], color="black", linestyle="--", label="True function")
fig = plot(model, fig=fig, range=[-2,2], color="red", label="Prediction")

# %%
from geometric_bayesian.densities import Normal, MultivariateNormal
from geometric_bayesian.functions.likelihood import neg_logll
from geometric_bayesian.operators import DiagOperator


likelihood = lambda f : Normal(var=jnp.array(1.), mean=f)

prior_cov = DiagOperator(
    diag = jnp.array(10.), 
    dim = num_params
)
prior = MultivariateNormal(cov=prior_cov)


# %%
def loss_fn(model, x, y):
    y_pred = model(x)
    return neg_logll(likelihood, y, y_pred)

cfg = TrainCfg(
    opt=optax.adam(1e-1), 
    steps=1000, 
    batch_size=X_train.shape[0],
    verbose=True
)
loss_val = train(model, X_train.reshape(-1,1), y_train, loss_fn, cfg)

# %%
fig = scatter(jnp.vstack((X_train, y_train)).T, color="blue", alpha=0.6, label="Training data")
fig = scatter(jnp.vstack((X_test, y_test)).T, fig=fig, color="green", alpha=0.6, label="Test data")
fig = plot(lambda x: jnp.sin(sinus_factor*x), fig=fig, range=[-2,2], color="black", linestyle="--", label="True function")
fig = plot(model, fig=fig, range=[-2,2], color="red", label="Prediction")

# %%
from geometric_bayesian.densities import MultivariateNormal
from geometric_bayesian.operators import DiagOperator

MultivariateNormal(model(X_train).squeeze(), DiagOperator(jnp.array(1.0), len(y_train)))._log(y_train)

# %%
ggn_mv = ggn(
    model=model,
    train_data=(X_train, y_train),
    likelihood_density=Normal, 
    cov=jnp.array(1.0)
)

# %%
graph_def, map_params = nnx.split(model)
def model_fn(input, params):
    return nnx.call((graph_def, params))(input)[0]


# %%
num_params = sum(x.size for x in jax.tree.leaves(map_params))
eye_pytree = array_to_pytree(jnp.eye(num_params), map_params)
# precision = pytree_to_array(jax.lax.map(ggn_mv, eye_pytree, batch_size=None),axis=0)

# %%
from geometric_bayesian.types import Vector, Matrix

def pf_jvp(input: Vector | Matrix, vector: Vector) -> Vector | Matrix:
    return jax.jvp(
        lambda p: model_fn(input=input, params=p),
        (map_params,),
        (vector,),
    )[1]

def pf_vjp(input: Vector | Matrix, vector: Vector | Matrix) -> Vector | Matrix:
    out, vjp_fun = jax.vjp(
        lambda p: model_fn(input=input, params=p), map_params
    )
    return vjp_fun(vector)


# %%
grad_net = jax.lax.map(lambda p : pf_jvp(X_test[0], p), eye_pytree, batch_size=None).squeeze()

# %%
pf_jvp(X_test[0], ggn_mv(pf_vjp(X_test[0],jnp.array([1.0]))[0]))

# %%
tmp, _ = jax.tree.flatten(eye_pytree)

# %%
tmp[0].shape

# %%
pf_jvp(X_test[0],eye_pytree)

# %%
