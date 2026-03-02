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
# jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.abspath(os.path.join('../')))

# %%
from geometric_bayesian.utils import DataLoader, plot_regression
rng_key = jax.random.key(0)
num_training_samples = 150
batch_size = 20
sigma_noise=0.3

# Split RNG key for reproducibility
rng_key, rng_noise = random.split(rng_key)

# Generate random training data
X_train = (
    random.uniform(rng_key, (num_training_samples, 1)) * 10
) 
noise = random.normal(rng_noise, X_train.shape) * sigma_noise
y_train = jnp.sin(X_train) + noise

# Generate testing data
X_test = jnp.linspace(-5, 15, 500).reshape(-1, 1)
y_test = jnp.sin(X_test)

# Create the training data loader
train_loader = DataLoader(X_train, y_train.squeeze(), batch_size)

# %%
from geometric_bayesian.models import MLP
model = MLP(
    layers=[1,64,1],
    # nl = nnx.relu, # nnx.tanh
    use_bias=True,
    # param_dtype=jax.numpy.float64
)
num_params = model.size

# %%
fig = plot_regression(model, X_test, y_test, X_train, y_train)

# %%
from geometric_bayesian.densities import Normal, MultivariateNormal
from geometric_bayesian.functions.likelihood import neg_logll
from geometric_bayesian.operators import DiagOperator

# likelihood_cov = DiagOperator(
#     diag = jnp.array(1.), 
#     dim = 1
# )
# p_ll = lambda f : MultivariateNormal(cov=likelihood_cov, mean=f)
p_ll = lambda f : Normal(var=jnp.array(1.), mean=f)

prior_cov = DiagOperator(
    diag = jnp.array(10.), 
    dim = num_params
)
p_prior = MultivariateNormal(cov=prior_cov)

# %%
n_epochs = 1000
step_size = 1e-2

optimizer = nnx.Optimizer(model, optax.adam(step_size))

# def criterion(x, y):
#     return jnp.mean(0.5*jnp.square(x - y))

def loss_fn(model, x, y):
    y_pred = model(x)
    return neg_logll(p_ll, y, y_pred) # + jnp.log(1/jnp.sqrt(2*jnp.pi)) - p_prior(model.params)/y.shape[0]

@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model,x,y)
    optimizer.update(grads) 
    return loss


# %%
losses = []
for epoch in range(n_epochs):
    for x_tr, y_tr in train_loader:
        losses.append(train_step(model, optimizer, x_tr, y_tr))

    if epoch % 100 == 0:
        print(f"[epoch {epoch}]: loss: {losses[-1]:.4f}")

print(f'{optimizer.step.value = }')
print(f"Final loss: {losses[-1]:.4f}")

# %%
fig = plot_regression(model, X_test, y_test, X_train, y_train)

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
