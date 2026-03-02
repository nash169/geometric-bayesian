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
from jax import grad, jit, vmap, random
jax.config.update('jax_enable_x64', True)


# %matplotlib widget
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times'

sys.path.insert(0, os.path.abspath(os.path.join('../')))

# %% [markdown]
# ## Find MAP

# %%
from geometric_bayesian.utils import DataLoader

num_samples_1, num_samples_2 = 100, 100
samples_c1 = jax.random.multivariate_normal(jax.random.key(0), jnp.array([-1,-1]), 0.2*jnp.eye(2), shape=(num_samples_1,))
targets_c1 = jnp.zeros(num_samples_1)
samples_c2 = jax.random.multivariate_normal(jax.random.key(0), jnp.array([1,1]), 0.2*jnp.eye(2), shape=(num_samples_2,))
targets_c2 = jnp.ones(num_samples_2)

samples = jnp.concatenate((samples_c1,samples_c2), axis=0)
targets = jnp.concatenate((targets_c1,targets_c2))
train_loader = DataLoader(samples, targets, 30, shuffle=True)

# %%
from geometric_bayesian.models import MLP
model = MLP(
    layers=[2,1],
    use_bias=False,
    param_dtype=jax.numpy.float64
)
num_params = model.size

# %%
from geometric_bayesian.densities import Bernoulli, MultivariateNormal
from geometric_bayesian.functions.likelihood import neg_logll
from geometric_bayesian.operators import DiagOperator

p_ll = lambda f : Bernoulli(f, logits=True)
prior_var = DiagOperator(
    diag = jnp.array(10.), 
    dim = num_params
)
p_prior = MultivariateNormal(cov=prior_var)

# %%
import optax
from flax import nnx

n_epochs = 1000
step_size = 1e-3
optimizer = nnx.Optimizer(model, optax.adam(step_size))

def loss_fn(model, x, y):
    y_pred = model(x)
    return neg_logll(p_ll, y, y_pred) - p_prior(model.params)/y.shape[0]

@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
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
from geometric_bayesian.utils.plot import contour_plot, surf_plot
fig = contour_plot(lambda x: jax.nn.sigmoid(model(x)), min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label='\sigma(f(X))')
ax = fig.axes[0]
ax.scatter(samples_c1[:,0], samples_c1[:,1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:,0], samples_c2[:,1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_aspect('equal', 'box')
ax.legend()


# %%
def loss(params):
    return neg_logll(p_ll, targets, model.fwd_params()(samples, params)) - p_prior(params)/targets.shape[0]
p_min, p_max = model.params-10, model.params+10
fig = contour_plot(jax.vmap(loss), min=p_min, max=p_max, res=100, iso=True, alpha=0.5, zorder=-1, label=r'\mathcal{L}(f,\theta)')

# fig = contour_plot(jax.vmap(loss), min=p_min, max=p_max, res=100, fig=None)
ax = fig.axes[0]
ax.scatter(*model.params)

# %%
fig = surf_plot(jax.vmap(loss), min=p_min, max=p_max, res=100)
ax = fig.axes[0]
ax.scatter(*jnp.append(model.params,loss(model.params)))

# %% [markdown]
# ## Laplace Approximation

# %%
from geometric_bayesian.curv.ggn import ggn
from geometric_bayesian.utils.helper import wrap_pytree_function
from geometric_bayesian.operators.psd_operator import PSDOperator

ggn_fn = wrap_pytree_function(
    ggn(
        p = p_ll,
        f = model,
        X = samples,
        y = targets,
        scaling = float(samples.shape[0])
    ), 
    nnx.state(model)
)
ggn_lr = PSDOperator(lambda v : ggn_fn(model.params, v), op_size=num_params).lowrank(num_modes=num_params)
cov_op = (ggn_lr + p_prior._cov.inverse()).inverse()
posterior = MultivariateNormal(cov=cov_op, mean=model.params)

# %%
from geometric_bayesian.integrate import integrate, ef, ode23, ode45
dt, T = 0.01, 5.0
params_samples = posterior.sample(size=10)
x0 = jnp.expand_dims(model.params, axis=0).repeat(params_samples.shape[1],axis=0)
v0 = params_samples.T - x0

# %%
from geometric_bayesian.approx.mc import pred_posterior_mean, pred_posterior_std, pred_posterior
mean_fn = pred_posterior_mean(model, params_samples)
std_fn = pred_posterior_std(model, params_samples)
pred_posterior_fn = pred_posterior(model, params_samples, p_ll)

# %%
from geometric_bayesian.utils.plot import contour_plot, surf_plot
fig = contour_plot(mean_fn, min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label=r'\mu[p(y|f)]')
ax = fig.axes[0]
ax.scatter(samples_c1[:,0], samples_c1[:,1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:,0], samples_c2[:,1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_aspect('equal', 'box')
ax.legend()

# %%
from geometric_bayesian.utils.plot import contour_plot, surf_plot
fig = contour_plot(std_fn, min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label=r'\sigma[p(y|f)]')
ax = fig.axes[0]
ax.scatter(samples_c1[:,0], samples_c1[:,1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:,0], samples_c2[:,1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_aspect('equal', 'box')
ax.legend()

# %%
from geometric_bayesian.utils.plot import contour_plot, surf_plot
fig = contour_plot(std_fn, min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label=r'\sigma[p(y|f)]')
ax = fig.axes[0]
ax.scatter(samples_c1[:,0], samples_c1[:,1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:,0], samples_c2[:,1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_aspect('equal', 'box')
ax.legend()

# %% [markdown]
# ## Pullback Embedding Geometry

# %%
from geometric_bayesian.geom.embedding import pullmetric, christoffel_sk

def embedding(x):
    return jnp.array([*x, loss(x)])

csk_pull = christoffel_sk(embedding)

@jax.jit
def dynamics_pull(t,x,u):
    x1, x2 = jnp.split(x, 2)
    return jnp.append(x2, -csk_pull(x1,x2))

metric_pull = lambda x, v : pullmetric(embedding)(x, v)
print("Metric at MAP:", jax.lax.map(lambda v: metric_pull(model.params,v), jnp.eye(num_params)))

# %%
step = integrate(
    f = dynamics_pull,
    dt = dt,
    T = T,
    integrator = ode45,
)

start = time.time()
trajectory_pull = jax.vmap(step)(jnp.hstack((x0, v0)))
print(time.time() - start)

# %%
fig = contour_plot(jax.vmap(loss), min=p_min, max=p_max, res=100, alpha=0.5, zorder=-1)
ax = fig.axes[0]
# for i in range(v0.shape[0]):
#     traj = trajectory_pull[0][i]
#     ax.plot(traj[:, 0], traj[:, 1], color="k")
ax.scatter(*model.params)
ax.quiver(x0[:,0], x0[:,1], params_samples.T[:,0]- x0[:,0], params_samples.T[:,1]- x0[:,1])
ax.scatter(params_samples.T[:,0], params_samples.T[:,1])

# %%
fig = surf_plot(jax.vmap(loss), min=p_min, max=p_max, res=100, fig=None)
ax = fig.axes[0]
ax.scatter(*jnp.append(model.params,loss(model.params)))
for i in range(v0.shape[0]):
    traj = trajectory_pull[0][i]
    ax.plot(
        traj[:, 0], 
        traj[:, 1],
        jax.vmap(loss)(jnp.vstack((traj[:, 0], traj[:, 1])).T),
        color="k")

# %% [markdown]
# ## GGN Metric Geometry

# %%
from geometric_bayesian.geom.metric import christoffel_sk

csk_ggn = christoffel_sk(ggn_fn)
@jax.jit
def dynamics_ggn(t,x,u):
    x1, x2 = jnp.split(x, 2)
    return jnp.append(x2, -csk_ggn(x1,x2))

print("Metric:", jax.lax.map(lambda v : ggn_fn(model.params, v), jnp.eye(num_params)))
D,U = jnp.linalg.eigh(jax.lax.map(lambda v : ggn_fn(model.params, v), jnp.eye(num_params)))
print("Eigvals:", D)
print("Eigvecs:", U)

# %%
step = integrate(
    f = dynamics_ggn,
    dt = dt,
    T = 1.0,
    integrator = ode45,
)

start = time.time()
trajectory_ggn = jax.vmap(step, in_axes=0)(jnp.hstack((x0, v0)))
print(time.time() - start)

# %%
fig = contour_plot(jax.vmap(loss), min=p_min, max=p_max, res=100, alpha=0.5, zorder=-1)
ax = fig.axes[0]
for i in range(v0.shape[0]):
    traj = trajectory_ggn[0][i]
    ax.plot(traj[:, 0], traj[:, 1], color="k")
ax.scatter(*model.params)
ax.quiver(x0[:,0], x0[:,1], params_samples.T[:,0]- x0[:,0], params_samples.T[:,1]- x0[:,1])
ax.scatter(params_samples.T[:,0], params_samples.T[:,1])

# %% [raw]
# def integrate(fn, grid_min, grid_max, res):
#     import numpy as np
#     dx = jnp.linspace(grid_min[0], grid_max[0], res)
#     dy = jnp.linspace(grid_min[1], grid_max[1], res)
#     x, y = jnp.meshgrid(dx, dy)
#     z = fn(jnp.vstack((x.ravel(), y.ravel())).transpose()).reshape(res, res)
#     return np.trapezoid(jnp.trapezoid(z, dy, axis=0), dx, axis=0)
#
# ll = lambda f : jax.vmap(lambda y, f: p_ll(f)(y), in_axes=(0,0))(y,f).sum()
# ll_log = lambda f : jax.vmap(lambda y, f: p_ll(f)._log(y), in_axes=(0,0))(y,f).sum()
# llXprior = jax.vmap(lambda p: ll(jax.nn.sigmoid(model_p(array_to_pytree(p, map_params))))*p_prior(p))
# llXprior_log = jax.vmap(lambda p: ll_log(model_p(array_to_pytree(p, map_params))) + p_prior._log(p))
# posterior = lambda p: llXprior_log(p) - jnp.log(integrate(llXprior, p_min, p_max, 100))
#
# fig = contour_plot(posterior, min=p_min, max=p_max, res=100, fig=None)
# ax = fig.axes[0]
# ax.scatter(map_p[0], map_p[1])

# %% [raw]
# from geometric_bayesian.geom.metric import christoffel_sk
#
# csk_ggn = christoffel_sk(ggn_fn)
# @jax.jit
# def dynamics_ggn(x,v):
#     return -csk_ggn(x,v)
#
# print("Metric:", jax.lax.map(lambda v : ggn_fn(model.params, v), jnp.eye(num_params)))
#
# from geometric_bayesian.integrate.integrate import integrate
# from geometric_bayesian.integrate.euler_forward import euler_forward
#
# step = integrate(
#     f = dynamics_ggn,
#     dt = dt,
#     T = T,
#     integrator = euler_forward,
# )
#
# start = time.time()
# trajectory_ggn = jax.vmap(step, in_axes=(0,0))(x0,v0)
# print(time.time() - start)
