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
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from flax import nnx
jax.config.update("jax_enable_x64", True)

# %matplotlib widget
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times'

sys.path.insert(0, os.path.abspath(os.path.join('../')))

# %%
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
TASK = [0, 1]
LAYER_SIZES = [784, 100, 1]

# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x: jnp.array(x.numpy())/255.0),
])
train = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

def collate_fn(batch: list[tuple[jax.Array, int]]) -> tuple[jax.Array, jax.Array]:
    inputs = [s[0] for s in batch]
    targets = [s[1] for s in batch]
    input_batch = jnp.stack(inputs, axis=0)
    target_batch = jnp.array(targets)
    return input_batch, target_batch
    
train = Subset(train, [i for i in range(len(train)) if train.targets[i] in TASK])
test_id = Subset(test, [i for i in range(len(test)) if test.targets[i] in TASK])
test_ood = Subset(test, [i for i in range(len(test)) if test.targets[i] not in TASK])
NUM_DATA_POINTS = len(train)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader_id = DataLoader(test_id, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader_ood = DataLoader(test_ood, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# ## Find MAP

# %%
from geometric_bayesian.models import MLP
model = MLP(
    layers=LAYER_SIZES,
    prob=False,
    param_dtype=jax.numpy.float64
)

# %%
from geometric_bayesian.densities import Bernoulli, MultivariateNormal
from geometric_bayesian.functions.likelihood import neg_logll
from geometric_bayesian.operators import DiagOperator
from geometric_bayesian.utils.helper import pytree_to_array, array_to_pytree

p_ll = lambda f : Bernoulli(f, logits=True)

num_params = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model)))
prior_var = DiagOperator(jnp.array(100.), num_params)
p_prior = MultivariateNormal(prior_var)

# %%
import optax
optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE))

def loss_fn(model, x, y):
    y_pred = model(x)
    return neg_logll(p_ll, y, y_pred) #- p_prior(pytree_to_array(nnx.state(model)))/y.shape[0]

@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(grads) 
    return loss


# %%
from tqdm import tqdm
losses = []
for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for x_tr, y_tr in pbar:
        losses.append(train_step(model, optimizer, x_tr.reshape(x_tr.shape[0], -1), y_tr).item())
        epoch_losses.append(losses[-1])
        avg_loss = sum(epoch_losses)/len(epoch_losses)
        pbar.set_postfix({
                'loss': f'{float(losses[-1]):.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
print(f'{optimizer.step.value = }')
print(f"Final loss: {losses[-1]:.5f}")

# %% [markdown]
# ## Posterior

# %%
from geometric_bayesian.curv.ggn import ggn
from geometric_bayesian.utils.helper import wrap_pytree_function, pytree_to_array, array_to_pytree
from geometric_bayesian.operators import PSDOperator

mini_batch: dict[str, Array] = next(iter(train_loader))
graph_def, map_params = nnx.split(model)

ggn_fn = wrap_pytree_function(
    ggn(
        p = p_ll,
        f = model,
        X = mini_batch[0].reshape(mini_batch[0].shape[0],-1),
        y = mini_batch[1],
    ), 
    map_params
)

ggn_mv = lambda v : ggn_fn(pytree_to_array(map_params), v)
ggn_op = PSDOperator(ggn_mv, op_size=num_params)
ggn_lr = ggn_op.lowrank(num_modes=150, method='lobpcg')
cov_op = (ggn_lr + p_prior._cov).inverse()

# %%
from laplax.api import GGN
from laplax.util.flatten import full_flatten
import numpy as np

graph_def, map_params = nnx.split(model)
def model_fn(input,params):
    return nnx.call((graph_def, params))(input.reshape(input.shape[0],-1))[0]

SCALING_FACTOR = 1 / BATCH_SIZE

ggn_mv = GGN(
    model_fn,
    map_params,
    data = mini_batch,
    loss_fn = "binary_cross_entropy",
    vmap_over_data = True,
    factor = SCALING_FACTOR,
)

test_laplax = full_flatten(ggn_mv(map_params))

# %%
jnp.allclose(test_laplax,ggn_op(pytree_to_array(map_params)))

# %%
from geometric_bayesian.densities import MultivariateNormal
from geometric_bayesian.approx.mc import pred_posterior_mean, pred_posterior_std, pred_posterior

posterior = MultivariateNormal(cov=cov_op, mean=pytree_to_array(map_params))
params_samples = posterior.sample(size=100, num_modes=150, method='lobpcg')

X_test, y_test = next(iter(test_loader_id))

mean_fn = pred_posterior_mean(model, params_samples, num_modes=150, method='lobpcg')
std_fn = pred_posterior_std(model, params_samples, num_modes=150, method='lobpcg')
pred_posterior_fn = pred_posterior(model, params_samples, p_ll, num_modes=150, method='lobpcg')

# %%
