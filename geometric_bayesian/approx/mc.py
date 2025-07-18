#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from flax import nnx
from geometric_bayesian.utils.types import Callable, Int, Optional, Matrix
from geometric_bayesian.utils.helper import wrap_pytree_function
from geometric_bayesian.densities.abstract_density import AbstractDensity


def pred_posterior_mean(
    model,
    posterior: AbstractDensity | Matrix,
    size: Optional[Int] = 100,
    **kwargs
):
    params_samples = posterior.sample(size=size, **kwargs) if isinstance(posterior, AbstractDensity) else posterior

    graph_def, map_params = nnx.split(model)

    def model_fn(x):
        return jax.vmap(
            wrap_pytree_function(
                lambda params: nnx.call((graph_def, params))(x.reshape(x.shape[0], -1))[0],
                map_params
            ),
            in_axes=1
        )(params_samples)

    return lambda x: jax.nn.sigmoid(model_fn(x)).mean(axis=0)


def pred_posterior_std(
    model,
    posterior: AbstractDensity | Matrix,
    size: Optional[Int] = 100,
    **kwargs
):
    params_samples = posterior.sample(size=size, **kwargs) if isinstance(posterior, Callable) else posterior

    graph_def, map_params = nnx.split(model)

    def model_fn(x):
        return jax.vmap(
            wrap_pytree_function(
                lambda params: nnx.call((graph_def, params))(x.reshape(x.shape[0], -1))[0],
                map_params
            ),
            in_axes=1
        )(params_samples)

    return lambda x: jax.nn.sigmoid(model_fn(x)).std(axis=0)


def pred_posterior(
        model,
        posterior: AbstractDensity | Matrix,
        likelihood: Callable,
        size: Optional[Int] = 100,
        **kwargs
):
    params_samples = posterior.sample(size=size, **kwargs) if isinstance(posterior, Callable) else posterior

    graph_def, map_params = nnx.split(model)

    def model_fn(x):
        return jax.vmap(
            wrap_pytree_function(
                lambda params: nnx.call((graph_def, params))(x.reshape(-1))[0],
                map_params
            ),
            in_axes=1
        )(params_samples)

    return lambda x, y: jax.vmap(lambda mu, y: likelihood(mu)._log(y), in_axes=(0, None))(model_fn(x), y).mean()
