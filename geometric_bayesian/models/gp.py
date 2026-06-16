#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.random as jr
import jax.numpy as jnp
from flax import nnx
import inspect

from geometric_bayesian.utils.types import (
    Matrix,
    Optional,
    Callable,
    Vector,
)
from geometric_bayesian.operators import PSDOperator
from geometric_bayesian.utils.math import gram
from geometric_bayesian.densities import MultivariateNormal
from geometric_bayesian.utils.helper import (
    pytree_to_array,
    array_to_pytree,
    wrap_pytree_function,
)


class GP(nnx.Module):
    def __init__(
        self,
        dim: int,
        kernel: Callable,
        mean: Optional[Callable] = None,
        seed: int = 0,
        **kwargs,
    ):
        r"""
        Gaussian Proces
        """
        key = jr.key(seed)
        self.noise = nnx.Param(jr.normal(key, shape=(1,)))

        if not isinstance(kernel, nnx.Module):
            k_params = list(inspect.signature(kernel).parameters.values())[2:]
            keys = jax.random.split(key, len(k_params) + 1)
            self.kernel_params = nnx.List(
                [
                    nnx.Param(
                        jr.normal(keys[i], shape=(dim if p.annotation.dims else 1,))
                    )
                    for i, p in enumerate(k_params)
                ]
            )
        self.kernel = kernel

        if mean is not None:
            key, subkey = jr.split(key)
            if not isinstance(mean, nnx.Module):
                mu_params = list(inspect.signature(mean).parameters.values())[1:]
                keys = jax.random.split(subkey, len(mu_params))
                self.mean_params = nnx.List(
                    [
                        nnx.Param(
                            jr.normal(keys[i], shape=(dim if p.annotation.dims else 1,))
                        )
                        for i, p in enumerate(mu_params)
                    ]
                )
            self.mean = mean

    def mu_fn(self, x: Vector):
        return (
            self.mean(x, *self.mean_params)
            if hasattr(self, "mean_params")
            else self.mean(x)
        )

    def k_fn(self, x: Vector, y: Vector):
        return (
            self.kernel(x, y, *self.kernel_params)
            if hasattr(self, "kernel_params")
            else self.kernel(x, y)
        )

    def cov_fn(self, x: Vector | Matrix):
        noise = jnp.exp(self.noise).squeeze() + 1e-6
        return gram(self.k_fn, x, noise)

    def prior(self, x: Vector | Matrix):
        return MultivariateNormal(
            cov=PSDOperator(op=self.cov_fn(x), op_size=x.shape[0]),
            mean=self.mu_fn(x) if hasattr(self, "mean") else None,
        )

    def posterior_mu(self, X, y):
        k_xy = lambda x: jax.vmap(self.k_fn, in_axes=(None, 0))(x, X).squeeze()
        prior_cov = PSDOperator(op=self.cov_fn(X), op_size=X.shape[0])
        if hasattr(self, "mean"):
            mean_X = self.mu_fn(X).squeeze()
            return lambda x: self.mu_fn(x) + k_xy(x) @ prior_cov.solve(y - mean_X)
        return lambda x: k_xy(x) @ prior_cov.solve(y)

    def posterior_cov(self, X):
        prior_cov = PSDOperator(op=self.cov_fn(X), op_size=X.shape[0])
        k_xy = lambda x: jax.vmap(self.k_fn, in_axes=(None, 0))(x, X).squeeze()
        k_yx = lambda x: jax.vmap(self.k_fn, in_axes=(0, None))(X, x).squeeze()
        return lambda x, y: self.k_fn(x, y) - k_xy(x) @ prior_cov.solve(k_yx(y))

    def posterior_var(self, X):
        return lambda x: self.posterior_cov(X)(x, x)

    def __call__(self, x: Vector, y: Vector):
        return self.prior(x)(y)

    @property
    def params(self) -> Vector:
        return pytree_to_array(nnx.state(self))

    @params.setter
    def params(self, value):
        nnx.update(self, array_to_pytree(value, nnx.state(self)))
        # self = nnx.merge(nnx.split(self)[0], value)
