#!/usr/bin/env python
# encoding: utf-8

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

from geometric_bayesian.utils.types import Scalar, Vector, Matrix, Key, Int
from geometric_bayesian.densities.abstract_density import AbstractDensity
from geometric_bayesian.operators.linear_operator import LinearOperator


class MultivariateNormal(AbstractDensity):
    def __init__(
        self,
        cov: LinearOperator,
        mean: Optional[Scalar | Vector] = None,
    ) -> None:
        r"""
        Define normal distribution.
        covType: Float              -> spherical covariance
                 Vector             -> diagonal covariace
                 LinearOperator   -> full covariance
        """
        self._cov = cov
        if mean is not None:
            assert isinstance(mean, Scalar) or isinstance(mean, Vector), "Mean can only be a Vector or Scalar."
            self._mean = mean
        assert isinstance(cov, LinearOperator)

    def __call__(
        self,
        x: Scalar | Vector,
        **kwargs
    ) -> Scalar | Vector:
        r"""
        Evaluate density.
        """
        return -0.5 * (self._cov.logdet(**kwargs) + self._cov.size()[0] * jnp.log(2 * jnp.pi) + self._cov.invquad(x - self._mean if hasattr(self, "_mean") else x, **kwargs))

    def sample(
        self,
        rng_key: Optional[Key] = None,
        size: Optional[Int] = 1,
        **kwargs
    ):
        if rng_key is None:
            rng_key = jax.random.key(0)
        rv = jax.random.normal(key=rng_key, shape=(self._cov.shape[0], ) if size == 1 else (self._cov.shape[0], size))
        samples = self._cov.squareroot(**kwargs) @ rv
        if hasattr(self, "_mean"):
            samples += self._mean if size == 1 else self._mean[:, None]
        return samples

    def jvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Gradient with respect to the input.
        """
        return jnp.dot(self._cov.solve(x - self._mean if hasattr(self, "_mean") else x, **kwargs), v)

    def hvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Hessian with respect to the input.
        """
        return -self._cov.solve(v, **kwargs)

    def jvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for gradient function with respect to the params.
        """
        return lambda x, v: self._jvp_mean(x, v, **kwargs), lambda x, v: self._jvp_cov(x, v, **kwargs)

    def hvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for hessian function with respect to the params.
        """
        return lambda x, v: self._hvp_mean(x, v, **kwargs), lambda x, v: self._hvp_cov(x, v, **kwargs)

    def _jvp_mean(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Scalar:
        return jnp.dot(self._cov.solve(x - self._mean if hasattr(self, "_mean") else x, **kwargs), v)

    def _jvp_cov(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> None:
        NotImplementedError("Jacobian vector product with respect to covariance matric not implemented yet")

    def _hvp_mean(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Vector:
        return -self._cov.solve(v, **kwargs)

    def _hvp_cov(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> None:
        raise NotImplementedError("Hessian vector product with respect to covariance matric not implemented yet")
