#!/usr/bin/env python
# encoding: utf-8

import math
import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Scalar, Vector, Matrix
from geometric_bayesian.operators.linear_operator import LinearOperator


class MultivariateNormal:
    def __init__(
        self,
        mean: Vector,
        cov: LinearOperator
    ) -> None:
        r"""
        Define normal distribution.
        covType: Float              -> spherical covariance
                 Vector             -> diagonal covariace
                 LinearOperator   -> full covariance
        """
        assert isinstance(mean, Vector)
        self._mean = mean
        assert isinstance(cov, LinearOperator)
        self._cov = cov

    def __call__(
        self,
        x: Scalar | Vector
    ) -> Scalar | Vector:
        r"""
        Evaluate density.
        """
        return jnp.exp(self.logpdf(x))

    def logpdf(
        self,
        x: Vector | Matrix
    ) -> Scalar | Vector:
        r"""
        Returns the log probability density.
        Note that when the mean is a scalar cov is 
        """
        if isinstance(x, Vector):
            return self._logpdf(x)
        elif isinstance(x, Matrix):
            return jax.vmap(self._logpdf, in_axes=0)(x)
        else:
            msg = "input type not valid [Vector | Matrix]"
            ValueError(msg)

    def logpdf_jvp(
        self,
        x: Vector | Matrix,
        v: Vector | Matrix
    ) -> Scalar | Vector:
        r"""
        Returns jacobian-vector product the log probability density.
        Note that when the mean is a scalar cov is 
        """
        jvp = self._logpdf_jvp

        if isinstance(x, Matrix):
            jvp = jax.vmap(jvp, in_axes=(0, None), out_axes=0)

        if isinstance(v, Matrix):
            jvp = jax.vmap(jvp, in_axes=(None, 1), out_axes=1)

        return jvp(x, v)

    def logpdf_jvp_mean(
        self,
        x: Vector | Matrix,
        v: Vector | Matrix
    ) -> Scalar | Vector:
        r"""
        Returns jacobian-vector product the log probability density.
        Note that when the mean is a scalar cov is 
        """
        jvp_mean = self._logpdf_jvp_mean

        if isinstance(x, Matrix):
            jvp_mean = jax.vmap(jvp_mean, in_axes=(0, None), out_axes=0)

        if isinstance(v, Matrix):
            jvp_mean = jax.vmap(jvp_mean, in_axes=(None, 1), out_axes=1)

        return jvp_mean(x, v)

    def _logpdf(
        self,
        x: Vector
    ) -> Scalar:
        return -0.5*(jnp.log(self._cov.det()) + self._mean.shape[0]*jnp.log(math.pi) + self._cov.inv_quad(x-self._mean))

    def _logpdf_jvp(
        self,
        x: Vector,
        v: Vector
    ) -> Scalar:
        return jnp.dot(self._cov.mv(self._mean-x), v)

    def _logpdf_jvp_mean(
        self,
        x: Vector,
        v: Vector
    ) -> Scalar:
        return jnp.dot(self._cov.mv(x-self._mean), v)

    def _logpdf_hvp(
        self,
        x: Vector,
        v: Vector
    ) -> Vector:
        return self._cov.solve(v)

    def _logpdf_hvp_mean(
        self,
        x: Vector,
        v: Vector
    ) -> Vector:
        return -self._cov.solve(v)
