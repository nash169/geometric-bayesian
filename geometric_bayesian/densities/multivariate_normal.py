#!/usr/bin/env python
# encoding: utf-8

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

from geometric_bayesian.utils.types import Scalar, Vector, Matrix
from geometric_bayesian.operators.linear_operator import LinearOperator


class MultivariateNormal:
    def __init__(
        self,
        mean: Scalar | Vector,
        cov: LinearOperator,
    ) -> None:
        r"""
        Define normal distribution.
        covType: Float              -> spherical covariance
                 Vector             -> diagonal covariace
                 LinearOperator   -> full covariance
        """
        # assert isinstance(mean, Scalar) or isinstance(mean, Vector)
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
        return jnp.exp(self.log(x))

    def log(
        self,
        x: Vector | Matrix
    ) -> Scalar | Vector:
        if isinstance(x, Vector):
            return self._log(x)
        elif isinstance(x, Matrix):
            return jax.vmap(self._log, in_axes=(0,))(x)
        else:
            msg = "Input type not valid [ Vector | Matrix ]"
            ValueError(msg)

    # def logpdf_jvp(
    #     self,
    #     x: Vector | Matrix,
    #     v: Vector | Matrix
    # ) -> Scalar | Vector:
    #     r"""
    #     Returns jacobian-vector product the log probability density.
    #     Note that when the mean is a scalar cov is
    #     """
    #     jvp = self._logpdf_jvp
    #
    #     if isinstance(x, Matrix):
    #         jvp = jax.vmap(jvp, in_axes=(0, None), out_axes=0)
    #
    #     if isinstance(v, Matrix):
    #         jvp = jax.vmap(jvp, in_axes=(None, 1), out_axes=1)
    #
    #     return jvp(x, v)

    # def logpdf_jvp_mean(
    #     self,
    #     x: Vector | Matrix,
    #     v: Vector | Matrix
    # ) -> Scalar | Vector:
    #     r"""
    #     Returns jacobian-vector product the log probability density.
    #     Note that when the mean is a scalar cov is
    #     """
    #     jvp_mean = self._logpdf_jvp_mean
    #
    #     if isinstance(x, Matrix):
    #         jvp_mean = jax.vmap(jvp_mean, in_axes=(0, None), out_axes=0)
    #
    #     if isinstance(v, Matrix):
    #         jvp_mean = jax.vmap(jvp_mean, in_axes=(None, 1), out_axes=1)
    #
    #     return jvp_mean(x, v)

    def _log(
        self,
        x: Vector
    ) -> Scalar:
        return -0.5*(jnp.log(self._cov.det()) + self._cov.size()[0]*jnp.log(math.pi) + self._cov.inv_quad(x-self._mean))

    def _jvp(
        self,
        x: Vector,
        v: Vector
    ) -> Scalar:
        return jnp.dot(self._cov.mv(self._mean-x), v)

    def _jvp_mean(
        self,
        x: Vector,
        v: Vector
    ) -> Scalar:
        return jnp.dot(self._cov.mv(x-self._mean), v)

    def _jvp_cov(
        self,
        x: Vector,
        v: Vector
    ) -> None:
        msg = "Jacobian vector product with respect to covariance matric not implemented yet"
        NotImplementedError(msg)

    def _jvp_params(
        self,
        x: Vector,
        v: Vector
    ) -> Tuple:
        return lambda x, v: self._jvp_mean(x, v), lambda x, v: self._jvp_cov(x, v)

    def _hvp(
        self,
        x: Vector,
        v: Vector
    ) -> Vector:
        return self._cov.solve(v)

    def _hvp_mean(
        self,
        x: Vector,
        v: Vector
    ) -> Vector:
        return -self._cov.solve(v)

    def _hvp_cov(
        self,
        x: Vector,
        v: Vector
    ) -> None:
        msg = "Hessian vector product with respect to covariance matric not implemented yet"
        NotImplementedError(msg)

    def _hvp_params(
        self,
    ) -> Tuple:
        return lambda x, v: self._hvp_mean(x, v), lambda x, v: self._hvp_cov(x, v)
