#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from geometric_bayesian.utils.types import Scalar, Optional, Tuple


class Normal:
    def __init__(
        self,
        mean: Scalar,
        var: Scalar
    ) -> None:
        r"""
        Define normal distribution.
        covType: Float              -> spherical covariance
                 Vector             -> diagonal covariace
                 LinearOperator   -> full covariance
        """
        self._mean, self._var = mean, var

    def __call__(
        self,
        x: Scalar
    ) -> Scalar:
        r"""
        Evaluate density.
        """
        return -0.5 * (jnp.log(self._var) + jnp.log(2 * jnp.pi) + jnp.square(x - self._mean) / self._var)

    def jvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return 0.5 * (self._mean - x) / self._var * v

    def hvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> None:
        NotImplementedError("Hessian vector product with respect to data not implemented yet.")

    def jvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for gradient function with respect to the params.
        """
        return lambda x, v: self._jvp_mean(x, v, **kwargs), lambda x, v: self._jvp_var(x, v, **kwargs)

    def hvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for hessian function with respect to the params.
        """
        return lambda x, v: self._hvp_mean(x, v, **kwargs), lambda x, v: self._hvp_var(x, v, **kwargs)

    def _jvp_mean(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return (x - self._mean) / self._var * v

    def _jvp_var(
        self,
        x: Scalar,
        v: Scalar,
        **kwargs
    ) -> None:
        NotImplementedError("Jacobian vector product with respect to variance not implemented yet")

    def _hvp_mean(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return -v / self._var

    def _hvp_var(
        self,
        x: Scalar,
        v: Scalar,
        **kwargs
    ) -> None:
        raise NotImplementedError("Hessian vector product with respect to variance not implemented yet")

# class Normal:
#     def __init__(
#         self,
#         mean: Scalar | Vector | ScalarFn | VectorFn,
#         cov: Scalar | Vector | LinearOperator
#     ) -> None:
#         r"""
#         Define normal distribution.
#         covType: Float              -> spherical covariance
#                  Vector             -> diagonal covariace
#                  LinearOperator   -> full covariance
#         """
#         self._mean, self._cov = mean, cov
#
#     def __call__(
#         self,
#         x: Scalar | Vector
#     ) -> Scalar | Vector:
#         r"""
#         Evaluate density.
#         """
#         return jnp.exp(self.logpdf(x))
#
#     def logpdf(
#         self,
#         x: Scalar | Vector
#     ) -> Scalar | Vector:
#         r"""
#         Returns the log probability density.
#         Note that when the mean is a scalar cov is
#         """
#         return self._logpdf(x) if self._mean.shape == x.shape else jax.vmap(self._logpdf, in_axes=0)(x)
#
#     def logpdf_jvp(
#         self,
#         x: Scalar | Vector,
#         v: Scalar | Vector
#     ) -> Scalar | Vector:
#         r"""
#         Returns jacobian-vector product the log probability density.
#         Note that when the mean is a scalar cov is
#         """
#         jvp = self._logpdf_jvp
#         if self._mean.shape != x.shape:
#             jvp = jax.vmap(jvp, in_axes=(0, None), out_axes=0)
#         if self._mean.shape != v.shape:
#             jvp = jax.vmap(jvp, in_axes=(None, 1), out_axes=1)
#         return jvp(x, v)
#
#     def logpdf_jvp_mean(
#         self,
#         x: Scalar | Vector | Matrix,
#         v: Scalar | Vector
#     ) -> Scalar | Vector:
#         r"""
#         Returns jacobian-vector product the log probability density.
#         Note that when the mean is a scalar cov is
#         """
#         assert self._mean.shape == v.shape
#         return self.logpdf_jvp_mean(x, v) if self._mean.shape == x.shape else jax.vmap(self._logpdf_jvp_mean, in_axes=(0, None))(x, v)
#
#     def _logpdf(
#         self,
#         x: Scalar | Vector
#     ) -> Scalar | Vector:
#         if jnp.isscalar(self._mean):
#             assert jnp.isscalar(self._cov) and jnp.isscalar(x)
#             return -0.5*(jnp.log(self._cov) + jnp.log(2*math.pi) + jnp.square(x - self._mean)/self._cov)
#         elif isinstance(self._mean, Vector):
#             if jnp.isscalar(self._cov) or isinstance(self._cov, Vector):
#                 return -0.5*(jnp.log(self._cov.prod()) + self._mean.shape[0]*jnp.log(2*math.pi) + jnp.sum(jnp.square(x - self._mean)/self._cov))
#             elif isinstance(self._cov, LinearOperator):
#                 return -0.5*(jnp.log(self._cov.det()) + self._mean.shape[0]*jnp.log(math.pi) + self._cov.inv_quad(x-self._mean))
#
#         msg = "unsupported mean provided"
#         raise ValueError(msg)
#
#     def _logpdf_jvp(
#         self,
#         x: Scalar | Vector,
#         v: Scalar | Vector
#     ) -> Scalar | Vector:
#         if jnp.isscalar(self._mean):
#             assert jnp.isscalar(self._cov) and jnp.isscalar(x)
#             return (self._mean - x)/self._cov*v
#         elif isinstance(self._mean, Vector):
#             if jnp.isscalar(self._cov) or isinstance(self._cov, Vector):
#                 return jnp.dot((self._mean - x)/self._cov, v)
#             elif isinstance(self._cov, LinearOperator):
#                 return jnp.dot(self._cov.mv(self._mean-x), v)
#
#         msg = "unsupported mean provided"
#         raise ValueError(msg)
#
#     def _logpdf_jvp_mean(
#         self,
#         x: Scalar | Vector,
#         v: Scalar | Vector
#     ) -> Scalar | Vector:
#         if jnp.isscalar(self._mean):
#             assert jnp.isscalar(self._cov) and jnp.isscalar(x)
#             return (x - self._mean)/self._cov*v
#         elif isinstance(self._mean, Vector):
#             if jnp.isscalar(self._cov) or isinstance(self._cov, Vector):
#                 return jnp.dot((x - self._mean)/self._cov, v)
#             elif isinstance(self._cov, LinearOperator):
#                 return jnp.dot(self._cov.mv(x-self._mean), v)
#
#         msg = "unsupported mean provided"
#         raise ValueError(msg)
#
#     def _logpdf_hvp_mean(
#         self,
#         x: Scalar | Vector,
#         v: Scalar | Vector
#     ) -> Scalar | Vector:
#         if jnp.isscalar(self._mean):
#             assert jnp.isscalar(self._cov) and jnp.isscalar(x)
#             return -v/self._cov
#         elif isinstance(self._mean, Vector):
#             if jnp.isscalar(self._cov) or isinstance(self._cov, Vector):
#                 return -jnp.reciprocal(self._cov)*v
#             elif isinstance(self._cov, LinearOperator):
#                 return -self._cov.solve(v)
#
#         msg = "unsupported mean provided"
#         raise ValueError(msg)
