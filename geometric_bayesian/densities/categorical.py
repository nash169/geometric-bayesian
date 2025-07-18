#!/usr/bin/env python
# encoding: utf-8

import math
import jax
import jax.numpy as jnp
from mpl_toolkits.axes_grid1.axes_size import Scaled

from geometric_bayesian.utils.types import ScalarInt, Scalar, Vector, Optional, Tuple
from geometric_bayesian.operators.linear_operator import LinearOperator


class Categorical:
    def __init__(
        self,
        mu: Vector,
        logits: Optional[bool] = True
    ) -> None:
        r"""
        Define categorical distribution.
        """
        self._mu, self._logits = mu, logits

    def __call__(
        self,
        x: ScalarInt | Vector
    ) -> Scalar:
        log = jax.nn.log_softmax(self._mu) if self._logits else jnp.log(self._mu)
        return log[x] if isinstance(x, ScalarInt) else jnp.sum(x * log)

    def jvp(
        self,
        x: ScalarInt | Vector,
        v: Vector
    ) -> Scalar:
        grad = jax.nn.log_softmax(self._mu) if self._logits else jnp.log(self._mu)
        return grad @ v

    def hvp(
        self,
        x: ScalarInt | Vector,
        v: Vector
    ) -> Vector:
        return jnp.zeros_like(self._mu)

    def jvp_params(
        self,
    ) -> Tuple:
        return (lambda x, v, : self._jvp_mu(x, v),)

    def hvp_params(
        self,
    ) -> Tuple:
        return (lambda x, v: self._hvp_mu(x, v),)

    def _jvp_mu(
        self,
        x: ScalarInt | Vector,
        v: Vector
    ) -> Scalar:
        if isinstance(x, ScalarInt):
            x = jax.nn.one_hot(x, self._mu.shape[0])
        grad = x - jax.nn.softmax(self._mu) if self._logits else x / self._mu
        return grad @ v

    def _hvp_mu(
        self,
        x: ScalarInt | Vector,
        v: Vector
    ) -> Scalar:
        if isinstance(x, ScalarInt):
            x = jax.nn.one_hot(x, self._mu.shape[0])
        if self._logits:
            prob = jax.nn.softmax(self._mu)
            return prob * (prob @ v) - prob * v
        else:
            return -x * jnp.pow(self._mu, -2) * v
