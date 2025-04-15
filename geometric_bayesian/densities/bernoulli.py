#!/usr/bin/env python
# encoding: utf-8

import math
import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Int, Scalar, Vector, Optional, Tuple


class Bernoulli:
    def __init__(
        self,
        mu: Scalar,
        logits: Optional[bool] = True
    ) -> None:
        r"""
        Define categorical distribution.
        """
        self._mu, self._logits = mu, logits

    def __call__(
        self,
        x: Scalar
    ) -> Scalar:
        return jnp.power(self._mu, x)*jnp.power(1-self._mu, 1-x)

    def _log(
        self,
        x: Scalar
    ) -> Scalar:
        log_mu = jax.nn.log_sigmoid(self._mu) if self._logits else jnp.log(self._mu)
        log_not_mu = jax.nn.log_sigmoid(-self._mu) if self._logits else jnp.log(1-self._mu)
        return x*log_mu + (1-x)*log_not_mu

    def _jvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        log_mu = jax.nn.log_sigmoid(self._mu) if self._logits else jnp.log(self._mu)
        log_not_mu = jax.nn.log_sigmoid(-self._mu) if self._logits else jnp.log(1-self._mu)
        return (log_mu - log_not_mu)*v

    def _hvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return jnp.array(0.0)

    def _jvp_mu(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return (x/self._mu - (1-x)/(1-self._mu))*v

    def _jvp_params(
        self,
    ) -> Tuple:
        return (lambda x, v, : self._jvp_mu(x, v),)

    def _hvp_mu(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return -(x*jnp.power(self._mu, -2) + (1-x)*jnp.power(1-self._mu, 2))*v

    def _hvp_params(
        self,
    ) -> Tuple:
        return (lambda x, v: self._hvp_mu(x, v),)
