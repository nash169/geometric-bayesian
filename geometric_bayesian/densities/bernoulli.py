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
        Define Bernoulli distribution.
        """
        self._mu, self._logits = mu, logits

    def __call__(
        self,
        x: Scalar
    ) -> Scalar:
        log_sig_mu = jax.nn.log_sigmoid(self._mu) if self._logits else jnp.log(self._mu)
        log_sig_not_mu = jax.nn.log_sigmoid(-self._mu) if self._logits else jnp.log(1 - self._mu)
        return x * log_sig_mu + (1 - x) * log_sig_not_mu

    def jvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        log_sig_mu = jax.nn.log_sigmoid(self._mu) if self._logits else jnp.log(self._mu)
        log_sig_not_mu = jax.nn.log_sigmoid(-self._mu) if self._logits else jnp.log(1 - self._mu)
        return (log_sig_mu - log_sig_not_mu) * v

    def hvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return jnp.array([0.0])

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
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        log_dev_mu = x - jax.nn.sigmoid(self._mu) if self._logits else x / self._mu - (1 - x) / (1 - self._mu)
        return log_dev_mu * v

    def _hvp_mu(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        log_hess_mu = -jax.nn.sigmoid(self._mu) * (1 - jax.nn.sigmoid(self._mu)) if self._logits else - x / self._mu**2 - (1 - x) / (1 - self._mu)**2
        return log_hess_mu * v
