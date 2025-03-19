#!/usr/bin/env python
# encoding: utf-8

import math
import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Int, Scalar, Vector


class Bernoulli:
    def __init__(
        self,
        mu: Scalar,
    ) -> None:
        r"""
        Define categorical distribution.
        """
        # assert (mu <= 1.0)
        self._mu = mu

    def __call__(
        self,
        x: Scalar
    ) -> Scalar:
        return self._mu[x]

    def _logpdf(
        self,
        x: Scalar
    ) -> Scalar:
        return x*jnp.log(self._mu) + (1-x)*jnp.log(1-self._mu)

    def _logpdf_jvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return 0

    def _logpdf_hvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return 0

    def _logpdf_jvp_params(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return (x/self._mu - (1-x)/(1-self._mu))*v

    def _logpdf_hvp_params(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return -(x*jnp.power(self._mu, -2) + (1-x)*jnp.power(1-self._mu, 2))*v
