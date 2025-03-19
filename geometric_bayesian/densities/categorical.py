#!/usr/bin/env python
# encoding: utf-8

import math
import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Int, Scalar, Vector
from geometric_bayesian.operators.linear_operator import LinearOperator


class Categorical:
    def __init__(
        self,
        mu: Vector,
    ) -> None:
        r"""
        Define categorical distribution.
        """
        assert (jnp.sum(mu) <= 1.0)
        self._mu = mu

    def __call__(
        self,
        x: Int
    ) -> Scalar:
        return self._mu[x]
