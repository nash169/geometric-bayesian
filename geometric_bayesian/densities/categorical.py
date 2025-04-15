#!/usr/bin/env python
# encoding: utf-8

import math
import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Int, Scalar, Vector, Optional
from geometric_bayesian.operators.linear_operator import LinearOperator


class Categorical:
    def __init__(
        self,
        mu: Vector,
        logits: Optional[bool] = True,
    ) -> None:
        r"""
        Define categorical distribution.
        """
        self._mu = mu

    def __call__(
        self,
        x: Int
    ) -> Scalar:
        return self._mu[x]
