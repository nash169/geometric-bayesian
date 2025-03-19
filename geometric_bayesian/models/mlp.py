#!/usr/bin/env python
# encoding: utf-8


import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Optional
from geometric_bayesian.utils.types import Size


class MLP(nnx.Module):
    def __init__(
        self,
        layers: List[int],
        rngs: nnx.Rngs,
        prob_out: Optional[bool] = False
    ):
        self.layers = [nnx.Linear(m, n, rngs=rngs) for m, n in zip(layers[:-1], layers[1:])]
        if prob_out:
            self.prob_wrap = nnx.sigmoid if layers[-1] == 1 else nnx.softmax

    def __call__(self, x: jax.Array):
        for layer in self.layers[:-1]:
            x = nnx.tanh(layer(x))
        x = self.layers[-1](x)
        if hasattr(self, "prob_wrap"):
            x = self.prob_wrap(x)
        return x

    @property
    def shape(self) -> Size:
        r"""
        Return size of the linear operator
        """
        return jnp.array([self.layers[0].in_features, self.layers[-1].out_features])
