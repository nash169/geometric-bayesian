#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from flax import nnx

from geometric_bayesian.utils.types import Size, List, Optional, Callable, Array, Int, Vector
from geometric_bayesian.utils.helper import pytree_to_array, array_to_pytree, wrap_pytree_function


class MLP(nnx.Module):
    def __init__(
        self,
        layers: List[int],
        nl: Optional[Callable] = nnx.tanh,
        prob: Optional[bool] = False,
        **kwargs
    ):
        self.layers = [nnx.Linear(m, n, rngs=nnx.Rngs(params=0), **kwargs) for m, n in zip(layers[:-1], layers[1:])]
        if nl is not None:
            self.nl = nl
        if prob:
            self.prob = nnx.sigmoid if layers[-1] == 1 else nnx.softmax

    def __call__(self, x: Array):
        for layer in self.layers[:-1]:
            x = self.nl(layer(x)) if hasattr(self, 'nl') else layer(x)
        x = self.layers[-1](x).squeeze()
        return self.prob(x) if hasattr(self, "prob") else x

    def features(self, x: Array, l: Int):
        assert l <= len(self.layers)
        if l == len(self.layers):
            return self(x)
        for i in range(l):
            x = self.nl(self.layers[i](x)) if hasattr(self, 'nl') else self.layers[i](x)
        return x

    def fwd_params(
        self
    ) -> Callable:
        graph_def, params = nnx.split(self)

        def fwd(x, p):
            return wrap_pytree_function(
                lambda p: nnx.call((graph_def, p))(x)[0],
                params
            )(p)
        return fwd

    @property
    def params(self) -> Vector:
        return pytree_to_array(nnx.state(self))

    @params.setter
    def params(self, value):
        nnx.update(self, value)
        # self = nnx.merge(nnx.split(self)[0], value)

    @property
    def size(self) -> Int:
        return sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(self)))

    @property
    def shape(self) -> Size:
        r"""
        Return network input-ouput dimensions
        """
        return jnp.array([self.layers[0].in_features, self.layers[-1].out_features])
