#!/usr/bin/env python
# encoding: utf-8

from abc import ABC, abstractmethod

import jax
from flax import nnx

from geometric_bayesian.utils.types import Vector, Matrix


class AbstractModel(ABC, nnx.Module):
    def model_fn(self):
        graph_def, params = nnx.split(self)

        def model_fwd(input, params):
            return nnx.call((graph_def, params))(input)[0]
        return model_fwd, params

    def jvp(self, input: Vector | Matrix, params: Vector) -> Vector | Matrix:
        model_fn, model_params = self.model_fn()
        return jax.jvp(
            lambda p: model_fn(input=input, params=p),
            (model_params,),
            (params,),
        )[1]

    def vjp(self, input: Vector | Matrix, output: Vector | Matrix) -> Vector | Matrix:
        model_fn, model_params = self.model_fn()
        vjp_fun = jax.vjp(
            lambda p: model_fn(input=input, params=p), model_params
        )[1]
        return vjp_fun(output)[0]
