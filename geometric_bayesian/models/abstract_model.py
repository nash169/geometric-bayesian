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


def hvp(func: Callable, primals: PyTree, tangents: PyTree) -> PyTree:
    r"""Compute the Hessian-vector product (HVP) for a given function.

    The Hessian-vector product is computed by differentiating the gradient of the
    function. This avoids explicitly constructing the Hessian matrix, making the
    computation efficient.

    Args:
        func: The scalar function for which the HVP is computed.
        primals: The point at which the gradient and Hessian are evaluated.
        tangents: The vector to multiply with the Hessian.

    Returns:
        The Hessian-vector product.
    """
    return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]
