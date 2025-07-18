#!/usr/bin/env python
# encoding: utf-8

import warnings
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from geometric_bayesian.utils.types import Array, Self, Scalar, Vector, Matrix, VectorInt, Num, Optional, Int, Key, Tuple


class AbstractDensity(ABC):
    @abstractmethod
    def __init__(
        self,
        *args: Array
    ) -> None:
        r"""
        Define set of arrays needed for the probability density function
        """
        pass

    @abstractmethod
    def __call__(
        self,
        x: Num | Array
    ) -> Scalar:
        r"""
        Return density evaluation
        """
        raise NotImplementedError("The class {} requires a __call__ function.".format(self.__class__.__name__))

    def sample(
        self,
        rng_key: Optional[Key] = None,
        size: Optional[Int] = 1,
        **kwargs
    ) -> Array:
        raise NotImplementedError(f"Method not implemented.")

    def jvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Gradient with respect to the input.
        """
        raise NotImplementedError(f"Method not implemented.")

    def hvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Hessian with respect to the input.
        """
        raise NotImplementedError(f"Method not implemented.")

    def jvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for gradient function with respect to the params.
        """
        raise NotImplementedError(f"Method not implemented.")

    def hvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for hessian function with respect to the params.
        """
        raise NotImplementedError(f"Method not implemented.")
