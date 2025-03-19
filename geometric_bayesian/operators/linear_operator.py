#!/usr/bin/env python
# encoding: utf-8

from abc import ABC, abstractmethod
from typing import Self

from geometric_bayesian.utils.types import Size, Scalar, Array, Vector, Matrix

import jax
import jax.numpy as jnp


class LinearOperator(ABC):
    @abstractmethod
    def __init__(
        self,
        *args: Array
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        pass

    @abstractmethod
    def size(
        self
    ) -> Size:
        r"""
        Return size of the linear operator
        """
        pass

    @abstractmethod
    def mv(
        self,
        vec: Vector
    ) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        pass

    @abstractmethod
    def transpose(
        self,
    ) -> Self:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        pass

    def __call__(
        self,
        input: Vector | Matrix
    ) -> Vector:
        r"""
        Overload () operator
        """
        return self.mv(input) if isinstance(input, Vector) else jax.vmap(self.mv, in_axes=0)(input)

    def __add__(
        self,
        other
    ):
        from geometric_bayesian.operators.sum_operator import SumOperator
        return SumOperator(self, other)

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return self.to_dense().solve(vec, **kwargs)

    def inv_quad(
        self,
        vec: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return jnp.dot(vec, self.solve(vec, **kwargs))

    def to_dense(
        self,
    ) -> Self:
        r"""
        Return dense matrix representation of the linear operator
        """
        from geometric_bayesian.operators.dense_operator import DenseOperator
        return DenseOperator(self(jnp.eye(self.size()[1])))

    # @abstractmethod
    # def det(
    #     self,
    # ) -> Scalar:
    #     r"""
    #     Return determinant of the linear operator
    #     """
    #     pass
