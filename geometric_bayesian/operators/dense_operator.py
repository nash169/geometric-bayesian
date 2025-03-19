#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp

from geometric_bayesian.utils.types import Size, Scalar, Vector, Matrix
from geometric_bayesian.operators.linear_operator import LinearOperator


class DenseOperator(LinearOperator):
    def __init__(
        self,
        mat: Matrix
    ) -> None:
        self._mat = mat

    def __call__(self, vec: Vector) -> Vector:
        return self.mv(vec)

    def size(self) -> Size:
        r"""
        Return size of the linear operator
        """
        return jnp.array([self._mat.shape[0], self._mat.shape[1]])

    def mv(self, vec: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return jnp.matmul(self._mat, vec)

    def transpose(self) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        return DenseOperator(jnp.transpose(self._mat))

    def solve(
        self,
        vec: Vector
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return jnp.linalg.solve(self._mat, vec)

    def det(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return jnp.linalg.det(self._mat)

    def inv_quad(
        self,
        vec: Vector
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return jnp.matmul(jnp.transpose(vec), self.solve(vec))

    def dense_operator(
        self,
    ) -> Matrix:
        r"""
        Return dense matrix representation of the linear operator
        """
        return self._mat
