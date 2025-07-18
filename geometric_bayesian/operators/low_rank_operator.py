#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp

from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.utils.types import Size, Scalar, Vector, Matrix, Optional


class LowRankOperator(LinearOperator):
    def __init__(
        self,
        diag: Vector,
        right: Matrix,
        left: Optional[Matrix] = None
    ) -> None:
        self.diag = diag
        self.right = right
        self.left = left if left is not None else self.right

    def size(self) -> Size:
        r"""
        Return size of the linear operator
        """
        return jnp.array([self.right.shape[0], self.left.shape[0]])

    def transpose(
        self,
    ) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        return LowRankOperator(self.diag, self.left, self.right)

    def mv(self, vec: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return self.right @ ((self.left.T @ vec) * self.diag)

    def solve(
        self,
        vec: Vector,
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return self.right @ ((self.left.T @ vec) / self.diag)

    def logdet(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return jnp.sum(jnp.log(self.diag))

    def invquad(
        self,
        vec: Vector
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return vec @ self.solve(vec)

    def dense(
        self,
    ) -> Matrix:
        r"""
        Return dense matrix representation of the linear operator
        """
        return (self.right * self.diag) @ self.left.T

    def inverse(
            self
    ) -> LinearOperator:
        return LowRankOperator(diag=jnp.reciprocal(self.diag), right=self.right, left=self.left)

    def squareroot(
            self
    ) -> LinearOperator:
        return LowRankOperator(diag=jnp.sqrt(self.diag), right=self.right, left=self.left)
