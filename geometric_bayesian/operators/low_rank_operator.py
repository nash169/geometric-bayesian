#!/usr/bin/env python
# encoding: utf-8

from geometric_bayesian.operators.dense_operator import DenseOperator
import jax.numpy as jnp

from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.utils.types import Size, Scalar, Vector, Matrix, Optional, Self


class LowRankOperator(LinearOperator):
    def __init__(
        self,
        diag: Vector,
        right: Matrix,
        left: Optional[Matrix] = None,
        zero_tol: Scalar = 1e-8,
        jitter: Optional[Scalar] = None
    ) -> None:
        self.diag = diag
        self.right = right
        self.left = left if left is not None else self.right
        self.zero_tol = zero_tol

    def size(self) -> Size:
        r"""
        Return size of the linear operator
        """
        return self.right.shape[0], self.left.shape[0]

    def transpose(
        self,
    ) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        return LowRankOperator(diag=self.diag, right=self.left, left=self.right, zero_tol=self.zero_tol, jitter=self.jitter)

    def mv(self, vec: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        res = self.right @ ((self.left.T @ vec) * self.diag)
        if self.jitter is not None:
            res += self.jitter * (vec - self.right @ (self.left.T @ vec))
        return res

    def solve(
        self,
        vec: Vector,
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        inv_d = jnp.where(self.diag <= self.zero_tol, 0.0, jnp.reciprocal(self.diag))
        res = self.right @ ((self.left.T @ vec) * inv_d)
        if self.jitter is not None:
            res += (vec - self.right @ (self.left.T @ vec)) / self.jitter
        return res
        # return self.right @ ((self.left.T @ vec) / self.diag)

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
    ) -> DenseOperator:
        r"""
        Return dense matrix representation of the linear operator
        """
        # return (self.right * self.diag) @ self.left.T
        return DenseOperator((self.right * self.diag) @ self.left.T)

    def inverse(
            self
    ) -> LinearOperator:
        inv_d = jnp.where(self.diag <= self.zero_tol, 0.0, jnp.reciprocal(self.diag))
        # return LowRankOperator(diag=inv_d, right=self.right, left=self.left, zero_tol=self.zero_tol)
        return LowRankOperator(diag=jnp.flip(inv_d), right=jnp.flip(self.right, axis=1), left=jnp.flip(self.left, axis=1), zero_tol=self.zero_tol, jitter=self.jitter if self.jitter is None else 1 / self.jitter)

    # square root
    def sqrt(
            self
    ) -> LinearOperator:
        sqrt_d = jnp.where(self.diag <= self.zero_tol, 0.0, jnp.sqrt(self.diag))
        return LowRankOperator(diag=sqrt_d, right=self.right, left=self.left, zero_tol=self.zero_tol, jitter=self.jitter)

    # square root factor
    def sqrtf(
            self
    ) -> DenseOperator:
        sqrt_d = jnp.where(self.diag <= self.zero_tol, 0.0, jnp.sqrt(self.diag))
        return DenseOperator(self.right * sqrt_d)

    def topcut(
        self,
        num_modes
    ) -> Self:
        self.diag, self.right, self.left = self.diag[:num_modes], self.right[:, :num_modes], self.left[:, :num_modes]
        return self

    def bottomcut(
        self,
        num_modes
    ) -> Self:
        self.diag, self.right, self.left = self.diag[-num_modes:], self.right[:, -num_modes:], self.left[:, -num_modes:]
        return self
