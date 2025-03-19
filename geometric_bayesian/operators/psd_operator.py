#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from matplotlib.pyplot import isinteractive

from geometric_bayesian.utils.types import Size, Scalar, Vector, Matrix, VectorFn, Optional, Callable
from geometric_bayesian.operators.sym_operator import SymOperator
from jax.scipy.sparse.linalg import cg


class PSDOperator(SymOperator):
    def __init__(
        self,
        op: Matrix | Callable,
        op_type: Optional[str] = None,
        op_size: Optional[int] = None
    ) -> None:
        r"""
        mat_type: ['raw', 'tril', 'triu']
        """
        if isinstance(op, Matrix):
            assert (op_type is not None)
            if op_type == 'raw':
                self._op = jnp.linalg.cholesky(op)
                self._op_is_tril = True
            elif op_type == 'tril':
                self._op = op
                self._op_is_tril = True
            elif op_type == 'triu':
                self._op = op
                self._op_is_tril = False
            else:
                msg = "invalid operator type [ 'raw', 'tril', 'triu' ]"
                raise ValueError(msg)
        elif isinstance(op, Callable):
            assert (op_size is not None)
            self._op = op
            self._op_size = op_size
        else:
            msg = "invalid operator [ Matrix | VectorFn ]"

    def size(self) -> Size:
        r"""
        Return size of the linear operator
        """
        return jnp.array([self._op.shape[0], self._op.shape[1]]) if isinstance(self._op, Matrix) else jnp.array([self._op_size, self._op_size])

    def mv(self, vec: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return jnp.matmul(self._op, jnp.matmul(jnp.transpose(self._op), vec)) if isinstance(self._op, Matrix) else self._op(vec)

    def transpose(self) -> SymOperator:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        return self

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return jax.scipy.linalg.cho_solve((self._op, self._op_is_tril), vec) if isinstance(self._op, Matrix) else cg(lambda v: self.mv(v), vec, **kwargs)[0]

    def det(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return jnp.prod(jnp.diag(self._op))

    def inv_quad(
        self,
        vec: Vector
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return jnp.sum(jnp.pow(jax.scipy.linalg.solve_triangular(self._op, vec, lower=True), 2))

    def dense_operator(
        self,
    ) -> Matrix:
        r"""
        Return dense matrix representation of the linear operator
        """
        return jnp.matmul(self._op, jnp.transpose(self._op))
