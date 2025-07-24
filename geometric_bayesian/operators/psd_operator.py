#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from geometric_bayesian.utils.types import Scalar, Vector, Matrix, Optional, Callable, Array, VectorInt
from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.operators import DenseOperator, SymOperator


class PSDOperator(SymOperator):
    def __init__(
        self,
        op: Optional[Matrix | Callable] = None,
        op_type: Optional[str] = None,
        op_size: Optional[int] = None,
        rng_key: Optional[Array] = None
    ) -> None:
        r"""
        mat_type: ['raw', 'tril', 'triu']
        """
        if op is not None:
            if isinstance(op, jax.Array):
                assert (op_type is not None), "Matrix provided but type not defined [ 'raw', 'tril', 'triu' ]"
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
                self._op_size = op.shape[0]
            elif isinstance(op, Callable):
                assert (op_size is not None), "Matrix-free operator provided; define operator dimension."
                self._op = op
                self._op_size = op_size
            else:
                msg = "invalid operator [ Matrix | Callable ]"
        else:
            assert (op_size is not None), "No operator provided; define operator dimension to generate random PSD."
            if rng_key is None:
                rng_key = jax.random.key(0)
            mat = jax.random.uniform(jax.random.split(rng_key)[1], shape=(op_size, op_size))
            mat = mat + mat.T + 100 * jnp.eye(op_size)
            self._op = jnp.linalg.cholesky(mat)
            self._op_is_tril = True

    def size(self) -> VectorInt:
        r"""
        Return size of the linear operator
        """
        return jnp.array([self._op_size, self._op_size]) if isinstance(self._op, Callable) else jnp.array([self._op.shape[0], self._op.shape[1]])

    def mv(self, vec: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return self._op(vec) if isinstance(self._op, Callable) else jnp.matmul(self._op, jnp.matmul(jnp.transpose(self._op), vec))

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return cg(lambda v: self.mv(v), vec, **kwargs)[0] if isinstance(self._op, Callable) else jax.scipy.linalg.cho_solve((self._op, self._op_is_tril), vec)

    def logdet(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return super().logdet() if isinstance(self._op, Callable) else 2 * jnp.sum(jnp.log(jnp.diag(self._op)))

    def invquad(
        self,
        vec: Vector
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return super().invquad(vec) if isinstance(self._op, Callable) else jnp.sum(jnp.pow(jax.scipy.linalg.solve_triangular(self._op, vec, lower=True), 2))

    def dense(
        self,
    ) -> LinearOperator:
        r"""
        Return dense matrix representation of the linear operator
        """
        return super().dense() if isinstance(self._op, Callable) else DenseOperator(self._op @ self._op.T)

    def lowrank(
        self,
        **kwargs
    ) -> LinearOperator:
        from geometric_bayesian.operators.low_rank_operator import LowRankOperator
        eigval, eigvec = self.diagonalize(**kwargs)
        return LowRankOperator(diag=eigval, right=eigvec)

    def squareroot(
        self,
        **kwargs
    ) -> LinearOperator:
        if isinstance(self._op, Callable):
            if self._op_size <= 100:
                from geometric_bayesian.operators import DenseOperator
                return DenseOperator(jnp.linalg.cholesky(self.dense()._mat))
            else:
                return self.lowrank().squareroot()
        else:
            from geometric_bayesian.operators import DenseOperator
            return DenseOperator(self._op)
