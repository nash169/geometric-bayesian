
#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from geometric_bayesian.utils.types import Size, Scalar, Array, Vector, Matrix, Self
from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.operators import *


class SumOperator(LinearOperator):
    def __init__(
        self,
        *ops: LinearOperator
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        assert all(ops[0].shape == ops[1].shape), "Error: cannot sum operators with different dimension."
        self._ops = ops

    def size(
        self
    ) -> Size:
        r"""
        Return size of the linear operator
        """
        return self._ops[0].size()

    def mv(
        self,
        vec: Vector
    ) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        y = jnp.zeros(self.size()[0])
        for op in self._ops:
            y += op.mv(vec)
        return y

    def transpose(
        self,
    ) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        return SumOperator(*[op.transpose() for op in self._ops])

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        if any(isinstance(op, LowRankOperator) for op in self._ops) and any(isinstance(op, DiagOperator) for op in self._ops):
            from geometric_bayesian.linalg.woodbury_solve import woodbury_solve
            lowrank_op = [op for op in self._ops if isinstance(op, LowRankOperator)][0]
            diag_op = [op for op in self._ops if isinstance(op, DiagOperator)][0]
            return woodbury_solve(lowrank_op.right, lowrank_op.diag, diag_op.diag, vec)
        elif any(isinstance(op, PSDOperator) for op in self._ops) and any(isinstance(op, DiagOperator) for op in self._ops):
            from geometric_bayesian.linalg.woodbury_solve import woodbury_chol_solve
            psd_op = [op for op in self._ops if isinstance(op, PSDOperator)][0]
            diag_op = [op for op in self._ops if isinstance(op, DiagOperator)][0]
            return woodbury_chol_solve(psd_op._op, diag_op.diag, vec)
        else:
            raise NotImplementedError(f"Method not implemented.")

    def diagonalize(
        self,
        **kwargs
    ) -> tuple[Vector, Matrix]:
        if any(issubclass(type(op), SymOperator) for op in self._ops):
            from geometric_bayesian.linalg.diagonalize import diagonalize
            return diagonalize(self.mv, self.shape[0], **kwargs)
        else:
            return super().diagonalize()

    def lowrank(
        self,
        **kwargs
    ) -> LinearOperator:
        from geometric_bayesian.operators.low_rank_operator import LowRankOperator
        if all(issubclass(type(op), (LowRankOperator, SymOperator)) for op in self._ops):
            eigval, eigvec = self.diagonalize(**kwargs)
            return LowRankOperator(diag=eigval, right=eigvec)
        else:
            return super().lowrank()

    def squareroot(
        self,
        **kwargs
    ) -> LinearOperator:
        if all(issubclass(type(op), (LowRankOperator, SymOperator)) for op in self._ops):
            if self.shape[0] * self.shape[1] <= 1e4:
                from geometric_bayesian.operators import DenseOperator
                return DenseOperator(jnp.linalg.cholesky(self.dense()._mat))
            else:
                return self.lowrank(**kwargs).squareroot()
        else:
            return super().squareroot()
