
#!/usr/bin/env python
# encoding: utf-8

from typing import Self
from ..types import Size, Scalar, Array, Vector, Matrix
from .linear_operator import LinearOperator

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


class SumOperator(LinearOperator):
    def __init__(
        self,
        *ops: LinearOperator
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        self._ops = ops  # check that all the operators have the same dimension

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
        y = jnp.zeros(self.size()[0])
        for op in self._ops:
            y += op.solve(vec, **kwargs)
        return y

    def to_dense(
        self,
    ) -> Matrix:
        r"""
        Return dense matrix representation of the linear operator
        """
        return jax.vmap(self.mv, in_axes=0)(jnp.eye(self.size()[1]))
