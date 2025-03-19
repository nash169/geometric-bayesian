#!/usr/bin/env python
# encoding: utf-8

from typing import Self
from geometric_bayesian.operators.sym_operator import SymOperator
from geometric_bayesian.utils.types import Size, Scalar, Array, Vector, Matrix, Optional

import jax
import jax.numpy as jnp


class DiagOperator(SymOperator):
    def __init__(
        self,
        op_diag: Scalar | Vector,
        op_size: Optional[int] = None
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        if isinstance(op_diag, Scalar):
            assert (op_size is not None)
            self._op_size = op_size
        self._op_diag = op_diag

    def size(
        self
    ) -> Size:
        r"""
        Return size of the linear operator
        """
        jnp.array([self._op_size, self._op_size])

    def mv(
        self,
        vec: Vector
    ) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return self._op_diag*vec

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return vec/self._op_diag

    def to_dense(
        self,
    ) -> Self:
        r"""
        Return dense matrix representation of the linear operator
        """
        from geometric_bayesian.operators.dense_operator import DenseOperator
        return DenseOperator(jnp.diag(self._op_diag) if isinstance(self._op_diag, Vector) else self._op_diag*jnp.eye(self._op_size))

    def det(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return jnp.prod(self._op_diag)

    def logdet(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return jnp.sum(jnp.log(self._op_diag))
