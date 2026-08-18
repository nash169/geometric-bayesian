#!/usr/bin/env python
# encoding: utf-8

from typing import Self
from bayax.operators.linear_operator import LinearOperator
from bayax.operators.sym_operator import SymOperator
from bayax.utils.types import Size, Scalar, Array, Vector, Matrix, Optional

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class DiagOperator(SymOperator):
    def __init__(
        self,
        diag: Scalar | Vector,
        dim: Optional[int] = None
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        if isinstance(diag, Scalar):
            assert (dim is not None)
            self._dim = dim
        elif isinstance(diag, Array) and diag.ndim == 1:
            self._dim = len(diag)
        else:
            raise NotImplementedError(f"Type {type(diag)} not a valid diag operator.")

        self.diag = diag

    def tree_flatten(self):
        return (self.diag,), {"dim": self._dim}

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(diag=children[0], dim=aux["dim"])

    def size(
        self
    ) -> Size:
        r"""
        Return size of the linear operator
        """
        return self._dim, self._dim

    def mv(
        self,
        vec: Vector
    ) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return self.diag * vec

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return vec / self.diag

    def dense(
        self,
    ) -> LinearOperator:
        r"""
        Return dense matrix representation of the linear operator
        """
        from bayax.operators.dense_operator import DenseOperator
        return DenseOperator(jnp.diag(self.diag) if isinstance(self.diag, Vector) else self.diag * jnp.eye(self._dim))

    def logdet(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return self._dim * jnp.log(self.diag) if isinstance(self.diag, Scalar) else jnp.sum(jnp.log(self.diag))

    def inverse(
        self,
    ) -> LinearOperator:
        r"""
        Return inverse operator
        """
        return DiagOperator(diag=1 / self.diag, dim=self._dim)

    def invquad(
        self,
        vec: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return vec.T @ (vec / self.diag)

    def sqrt(
            self
    ) -> LinearOperator:
        return DiagOperator(diag=jnp.sqrt(self.diag), dim=self._dim)

    def sqrtf(
            self
    ) -> LinearOperator:
        return self.sqrt()
