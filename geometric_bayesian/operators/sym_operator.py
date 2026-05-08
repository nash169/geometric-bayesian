#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.utils.types import Scalar, Vector, Matrix


class SymOperator(LinearOperator):
    def transpose(self) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the symmetric operator
        """
        return self

    def diagonalize(self, **kwargs) -> tuple[Vector, Matrix]:
        from geometric_bayesian.linalg.eig import eigh
        return eigh(self, self.shape[0], **kwargs)

    def logdet(
        self,
        **kwargs
    ) -> Scalar:
        r"""
        Return log determinat via stocastich Lanczos quadrature
        """
        return jnp.sum(jnp.log(self.diagonalize(**kwargs)[0]))
