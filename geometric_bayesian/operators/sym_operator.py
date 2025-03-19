#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.utils.types import Scalar, Vector
from geometric_bayesian.utils.lanczos import lanczos
from jax.scipy.linalg import eigh_tridiagonal
from jax.experimental.sparse.linalg import lobpcg_standard


class SymOperator(LinearOperator):
    def transpose(self) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the symmetric operator
        """
        return self

    def diagonalize(
        self,
        num_iterations: int,
        rng_key
    ) -> Vector:
        r"""
        Return eigenvalues of the linear operator
        Eigenvectors calculation not available (https://github.com/jax-ml/jax/issues/14019)
        """
        # lobpcg_standard(op, jax.random.uniform(rng_key, (op.size()[0], 1)), m=100)
        d, e, v = lanczos(self.mv, self.size()[0], num_iterations, rng_key)

        return eigh_tridiagonal(d, e, eigvals_only=True)

    def log_det(
        self,
        num_iteration: int,
        rng_key
    ) -> Scalar:
        r"""
        Return log determinat via stocastich Lanczos quadrature
        """
        return jnp.sum(jnp.log(self.diagonalize(num_iteration, rng_key)))
