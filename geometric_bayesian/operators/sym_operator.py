#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from geometric_bayesian.operators.linear_operator import LinearOperator
from geometric_bayesian.utils.types import Matrix, Scalar, Tuple, Optional, Key, Vector, Int, Float

from geometric_bayesian.linalg.eigvec_tridiagonal import eigvec_tridiagonal
from geometric_bayesian.linalg.lanczos import lanczos
from jax.experimental.sparse.linalg import lobpcg_standard


class SymOperator(LinearOperator):
    def transpose(self) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the symmetric operator
        """
        return self

    # def diagonalize(self, **kwargs) -> tuple[Vector, Matrix]:
    #     from geometric_bayesian.linalg.diagonalize import diagonalize
    #     dim = self.shape[0]
    #     if not isinstance(dim, int):
    #         dim = jnp.array(dim, int)  # force conversion to Python int, if shape is static
    #     return diagonalize(self, self.shape[0], **kwargs)

    def diagonalize(
        self,
        rng_key: Optional[Key] = None,
        method: Optional[str] = 'symeig',
        num_modes: Optional[Int] = None,
        num_iterations: Optional[Int] = 100,
        tol: Optional[Float] = None
    ) -> Tuple[Vector, Matrix]:
        r"""
        Return eigenvalues of the linear operator
        Eigenvectors calculation not available (https://github.com/jax-ml/jax/issues/14019)
        """
        if method == 'symeig':
            d, v = jnp.linalg.eigh(self.dense()._mat)
        elif method == 'lanczos':
            assert num_modes is not None
            key, subkey = jax.random.split(rng_key if rng_key is not None else jax.random.key(0))
            alpha, beta, v = lanczos(self, self.shape[0], num_modes, key)
            d = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True)
            v = jnp.matmul(v.T, eigvec_tridiagonal(subkey, alpha, beta, d))
        elif method == 'lobpcg':
            assert num_modes is not None and num_iterations is not None
            mv_batch = jax.vmap(self, in_axes=(1,), out_axes=1)
            d, v, _ = lobpcg_standard(mv_batch, jax.random.uniform(rng_key if rng_key is not None else jax.random.key(0),
                                      shape=(self.shape[0], num_modes)), m=num_iterations, tol=tol)
        else:
            msg = "provide valid method ['lanczos', 'lobpcg']"
            ValueError(msg)
        return d, v

    # def diagonalize(
    #     self,
    #     rng_key: Optional[Key] = None,
    #     num_modes: Optional[int] = None,
    #     num_iterations: Optional[int] = None,
    #     method: Optional[str] = None
    # ) -> Tuple[Vector, Matrix]:
    #     r"""
    #     Return eigenvalues of the linear operator
    #     Eigenvectors calculation not available (https://github.com/jax-ml/jax/issues/14019)
    #     """
    #     if rng_key is None:
    #         rng_key = jax.random.key(0)
    #     if num_modes is None:
    #         num_modes = (0.3 * self.shape[0]).astype(int)
    #     # assert num_modes <= self.size()[0], f"Number of modes requested exceeds opeartor dimension [{self.size()[0]}]"
    #     if num_iterations is None:
    #         num_iterations = 10
    #     if method is None:
    #         method = 'lanczos'
    #
    #     if method == 'lanczos':
    #         alpha, beta, v = lanczos(self.mv, self.size()[0], num_modes, rng_key)
    #         d = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True)
    #         v = jnp.matmul(v.T, eigvec_tridiagonal(rng_key, alpha, beta, d))
    #     elif method == 'lobpcg':
    #         assert num_iterations is not None
    #         mv_batch = jax.vmap(self.mv, in_axes=(1,), out_axes=1)
    #         d, v, _ = lobpcg_standard(mv_batch, jax.random.uniform(rng_key, shape=(self.size()[0], num_modes)), m=num_iterations)
    #     else:
    #         msg = "provide valid method ['lanczos', 'lobpcg']"
    #         ValueError(msg)
    #
    #     return d, v

    def logdet(
        self,
        **kwargs
    ) -> Scalar:
        r"""
        Return log determinat via stocastich Lanczos quadrature
        """
        return jnp.sum(jnp.log(self.diagonalize(**kwargs)[0]))
