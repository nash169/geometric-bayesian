#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from geometric_bayesian.utils.types import Int, Key, Optional, Tuple, Vector, Matrix, Callable
from geometric_bayesian.linalg.eigvec_tridiagonal import eigvec_tridiagonal
from geometric_bayesian.linalg.lanczos import lanczos
from jax.experimental.sparse.linalg import lobpcg_standard


def diagonalize(
    mv: Callable,
    dim: Int,
    rng_key: Optional[Key] = None,
    num_modes: Optional[Int] = None,
    num_iterations: Optional[Int] = None,
    method: Optional[str] = None
) -> Tuple[Vector, Matrix]:
    r"""
    Return eigenvalues of the linear operator
    Eigenvectors calculation not available (https://github.com/jax-ml/jax/issues/14019)
    """
    if rng_key is None:
        rng_key = jax.random.key(0)
    if num_modes is None:
        num_modes = (0.3 * dim).astype(int)
    # assert num_modes <= self.size()[0], f"Number of modes requested exceeds opeartor dimension [{self.size()[0]}]"
    if num_iterations is None:
        num_iterations = 10
    if method is None:
        # method = 'lanczos' if dim >= 1e3 else 'symeig'
        method = 'symeig'

    if method == 'symeig':
        d, v = jnp.linalg.eigh(mv.dense()._mat)
    elif method == 'lanczos':
        alpha, beta, v = lanczos(mv, dim, num_modes, rng_key)
        d = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True)
        v = jnp.matmul(v.T, eigvec_tridiagonal(rng_key, alpha, beta, d))
    elif method == 'lobpcg':
        assert num_iterations is not None
        mv_batch = jax.vmap(mv, in_axes=(1,), out_axes=1)
        d, v, _ = lobpcg_standard(mv_batch, jax.random.uniform(rng_key, shape=(dim, num_modes)), m=num_iterations)
    else:
        msg = "provide valid method ['lanczos', 'lobpcg']"
        ValueError(msg)

    return d, v
