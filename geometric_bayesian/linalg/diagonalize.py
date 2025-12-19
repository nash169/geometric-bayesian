#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from geometric_bayesian.utils.types import Int, Key, Optional, Tuple, Vector, Matrix, Callable, Float
from geometric_bayesian.linalg.eigvec_tridiagonal import eigvec_tridiagonal
from geometric_bayesian.linalg.lanczos import lanczos
from jax.experimental.sparse.linalg import lobpcg_standard


def diagonalize(
    mv: Callable,
    dim: Int,
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
        mv_batch = jax.vmap(mv, in_axes=(1,), out_axes=1)
        d, v = jnp.linalg.eigh(mv_batch(jnp.eye(dim)))
        d, v = jnp.flip(d), jnp.flip(v, axis=1)
    elif method == 'lanczos':
        assert num_modes is not None
        key, subkey = jax.random.split(rng_key if rng_key is not None else jax.random.key(0))
        alpha, beta, v = lanczos(mv, dim, num_modes, key)
        d = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True)
        v = jnp.matmul(v.T, eigvec_tridiagonal(subkey, alpha, beta, d))
    elif method == 'lobpcg':
        assert num_modes is not None and num_iterations is not None
        mv_batch = jax.vmap(mv, in_axes=(1,), out_axes=1)
        d, v, _ = lobpcg_standard(mv_batch, jax.random.uniform(rng_key if rng_key is not None else jax.random.key(0),
                                  shape=(dim, num_modes)), m=num_iterations, tol=tol)
    else:
        msg = "provide valid method ['symeig', 'lanczos', 'lobpcg']"
        ValueError(msg)
    return d, v
