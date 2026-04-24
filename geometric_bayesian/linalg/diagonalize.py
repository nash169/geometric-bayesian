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
    method: str = 'symeig',
    num_modes: Optional[Int] = None,
    num_iters: Int = 100,
    tol: Float = 1e-4,
) -> Tuple[Vector, Matrix]:
    if method == 'symeig':
        d, v = jnp.linalg.eigh(jax.vmap(mv, in_axes=(1,), out_axes=1)(jnp.eye(dim)))
        d, v = jnp.flip(d), jnp.flip(v, axis=1)
        if num_modes is not None:
            d, v = d[:num_modes], v[:, :num_modes]
    elif method == 'lanczos':
        assert num_modes is not None
        key, subkey = jax.random.split(rng_key if rng_key is not None else jax.random.key(0))
        alpha, beta, v = lanczos(mv, dim, num_modes, key)
        d = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True)
        v = jnp.matmul(v.T, eigvec_tridiagonal(subkey, alpha, beta, d))  # https://github.com/jax-ml/jax/issues/14019
    elif method == 'lobpcg':
        assert num_modes is not None
        d, v, _ = lobpcg_standard(jax.vmap(mv, in_axes=(1,), out_axes=1), jax.random.uniform(
            rng_key if rng_key is not None else jax.random.key(0), shape=(dim, num_modes)), m=num_iters, tol=tol)
    else:
        msg = "provide valid method ['symeig', 'lanczos', 'lobpcg']"
        ValueError(msg)
    return d, v
