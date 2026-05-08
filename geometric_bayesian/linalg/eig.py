#!/usr/bin/env python

# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from geometric_bayesian.utils.types import Optional, Callable, Array, Tuple, Vector, Matrix
from geometric_bayesian.linalg.lanczos import lanczos_tridiag
from jax.experimental.sparse.linalg import lobpcg_standard


def eigh(
    mv: Array | Callable[[Array], Array],
    n: int,
    *,
    method: str = 'symeig',
    k: Optional[int] = None,
    m: Optional[int] = None,
    tol: float = 1e-4,
    key: Optional[Array] = None,
    **kwargs
) -> Tuple[Vector, Matrix]:
    if method == 'symeig':
        d, v = jnp.linalg.eigh(jax.vmap(mv, in_axes=(1,), out_axes=1)(jnp.eye(n)) if isinstance(mv, Callable) else mv)
        d, v = jnp.flip(d), jnp.flip(v, axis=1)
        if k is not None:
            d, v = d[:k], v[:, :k]
    elif method == 'lanczos':
        assert k is not None
        key, subkey = jax.random.split(key if key is not None else jax.random.key(0))
        alpha, beta, Q = lanczos_tridiag(mv, n, m=m if m is not None else k, key=key, **kwargs)
        d, v = _eig_lanczos(k, alpha, beta, Q, descending=True, key=subkey)
    elif method == 'lobpcg':
        assert k is not None
        d, v, _ = lobpcg_standard(jax.vmap(mv, in_axes=(1,), out_axes=1), jax.random.uniform(
            key if key is not None else jax.random.key(0), shape=(n, k)), m=m if m is not None else k, tol=tol)
    else:
        msg = "provide valid method ['symeig', 'lanczos', 'lobpcg']"
        ValueError(msg)
    return d, v


def _eig_lanczos(
    k: int,
    alpha: Array,
    beta: Array,
    Q: Array,
    *,
    descending: bool = True,
    key: Optional[Array] = None
):
    evals, evecs = jsp.eigh_tridiagonal(alpha, beta, eigvals_only=False, key=key)

    idx = jnp.argsort(evals, descending=descending)[:k]
    evals = evals[idx]

    evecs = Q.T @ evecs[:, idx]
    evecs = evecs / jnp.linalg.norm(evecs, axis=0, keepdims=True)

    return evals, evecs
