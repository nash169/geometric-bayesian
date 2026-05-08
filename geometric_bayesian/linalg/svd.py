#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
from jaxtyping import Array


def svd():
    pass


def _svd_lanczos(
    k: int,
    alpha: Array,
    beta: Array,
    U: Array,
    V: Array,
):
    """
    Recover approximate singular triplets from Lanczos bidiagonalization.

    The bidiagonalization stores ``m`` left vectors in ``U`` and ``m + 1``
    right vectors in ``V``. The projected square bidiagonal matrix uses the
    diagonal ``alpha`` and the first ``m - 1`` entries of ``beta``.
    """
    B = jnp.diag(alpha) + jnp.diag(beta[:-1], k=1)
    Ub, s, Vhb = jnp.linalg.svd(B, full_matrices=False)

    idx = jnp.argsort(s, descending=True)[:k]
    s = s[idx]

    left = U.T @ Ub[:, idx]
    right = V[:-1].T @ Vhb.T[:, idx]

    left = left / jnp.maximum(jnp.linalg.norm(left, axis=0, keepdims=True), 1e-12)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=0, keepdims=True), 1e-12)

    return s, left, right
