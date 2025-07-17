#!/usr/bin/env python
# encoding: utf-8
# Slightly modified version from: https://github.com/google/spectral-density/blob/8330354e55d42535b13b2e1c618f11904582355c/jax/lanczos.py
# Main changes: 1) returns 1D array of main and lower/upper diagonal elements instead of tridiagonal matrix
#               2) update to new jax syntax function array modification

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as jnp
import jax.random as random


# TODO(gilmer) This function should use higher numerical precision?
def lanczos(mv, dim, order, rng_key):
    """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

    This function applies Lanczos algorithm of a given order.  This function
    does full reorthogonalization.

    WARNING: This function may take a long time to jit compile (e.g. ~3min for
    order 90 and dim 1e7).

    Args:
      mv: Maps v -> Hv for a real symmetric matrix H.
        Input/Output must be of shape [dim].
      dim: Matrix H is [dim, dim].
      order: An integer corresponding to the number of Lanczos steps to take.
      rng_key: The jax PRNG key.

    Returns:
      d: main diagonal elements of the tridiagonal matrix of size (order)
      e: lower/upper diagonal elements of the tridiagonal matrix of size (order - 1)
      vecs: A numpy array of size (order, dim) corresponding to the Lanczos
        vectors.
    """

    d, e = jnp.zeros(order), jnp.zeros(order-1)
    vecs = jnp.zeros((order, dim))

    init_vec = random.normal(rng_key, shape=(dim,))
    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0].set(init_vec)

    beta = 0
    # TODO(gilmer): Better to use lax.fori loop for faster compile?
    for i in range(order):
        v = vecs[i, :].reshape((dim))
        if i == 0:
            v_old = 0
        else:
            v_old = vecs[i - 1, :].reshape((dim))

        w = mv(v)
        assert (w.shape[0] == dim and len(w.shape) == 1), ('Output of mv(v) must be of shape [dim].')
        w = w - beta * v_old

        alpha = jnp.dot(w, v)
        d = d.at[i].set(alpha)
        w = w - alpha * v

        # Full Reorthogonalization
        for j in range(i):
            tau = vecs[j, :].reshape((dim))
            coeff = jnp.dot(w, tau)
            w += -coeff * tau

        beta = jnp.linalg.norm(w)

        # TODO(gilmer): The tf implementation raises an exception if beta < 1e-6
        # here. However JAX cannot compile a function that has an if statement
        # that depends on a dynamic variable. Should we still handle this base?
        # beta being small indicates that the lanczos vectors are linearly
        # dependent.

        if i + 1 < order:
            e = e.at[i].set(beta)
            vecs = vecs.at[i+1].set(w/beta)

    return d, e, vecs
