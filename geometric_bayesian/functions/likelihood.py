#!/usr/bin/env python
# encoding: utf-8

from collections.abc import Callable
import jax

from geometric_bayesian.utils.types import Vector


def neg_logll(
    p: Callable,
    y: Vector,
    f: Vector
):
    r"""
    Calculate mapping from latent function values f to oberved labels via density p.

    Args:
        p: Probability density
        f: Latent function values
        y: Observed labels

    Returns:
        Negative log-likelihood
    """
    return jax.vmap(lambda y, f: -p(f)._log(y), in_axes=(0, 0))(y, f).sum()


def neg_logll_jvp(
    p: Callable,
    y: Vector,
    f: Vector,
    v: Vector
):
    r"""
    Calculate differential of the mapping from latent function values f to oberved labels via density p.

    Args:
        p: Probability density
        y: Observed labels
        f: Latent function values
        v: Tangent vector

    Returns:
        JVP of Negative log-likelihood
    """
    return jax.vmap(lambda y, f, v: -p(f)._jvp_params()[0](y, v), in_axes=(0, 0, 0))(y, f, v)


def neg_logll_hvp(
    p: Callable,
    y: Vector,
    f: Vector,
    v: Vector
):
    r"""
    Calculate hessian of the mapping from latent function values f to oberved labels via density p.

    Args:
        p: Probability density
        y: Observed labels
        f: Latent function values
        v: Tangent vector

    Returns:
        HVP of Negative log-likelihood
    """
    return jax.vmap(lambda y, f, v: -p(f)._hvp_params()[0](y, v), in_axes=(0, 0, 0))(y, f, v)
