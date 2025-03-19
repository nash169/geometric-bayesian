#!/usr/bin/env python
# encoding: utf-8

import jax
from flax import nnx

from geometric_bayesian.utils.jax import array_to_pytree, pytree_to_array, pytree_size
from geometric_bayesian.utils.types import VectorFn, Vector, Matrix, PyTree, Optional, Int

# logll - model


def logll_fwd(X, y, pdf, **pdf_params):
    return lambda model: pdf(model(X), **pdf_params)._logpdf(y)


@jax.custom_jvp
def logll_fn(model_fn):
    return jax.vmap(lambda y, model_fn: -likelihood(model_fn)._logpdf(y), in_axes=(0, 0), out_axes=0)(targets, model_fn)


@logll_fn.defjvp
@jax.custom_jvp
def logll_jvp(primals, tangents):
    model_fn = primals[0]
    v = tangents[0]
    return logll_fn(model_fn), jax.vmap(lambda y, model_fn, v: -likelihood(model_fn)._logpdf_jvp_mean(y, v), in_axes=(0, 0, 0))(targets, model_fn, v)

# log-likelihood hessian-vector product


@logll_jvp.defjvp
def logll_hvp(primals, tangents):
    model_fn = primals[0]
    v = tangents[0]
    return logll_fn(model_fn), jax.vmap(lambda y, model_fn, v: -likelihood(model_fn)._logpdf_hvp_mean(y, v), in_axes=(0, 0, 0))(targets, model_fn, v)
