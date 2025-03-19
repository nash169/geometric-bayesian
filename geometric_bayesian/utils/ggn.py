#!/usr/bin/env python
# encoding: utf-8

import jax
from flax import nnx

from geometric_bayesian.utils.jax import array_to_pytree, pytree_to_array, pytree_size
from geometric_bayesian.utils.types import VectorFn, Vector, Matrix, PyTree, Optional, Int


def ggn(
    model,
    train_data,
    likelihood_density,  # e.g. lambda model_fn: Normal(model_fn, jnp.array(2.0))
    **likelihood_params,
) -> VectorFn:
    samples, targets = train_data

    # get model params
    graph_def, params = nnx.split(model)

    # model forward function wrt parameters
    def model_fn(params):
        return nnx.call((graph_def, params))(samples)[0]

    # likelihood
    def likelihood(model_fn): return likelihood_density(model_fn, **likelihood_params)

    # log-likelihood function handle
    @jax.custom_jvp
    def logll_fn(model_fn):
        return jax.vmap(lambda y, model_fn: -likelihood(model_fn)._logpdf(y), in_axes=(0, 0), out_axes=0)(targets, model_fn)

    # log-likelihood jacobian-vector product
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

    @jax.jit
    def ggn_mv(
        vec: PyTree,
    ):
        _, jvp = jax.linearize(model_fn, params)
        HJv = logll_hvp((model_fn(params),), (jvp(vec),))[1]
        return jax.linear_transpose(jvp, vec)(HJv)[0]

    def wrap_ggn_mv(
        vec: PyTree | Vector | Matrix,
        batch_size: Optional[Int] = None
    ):
        if isinstance(vec, Vector) or isinstance(vec, Matrix):
            vec = array_to_pytree(vec, params)

        if pytree_size(params) == pytree_size(vec):
            return ggn_mv(vec)
        else:
            return jax.lax.map(ggn_mv, vec, batch_size=batch_size)

    return wrap_ggn_mv
