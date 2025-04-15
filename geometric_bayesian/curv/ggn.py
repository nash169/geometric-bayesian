#!/usr/bin/env python
# encoding: utf-8

import jax
from flax import nnx

from geometric_bayesian.utils.types import VectorFn, Vector, Matrix, PyTree, Callable
from geometric_bayesian.functions.likelihood import neg_logll_hvp


def ggn(
    p: Callable,
    f: Callable,
    X: Vector | Matrix,
    y: Vector | Matrix
) -> VectorFn:

    graph_def, _ = nnx.split(f)
    def model_fn(p): return nnx.call((graph_def, p))(X)[0]

    def fn(
        x: PyTree,
        v: PyTree,
    ):
        _, jvp = jax.linearize(model_fn, x)
        return jax.linear_transpose(jvp, x)(neg_logll_hvp(p, y, model_fn(x), jvp(v)))[0]

    return fn
