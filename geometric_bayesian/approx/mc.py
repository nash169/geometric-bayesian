#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import inspect
from functools import wraps
from geometric_bayesian.utils.types import Callable, Matrix
from geometric_bayesian.densities.abstract_density import AbstractDensity

from flax import nnx
from geometric_bayesian.utils.helper import wrap_pytree_function
from geometric_bayesian.utils.types import Optional, Int


def mc(
        fn: Callable,
        p: AbstractDensity | Matrix,
        n_samples: int = 100,
        reduce_fn: Callable = jnp.mean,
        **kwargs
):
    fn_params = list(inspect.signature(fn).parameters.values())
    samples = p.sample(size=n_samples, **kwargs) if isinstance(p, AbstractDensity) else p

    if len(fn_params) == 1:
        return jax.vmap(fn)(samples).mean()
    else:
        def fn_vmap(*args):
            f = lambda x: fn(*args, **{fn_params[-1].name: x})
            return reduce_fn(jax.vmap(f)(samples), axis=0)
        return fn_vmap


def mc_mean(
        fn: Callable,
        p: AbstractDensity | Matrix,
        n_samples: int = 100,
        **kwargs
):
    sig = inspect.signature(fn)

    @wraps(fn)
    def mean_fn(*args, **kwargs):
        return fn(*args, **kwargs).mean()
    mean_fn.__signature__ = sig

    return mc(mean_fn, p, n_samples, **kwargs)


def mc_var(
        fn: Callable,
        p: AbstractDensity | Matrix,
        n_samples: int = 100,
        **kwargs
):
    sig = inspect.signature(fn)

    @wraps(fn)
    def var_fn(*args, **kwargs):
        return fn(*args, **kwargs).var()
    var_fn.__signature__ = sig

    # expectation of the variance
    exp_var = mc(var_fn, p, n_samples, **kwargs)
    # variance of expectation
    var_exp = mc_mean(fn, p, n_samples, reduce_fn=jnp.std, **kwargs)

    return lambda *args: exp_var(*args) + var_exp(*args)
