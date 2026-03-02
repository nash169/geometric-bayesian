#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join('../')))
from geometric_bayesian.utils.types import Array
from geometric_bayesian.densities import Bernoulli, Normal, MultivariateNormal, Categorical
from geometric_bayesian.operators.psd_operator import PSDOperator

jax.config.update('jax_enable_x64', True)
rng_key = jax.random.key(0)
dim = 3
debug = False


class bcolors:
    HEADER = '\033[95m'
    WARNING = '\033[93m'
    OK = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def random_like(rng_key, x):
    if isinstance(x, Array):
        return jax.random.uniform(jax.random.split(rng_key)[1], shape=x.shape)
    elif isinstance(x, PSDOperator):
        return PSDOperator(rng_key=jax.random.split(rng_key)[1], op_size=x.size()[0])


def gradient_check(f, df, x, v):
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))

    grad = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v))

    t = jnp.log(t)
    grad = jnp.log(grad)

    w = [25, 35]
    slope = jnp.mean((grad[w[0] + 1:w[1]] - grad[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("First order Taylor expansion slope:", slope, "- It should be approximately equal to 2.0")

    return True if jnp.abs(slope - 2.0) <= 1e-1 else False, grad, t


def hessian_check(f, df, ddf, x, v):
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))

    hess = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v) - 0.5 * jnp.pow(t, 2) * (ddf(x, v) @ v))

    t = jnp.log(t)
    hess = jnp.log(hess)

    w = [30, 40]
    slope = jnp.mean((hess[w[0] + 1:w[1]] - hess[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("Second order Taylor expansion slope:", slope, "- It should be approximately equal to 3.0")

    return True if jnp.abs(slope - 3.0) <= 1e-1 else False, hess, t


densities = {
    # 'Bernoulli': {
    #     'dist': lambda mu: Bernoulli(mu=mu, logits=False),
    #     'input': jax.random.uniform(rng_key, (1,)),
    #     'params': {
    #         'mu': jax.random.uniform(rng_key, (1,))
    #     }
    # },
    # 'Bernoulli Logits Input': {
    #     'dist': lambda mu: Bernoulli(mu=mu, logits=True),
    #     'input': jax.random.uniform(rng_key, (1,)),
    #     'params': {
    #         'mu': 10 * jax.random.uniform(rng_key, (1,))
    #     }
    # },
    'Categorical': {
        'dist': lambda mu: Categorical(mu=mu, logits=False),
        'input': jax.nn.one_hot(1, dim),
        'inputInt': jnp.array(1),
        'params': {
            'mu': jax.random.uniform(rng_key, (dim,))
        }
    },
    # 'Categorical Logits Input': {
    #     'dist': lambda mu: Categorical(mu=mu, logits=True),
    #     'input': jax.nn.one_hot(1, dim),
    #     'inputInt': jnp.array(1),
    #     'params': {
    #         'mu': jax.random.uniform(rng_key, (dim,))
    #     }
    # },
    # 'Multivariate Normal - Full Covariance': {
    #     'dist': lambda mean, cov: MultivariateNormal(mean=mean, cov=cov),
    #     'input': jax.random.uniform(rng_key, (dim,)),
    #     'params': {
    #         'mean': jax.random.uniform(jax.random.split(rng_key)[1], (dim,)),
    #         'cov': PSDOperator(rng_key=rng_key, op_size=dim),
    #     }
    # }
}

for key, val in densities.items():
    print(f'Testing {bcolors.HEADER}{key}{bcolors.ENDC} distribution.')
    p, input_fix, params_fix = val['dist'], val['input'], val['params']

    print(
        f'Testing {bcolors.WARNING}input{bcolors.ENDC} [dimension {input_fix.shape[0] if input_fix.ndim > 0 else 1}] evaluation and derivatives distribution.')
    f = lambda x: p(**params_fix)(x)
    df = lambda x, v: p(**params_fix).jvp(x, v)
    ddf = lambda x, v: p(**params_fix).hvp(x, v)

    ok_grad, grads, ts = gradient_check(f, df, input_fix, jax.random.uniform(rng_key, shape=input_fix.shape))
    if ok_grad:
        print(f'{bcolors.OK}Gradient CORRECT.{bcolors.ENDC}')
    else:
        print(f'{bcolors.FAIL}Gradient NOT correct.{bcolors.ENDC}')
        if debug:
            fig, ax = plt.subplots()
            ax.scatter(ts, grads)
            ax.scatter(ts[25].item(), grads[25].item(), c='r')
            plt.show()
    ok_hess, hesss, ts = hessian_check(f, df, ddf, input_fix, random_like(rng_key, input_fix))
    if ok_hess:
        print(f'{bcolors.OK}Hessian CORRECT.{bcolors.ENDC}')
    else:
        print(f'{bcolors.FAIL}Hessian NOT correct.{bcolors.ENDC}')
        if debug:
            fig, ax = plt.subplots()
            ax.scatter(ts, hesss)
            ax.scatter(ts[25].item(), hesss[25].item(), c='r')
            plt.show()
    print(f"Function eval: {f(input_fix)}")
    print(f"Gradient eval: {df(input_fix, jax.random.uniform(rng_key, shape=input_fix.shape))}")
    print(f"Hessian eval: {ddf(input_fix, jax.random.uniform(rng_key, shape=input_fix.shape))}")
    if "inputInt" in val.keys():
        if jnp.allclose(f(input_fix), f(val['inputInt'])):
            print(f"{bcolors.OK}Function Int input CORRECT{bcolors.ENDC}.")
        else:
            print(f"{bcolors.FAIL}Function Int input NOT correct{bcolors.ENDC}.")
        if jnp.allclose(df(input_fix, jax.random.uniform(rng_key, shape=input_fix.shape)), df(val['inputInt'], jax.random.uniform(rng_key, shape=input_fix.shape))):
            print(f"{bcolors.OK}Gradient Int input CORRECT{bcolors.ENDC}.")
        else:
            print(f"{bcolors.FAIL}Gradient Int input NOT correct{bcolors.ENDC}.")
        if jnp.allclose(ddf(input_fix, jax.random.uniform(rng_key, shape=input_fix.shape)), ddf(val['inputInt'], jax.random.uniform(rng_key, shape=input_fix.shape))):
            print(f"{bcolors.OK}Hessian Int input CORRECT{bcolors.ENDC}.")
        else:
            print(f"{bcolors.FAIL}Hessian Int input NOT correct{bcolors.ENDC}.")

    print(f'Testing {bcolors.WARNING}params{bcolors.ENDC} [dimension {len(params_fix)}] evaluation and derivatives distribution.')
    for count, param_name in enumerate(params_fix.keys()):
        curr_params = {k: v for k, v in params_fix.items() if k != param_name}
        f = lambda x: partial(p, **curr_params)(x)(input_fix)
        df = lambda x, v: partial(p, **curr_params)(x).jvp_params()[count](input_fix, v)
        ddf = lambda x, v: partial(p, **curr_params)(x).hvp_params()[count](input_fix, v)

        # try:
        ok_grad, grads, ts = gradient_check(f, df, params_fix[param_name], random_like(rng_key, params_fix[param_name]))
        if ok_grad:
            print(f'{bcolors.OK}Gradient {param_name} CORRECT.{bcolors.ENDC}')
        else:
            print(f'{bcolors.FAIL}Gradient {param_name} NOT correct.{bcolors.ENDC}')
            if debug:
                fig, ax = plt.subplots()
                ax.scatter(ts, grads)
                ax.scatter(ts[25].item(), grads[25].item(), c='r')
                plt.show()
        ok_hess, hesss, ts = hessian_check(f, df, ddf, params_fix[param_name], random_like(rng_key, params_fix[param_name]))
        if ok_hess:
            print(f'{bcolors.OK}Hessian {param_name} CORRECT.{bcolors.ENDC}')
        else:
            print(f'{bcolors.FAIL}Hessian {param_name} NOT correct.{bcolors.ENDC}')
            if debug:
                fig, ax = plt.subplots()
                ax.scatter(ts, hesss)
                ax.scatter(ts[25].item(), hesss[25].item(), c='r')
                plt.show()
        print(f"Function eval: {f(params_fix[param_name])}")
        print(f"Gradient eval: {df(params_fix[param_name], random_like(rng_key, params_fix[param_name]))}")
        print(f"Hessian eval: {ddf(params_fix[param_name], random_like(rng_key, params_fix[param_name]))}")
        if "inputInt" in val.keys():
            f_int = lambda x: partial(p, **curr_params)(x)(val['inputInt'])
            df_int = lambda x, v: partial(p, **curr_params)(x).jvp_params()[count](val['inputInt'], v)
            ddf_int = lambda x, v: partial(p, **curr_params)(x).hvp_params()[count](val['inputInt'], v)
            if jnp.allclose(f(params_fix[param_name]), f_int(params_fix[param_name])):
                print(f"{bcolors.OK}Function params Int input CORRECT{bcolors.ENDC}.")
            else:
                print(f"{bcolors.FAIL}Function params Int input NOT correct{bcolors.ENDC}.")
            if jnp.allclose(df(params_fix[param_name], random_like(rng_key, params_fix[param_name])), df_int(params_fix[param_name], random_like(rng_key, params_fix[param_name]))):
                print(f"{bcolors.OK}Gradient Int input CORRECT{bcolors.ENDC}.")
            else:
                print(f"{bcolors.FAIL}Gradient params Int input NOT correct{bcolors.ENDC}.")
            if jnp.allclose(ddf(params_fix[param_name], random_like(rng_key, params_fix[param_name])), ddf_int(params_fix[param_name], random_like(rng_key, params_fix[param_name]))):
                print(f"{bcolors.OK}Hessian params Int input CORRECT{bcolors.ENDC}.")
            else:
                print(f"{bcolors.FAIL}Hessian params Int input NOT correct{bcolors.ENDC}.")
        # except:
        #     print(f'Testing {param_name} param failed')


# from tests.check_derivatives import gradient_check, hessian_check
# p = lambda mu: Bernoulli(mu=mu, logits=False)
# fixed_mu = jnp.array(0.3)
# fixed_x = jnp.array(0.0)
# # f = lambda x: p(fixed_mu)._log(x)
# # df = lambda x, v: p(fixed_mu)._jvp(x, v)
# # df_tmp = lambda x, v: jax.grad(f)(x) * v
# f = lambda mu: p(mu)._log(fixed_x)
# df = lambda mu, v: p(mu)._jvp_mu(fixed_x, v)
# ddf = lambda mu, v: p(mu)._hvp_mu(fixed_x, v)
#
#
# def df_tmp(mu):
#     return fixed_x / mu[0] - (1 - fixed_x) / (1 - mu[0])
#
#
# ddf_tmp = lambda mu, v: jax.grad(df_tmp)(mu) * v
#
# ok_grad, grads, ts = gradient_check(f, df, in_dim=1, out_dim=1)
# ok_hess, hesss, ts = hessian_check(f, df, ddf, in_dim=1, out_dim=1)
#
# # fig, ax = plt.subplots()
# # ax.plot(jnp.linspace(0, 1, 100), df(jnp.linspace(0, 1, 100), 1))
# # plt.show()
#
# # fig, ax = plt.subplots()
# # ax.scatter(ts, grads)
# # ax.scatter(ts[25].item(), grads[25].item(), c='r')
# # plt.show()
