#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from geometric_bayesian.operators import PSDOperator


class bcolors:
    OK = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def compare_to_dense(op, mat):
    vec = jax.random.uniform(jax.random.key(0), (mat.shape[1],))

    # matrix-vector product
    assert (jnp.allclose(op(vec), mat @ vec)), f"{bcolors.FAIL}Matrix-Vector product operation failed!{bcolors.ENDC}"
    print(f"{bcolors.OK}Matrix-Vector product operation ok!{bcolors.ENDC}")

    # solve
    assert (jnp.allclose(op.solve(vec), jnp.linalg.solve(mat, vec))), f"{bcolors.FAIL}Solve operation failed!{bcolors.ENDC}"
    print(f"{bcolors.OK}Solve operation ok!{bcolors.ENDC}")


if __name__ == "__main__":
    dim = 10
    mat = jax.random.uniform(jax.random.key(0), (dim, dim))
    mat = mat.T + mat + 100 * jnp.eye(dim)

    op = PSDOperator(
        op=mat,
        op_type='raw'
    )

    compare_to_dense(op, mat)
