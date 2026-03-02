#!/usr/bin/env python
# encoding: utf-8

import os
import sys

import jax.random as jr
import optax

gb_root = os.path.abspath(
    os.path.join(os.getcwd(), "/Users/bernardo/Repos/geometric-bayesian")
)
if not os.path.isdir(gb_root):
    gb_root = "/home/ubuntu/geometric-bayesian"
sys.path.insert(0, gb_root)

from geometric_bayesian.models.gp import GP
from geometric_bayesian.kernels import rbf
from geometric_bayesian.utils.train import TrainCfg, train


def main() -> None:
    model = GP(dim=3, kernel=rbf)
    x = jr.uniform(jr.key(0), (10, 3))
    y = jr.uniform(jr.key(0), (10,))

    def loss_fn(m, x, y):
        return -m(x, y)

    print(loss_fn(model, x, y))
    cfg = TrainCfg(opt=optax.adam(1e-2), steps=10, batch_size=5)
    loss_val = train(model, x, y, loss_fn, cfg)

    print("loss", float(loss_val))


if __name__ == "__main__":
    main()
