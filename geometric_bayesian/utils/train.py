#!/usr/bin/env python
# encoding: utf-8

import time
import jax.numpy as jnp
from jax import grad, jit


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def loss(model, x, targets):
    preds = model(x)
    return -jnp.mean(preds * targets)


def update(model, x, y, step_size=1e-2):
    grads = grad(loss)(model, x, y)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(model.params, grads)]


def train(model, data, num_epochs, num_labels=None, verbose=False):
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in data:
            x = jnp.reshape(x, (len(x), -1))
            if num_labels is not None:
                y = one_hot(y, num_labels)
            model.params = update(model, x, y)
        epoch_time = time.time() - start_time

        if verbose:
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))


def accuracy(model, x, targets):
    target = jnp.argmax(targets, axis=1)
    predicted = jnp.argmax(model(x), axis=1)
    return jnp.mean(predicted == target)
