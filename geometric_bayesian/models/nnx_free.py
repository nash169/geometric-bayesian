#!/usr/bin/env python
# encoding: utf-8

from abc import ABC, abstractmethod
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp


class AbstractApproximator(ABC):
    # force definition of the domain
    def __init__(self):
        pass

    # overload () operator calls predict method
    def __call__(self, x):
        return self.predict(x) if len(x.shape) == 1 else vmap(self.predict, in_axes=0)(x)

    # predict function
    @abstractmethod
    def predict(self, x):
        pass

    # # params
    # @property
    # @abstractmethod
    # def params(self):
    #     return self._params
    #
    # @params.setter
    # def params(self, value):
    #     self._params = value


class MLP(AbstractApproximator):
    def __init__(
            self,
            layers: List[int],
            scale: Optional[float] = 1e-2,
            softmax: Optional[bool] = False,
    ) -> None:
        super().__init__()

        # classification
        self.softmax = softmax

        # initialize all layers for a fully-connected neural network with sizes "layers"
        keys = random.split(random.key(0), len(layers))
        self.params = [self._random_layer_params(m, n, k, scale) for m, n, k in zip(layers[:-1], layers[1:], keys)]

    def predict(self, x):
        activations = x
        for w, b in self.params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self._relu(outputs)

        final_w, final_b = self.params[-1]
        logits = jnp.dot(final_w, activations) + final_b

        return logits - logsumexp(logits) if self.softmax else logits

    # helper function to randomly initialize weights and biases for a dense neural network layer
    def _random_layer_params(self, m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    # activation
    @staticmethod
    def _relu(x):
        return jnp.maximum(0, x)


class Block(nnx.Module):
    def __init__(self, input_dim, features, rngs):
        self.linear = nnx.Linear(input_dim, features, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array):  # No need to require a second input!
        x = self.linear(x)
        x = self.dropout(x)
        x = jax.nn.relu(x)
        return x   # No need to return a second output!
