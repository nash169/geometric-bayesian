#!/usr/bin/env python
# encoding: utf-8

import jax
from collections.abc import Callable, Iterable, Mapping
from typing import Any, List, Tuple, Optional

from jaxtyping import Array, Float, Int, Num, PRNGKeyArray, PyTree

# vector, matrix, tensor size
Size = Int[Array, "..."]

# scalar
Scalar = Float[Array, ""]

# vector -> 1d float array
Vector = Float[Array, "m"]

# matrix -> 2d float array
Matrix = Float[Array, "m n"]

# generic tensor
Tensor = Float[Array, "..."]

# scalar-valued function
ScalarFn = Callable[..., Scalar]

# vector-valued function
VectorFn = Callable[..., Vector]

# graph def
PyTreeDef = jax.tree_util.PyTreeDef

# parameters pytree
Params = PyTree[Num[Array, "..."]]

# vector -> 1d int array
VectorInt = Int[Array, "m"]
