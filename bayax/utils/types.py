#!/usr/bin/env python
# encoding: utf-8

from collections.abc import Callable, Iterable, Mapping
from typing import Any, List, Tuple, Optional, Self
from jaxtyping import Array, Float, Int, Num, PRNGKeyArray, PyTree, PyTreeDef

# ====================================================================================================
# FLOAT
# ====================================================================================================
Scalar = Float[Array, ""]
Vector = Float[Array, "m"]
Matrix = Float[Array, "m n"]
Tensor = Float[Array, "..."]

ScalarFn = Callable[..., Scalar]
VectorFn = Callable[..., Vector]

# ====================================================================================================
# INT
# ====================================================================================================
ScalarInt = Int[Array, ""]
VectorInt = Int[Array, "m"]
MatrixInt = Int[Array, "m n"]
TensorInt = Int[Array, "..."]

# ====================================================================================================
# HELPERS
# ====================================================================================================
Size = VectorInt

# ====================================================================================================
# JAX
# ====================================================================================================
Key = PRNGKeyArray
Params = PyTree[Num[Array, "..."]]
