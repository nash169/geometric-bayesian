#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from bayax.operators.dense_operator import DenseOperator
from bayax.operators.linear_operator import LinearOperator
from bayax.operators.low_rank_operator import LowRankOperator
from bayax.utils.types import Scalar, Size, Vector, Matrix


@jax.tree_util.register_pytree_node_class
class ScaledOperator(LinearOperator):
    def __init__(self, scalar: Scalar, op: LinearOperator) -> None:
        if isinstance(op, ScaledOperator):
            scalar = scalar * op.scalar
            op = op.op
        self.scalar = scalar
        self.op = op

    def tree_flatten(self):
        if isinstance(self.scalar, jax.Array):
            return (self.scalar, self.op), {"scalar_is_static": False}
        return (self.op,), {"scalar_is_static": True, "scalar": self.scalar}

    @classmethod
    def tree_unflatten(cls, aux, children):
        if aux["scalar_is_static"]:
            return cls(aux["scalar"], children[0])
        return cls(children[0], children[1])

    def size(self) -> Size:
        return self.op.size()

    def mv(self, vec: Vector) -> Vector:
        return self.scalar * self.op.mv(vec)

    def solve(self, vec: Vector, **kwargs) -> Vector:
        try:
            return self.op.solve(vec, **kwargs) / self.scalar
        except (AttributeError, NotImplementedError, TypeError):
            return jnp.linalg.solve(self._dense_matrix(), vec) / self.scalar

    def invquad(self, vec: Vector, **kwargs) -> Scalar:
        try:
            return self.op.invquad(vec, **kwargs) / self.scalar
        except (AttributeError, NotImplementedError, TypeError):
            try:
                return self.op.inv_quad(vec, **kwargs) / self.scalar
            except (AttributeError, NotImplementedError, TypeError):
                return vec @ self.solve(vec, **kwargs)

    def logdet(self, **kwargs) -> Scalar:
        try:
            logdet = self.op.logdet(**kwargs)
            if isinstance(logdet, tuple):
                sign, logdet = logdet
                logdet = jnp.where(sign > 0, logdet, jnp.inf)
        except (AttributeError, NotImplementedError, TypeError):
            sign, logdet = jnp.linalg.slogdet(self._dense_matrix())
            logdet = jnp.where(sign > 0, logdet, jnp.inf)
        return self._logdet_dim() * jnp.log(self.scalar) + logdet

    def det(self) -> Scalar:
        try:
            det = self.op.det()
        except (AttributeError, NotImplementedError, TypeError):
            det = jnp.linalg.det(self._dense_matrix())
        return jnp.power(self.scalar, self._logdet_dim()) * det

    def transpose(self) -> LinearOperator:
        try:
            return ScaledOperator(self.scalar, self.op.transpose())
        except (AttributeError, NotImplementedError, TypeError):
            return self.dense().transpose()

    def inverse(self) -> LinearOperator:
        try:
            return ScaledOperator(jnp.reciprocal(self.scalar), self.op.inverse())
        except (AttributeError, NotImplementedError, TypeError):
            return DenseOperator(jnp.linalg.inv(self._dense_matrix()) / self.scalar)

    def dense(self) -> DenseOperator:
        return DenseOperator(self.scalar * self._dense_matrix())

    def lowrank(self, **kwargs) -> LinearOperator:
        if isinstance(self.op, LowRankOperator):
            return LowRankOperator(
                diag=self.scalar * self.op.diag,
                right=self.op.right,
                left=self.op.left,
                zero_tol=self.op.zero_tol,
            )
        try:
            op = self.op.lowrank(**kwargs)
            if isinstance(op, LowRankOperator):
                return LowRankOperator(
                    diag=self.scalar * op.diag,
                    right=op.right,
                    left=op.left,
                    zero_tol=op.zero_tol,
                )
            return ScaledOperator(self.scalar, op)
        except (AttributeError, NotImplementedError, TypeError):
            eigval, eigvec = self._dense_eigh(**kwargs)
            return LowRankOperator(
                diag=eigval,
                right=eigvec,
                zero_tol=kwargs.get("zero_tol", 1e-8),
            )

    def diagonalize(self, **kwargs) -> tuple[Vector, Matrix]:
        try:
            eigval, eigvec = self.op.diagonalize(**kwargs)
            return self.scalar * eigval, eigvec
        except (AttributeError, NotImplementedError, TypeError):
            return self._dense_eigh(**kwargs)

    def squareroot(self, **kwargs) -> LinearOperator:
        try:
            return ScaledOperator(jnp.sqrt(self.scalar), self.op.squareroot(**kwargs))
        except (AttributeError, NotImplementedError, TypeError):
            return self._dense_sqrt_factor()

    def sqrtf(self, **kwargs) -> LinearOperator:
        try:
            return ScaledOperator(jnp.sqrt(self.scalar), self.op.sqrtf(**kwargs))
        except (AttributeError, NotImplementedError, TypeError):
            return self.squareroot(**kwargs)

    @property
    def sqrt(self) -> Matrix:
        try:
            return jnp.sqrt(self.scalar) * self.op.sqrt
        except (AttributeError, NotImplementedError, TypeError):
            sqrtf = self.sqrtf()
            if hasattr(sqrtf, "_mat"):
                return sqrtf._mat
            return self._operator_matrix(sqrtf)

    def diag(self, **kwargs) -> Vector:
        diag = getattr(self.op, "diag", None)
        if isinstance(diag, jax.Array):
            return self.scalar * diag
        if callable(diag):
            return self.scalar * diag(**kwargs)
        return self.scalar * jnp.diag(self._dense_matrix())

    def _logdet_dim(self) -> int:
        diag = getattr(self.op, "diag", None)
        if isinstance(diag, jax.Array) and diag.ndim == 1:
            return diag.shape[0]
        return self.size()[0]

    def _dense_matrix(self) -> Matrix:
        if hasattr(self.op, "_mat"):
            return self.op._mat
        if hasattr(self.op, "dense_operator"):
            mat = self.op.dense_operator()
            if isinstance(mat, jax.Array):
                return mat
        try:
            dense = self.op.dense()
            if hasattr(dense, "_mat"):
                return dense._mat
            if isinstance(dense, jax.Array):
                return dense
        except (AttributeError, NotImplementedError, TypeError):
            pass
        return self._operator_matrix(self.op)

    @staticmethod
    def _operator_matrix(op: LinearOperator) -> Matrix:
        m, n = op.size()
        eye = jnp.eye(n)
        return jax.vmap(op.mv, in_axes=1, out_axes=1)(eye).reshape(m, n)

    def _dense_sqrt_factor(self) -> DenseOperator:
        mat = self.scalar * self._dense_matrix()
        eigval, eigvec = jnp.linalg.eigh(mat)
        eigval = jnp.clip(eigval, min=0.0)
        return DenseOperator(eigvec * jnp.sqrt(eigval))

    def _dense_eigh(self, **kwargs) -> tuple[Vector, Matrix]:
        eigval, eigvec = jnp.linalg.eigh(self._dense_matrix())
        eigval, eigvec = self.scalar * eigval, eigvec
        eigval, eigvec = jnp.flip(eigval), jnp.flip(eigvec, axis=1)
        k = kwargs.get("k", None)
        if k is not None:
            eigval, eigvec = eigval[:k], eigvec[:, :k]
        return eigval, eigvec
