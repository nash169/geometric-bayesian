
#!/usr/bin/env python
# encoding: utf-8

from ..types import Vector
from .linear_operator import LinearOperator


class InvOperator(LinearOperator):
    def __init__(
        self,
        fwd_operator: LinearOperator
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        self.fwd_op = fwd_operator

    def mv(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return self.fwd_op.solve(vec, **kwargs)

    def solve(
        self,
        vec: Vector,
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return self.mv(vec)
