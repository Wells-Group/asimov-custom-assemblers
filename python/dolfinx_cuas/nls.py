# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import typing
from petsc4py import PETSc
from dolfinx import fem, cpp
import mpi4py

__all__ = ["NonlinearProblemCUAS", "NewtonSolver"]


class NonlinearProblemCUAS:
    """Nonlinear problem class for solving the non-linear problem
    F(u, v) = 0 for all v in V

    """

    def __init__(self, F: typing.Callable[[PETSc.Vec, PETSc.Vec], None],
                 J: typing.Callable[[PETSc.Vec, PETSc.Mat], None],
                 create_b: typing.Callable[[], PETSc.Vec], create_A: typing.Callable[[], PETSc.Mat],
                 bcs: typing.List[fem.DirichletBCMetaClass] = [], form_compiler_options={}, jit_options={}):

        """Initialize class that sets up structures for solving the non-linear problem using Newton's method,
        dF/du(u) du = -F(u)

        Parameters
        ----------
        F
            Function that computes the residual F(u). The first input argument to the callable is a PETSc
            vector with the values of u, the second the tensor to assemble F(u) into. F(x, b) -> b = F(x).
        J
            Function that compute the Jacobian matrix J(u). J(x, A) -> A = J(x)
        create_b
            Function that creates the vector used in residual assembly
        create_A
            Function that creates the matrix used for the Jacobian assembly
        bcs
            List of Dirichlet boundary conditions
        form_compiler_options
            Parameters used in FFCX compilation of this form. Run `ffcx --help` at
            the commandline to see all available options. Takes priority over all
            other parameter values, except for `scalar_type` which is determined by
            DOLFINX.
        jit_options
            Parameters used in CFFI JIT compilation of C code generated by FFCX.
            See `python/dolfinx/jit.py` for all available parameters.
            Takes priority over all other parameter values.
        """
        self._F = F
        self._J = J
        self.bcs = bcs
        self.create_b = create_b
        self.create_A = create_A

    def form(self, x: PETSc.Vec):
        """
        This function is called before the residual or Jacobian is computed.
        This is usually used to update ghost values.
        Parameters
        ----------
        x
            The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.
        Parameters
        ----------
        x
            The vector containing the latest solution
        b
            Vector to assemble the residual into
        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        self._F(x, b)
        b.assemble()
        # Apply boundary condition
        # FIXME: Not yet implemented

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        """Assemble the Jacobian matrix.
        Parameters
        ----------
        x
            The vector containing the latest solution
        A
            The matrix to assemble the Jacobian into
        """
        A.zeroEntries()
        self._J(x, A)
        A.assemble()


class NewtonSolver(cpp.nls.petsc.NewtonSolver):
    def __init__(self, comm: mpi4py.MPI.Intracomm, problem: NonlinearProblemCUAS):
        """
        Create a Newton solver for a given MPI communicator and non-linear problem.
        """
        super().__init__(comm)

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self._A = problem.create_A()
        self.setJ(problem.J, self._A)
        self._b = problem.create_b()
        self.setF(problem.F, self._b)
        self.set_form(problem.form)

    def solve(self, u: fem.Function):
        """
        Solve non-linear problem into function u.
        Returns the number of iterations and if the solver converged
        """
        n, converged = super().solve(u.vector)
        u.x.scatter_forward()
        return n, converged

    @property
    def A(self) -> PETSc.Mat:
        """Get the Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Get the residual vector"""
        return self._b

    def setP(self, P: typing.Callable, Pmat: PETSc.Mat):
        """
        Set the function for computing the preconditioner matrix
        Parameters
        -----------
        P
          Function to compute the preconditioner matrix b (x, P)
        Pmat
          The matrix to assemble the preconditioner into
        """
        super().setP(P, Pmat)
