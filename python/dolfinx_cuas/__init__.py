# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

"""User interface for custom assemblers"""

# flake8: noqa

from .custom_assembler import assemble_matrix, assemble_matrix_numba, assemble_vector
from .utils import estimate_max_polynomial_degree
from .nls import NewtonSolver, NonlinearProblemCUAS
from .packing import pack_coefficients
from dolfinx_cuas.cpp import QuadratureRule, compute_active_entities, Kernel

__all__ = ["assemble_matrix", "assemble_matrix_numba", "assemble_vector",
           "estimate_max_polynomial_degree", "NewtonSolver", "NonlinearProblemCUAS",
           "QuadratureRule", "pack_coefficients", "compute_active_entities", "Kernel"]
