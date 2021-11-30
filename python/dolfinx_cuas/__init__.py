# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

"""User interface for custom assemblers"""

# flake8: noqa

from .custom_assembler import assemble_matrix, assemble_matrix_numba, assemble_vector
from .verification import compute_reference_mass_matrix, compute_reference_stiffness_matrix, compute_reference_surface_matrix
from .utils import estimate_max_polynomial_degree
from .nls import NewtonSolver, NonlinearProblemCUAS
