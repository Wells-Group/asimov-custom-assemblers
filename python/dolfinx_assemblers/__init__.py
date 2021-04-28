# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""User interface for custom assemblers"""

# flake8: noqa

from .custom_assembler import assemble_matrix
from .verification import compute_reference_mass_matrix, compute_reference_stiffness_matrix
from .utils import estimate_max_polynomial_degree
