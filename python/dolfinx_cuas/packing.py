# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import typing
import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
import numpy
import dolfinx_cuas.cpp as _cpp


def pack_coefficients(functions: typing.Union[typing.List[_fem.Function], _fem.Function],
                      entities: numpy.ndarray):
    """
    Pack coefficients for a set of functions over a set of integral entities
    TODO: Add better description of integral entities
    """
    cpp_list = None
    if isinstance(functions, list):
        cpp_list = [getattr(u_, "_cpp_object", u_) for u_ in functions]
    else:
        cpp_list = [getattr(functions, "_cpp_object", functions)]
    return _cpp.pack_coefficients(cpp_list, entities)
