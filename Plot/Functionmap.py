from __future__ import print_function
from fenics import *
import numpy as np

"""This module provides the classes that extract coordinates or values from subspaces
"""


class Functionmap:
    def __init__(self, functionspace):
        self.dofs_coor = functionspace.tabulate_dof_coordinates().reshape((-1, 2))

    def get_subcoor(self, subspace):
        dofs_sub = subspace.dofmap().dofs()
        subcoor = self.dofs_coor[dofs_sub, :]
        return subcoor

    def get_subvalue(self, subspace, function):
        dofs_sub = subspace.dofmap().dofs()
        subvalue = function.vector().array()[dofs_sub]
        return subvalue

def Boundary_Coor(boundaries,bodymark):
    cylinder_coorx = []
    cylinder_coory = []
    index = []
    It_mesh = SubsetIterator(boundaries, bodymark)
    for face in It_mesh:
        for vert in vertices(face):
            index.append(vert.index())
            cylinder_coorx.append(vert.midpoint().x())
            cylinder_coory.append(vert.midpoint().y())
    
    u, indices = np.unique(index, return_index=True)
    index = np.array(index)[indices[:]]
    cylinder_coorx=np.array(cylinder_coorx)[indices[:]]
    cylinder_coory=np.array(cylinder_coory)[indices[:]]
    return cylinder_coorx, cylinder_coory, index