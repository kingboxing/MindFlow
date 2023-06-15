"""
For Oscillating acceleration input & lift output system, the lift magnitude at 
infinite large frequency is decided by \nabla p = a, where a is the acceleration
of the cylinder.
"""

from __future__ import print_function
from fenics import *
import os
import numpy as np
from Boundary.Set_Boundary import Boundary
from FlowSolver.FiniteElement import TaylorHood

mesh=Mesh("./mesh/cylinder_26thousand.xml")
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }

# define boundaries and mark them with integers
boundary=Boundary(mesh)
element = TaylorHood(mesh=mesh)
ds = boundary.get_measure()
n = FacetNormal(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])

Re=[45]
# unit acceleration in y direction
pressure=Expression(('0.0','0.0','x[1]'), degree=2)
element.w.interpolate(pressure)
force_p=assemble((element.p*n)[1]*ds(5))
print('Pressure Integral on the cylinder: %1.4f; Mag2db: %1.4f' % (force_p,20*np.log10(abs(force_p))))

