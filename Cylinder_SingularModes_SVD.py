"""
Problem type : Perturbation equations in the frequency domain

Details : compute the first three global singular modes of the cylinder flow at Re = 45
    five boundaries : top and bottom symmetric wall,
    no-slip cylinder surface (d = 1), velocity inlet (0, 0), outlet with -p*n+nu*grad(u)*n=0

Method : Singular Value Decomposition
"""
from __future__ import print_function
from fenics import *
import os
import time as tm
import numpy as np
from Boundary.Set_Boundary import Boundary
from SVD.SingularMode import SingularMode


mesh=Mesh("./mesh/cylinder_26thousand.xml")
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }
BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top'},
                    'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom'},
                    'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet'},
                    'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'},
                    'Outlet'  : {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet'}
                    }

# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
# Reynolds number
Re = 120
nu = Constant(1.0 / Re)


# frequency of the singular modes
omega=Constant(1.034)
# path to the base flow
path = os.getcwd()+"/mean flow/cylinder_DNS_dt_001_IPCS_Re_060meanflow"#
#path = os.getcwd()+"/base flow/cylinder_baseflow_newton_26thousand"+str(Re).zfill(3)
# initialise solver
singularmodes=SingularMode(mesh=mesh, boundary=boundary, omega=omega, nu=nu, path=path,dim='2D')
# set boundary conditions
for key in BoundaryConditions.keys():
    singularmodes.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
start_time = tm.time()
# solve the first three modes
singularmodes.solve_SVD(k=3)
#Qf=singularmodes.Qf
## get the first singular value
#vals=singularmodes.get_mode()[2]
#print(vals)
end_time=tm.time()
elapsed_time = end_time - start_time
print('time %e' % (elapsed_time/60))
#vecs=np.matrix(singularmodes.vecs)
#aa=vecs.H*Qf*vecs
# plot real part of the first mode
#singularmodes.plot_mode(direction=1)
# save the first mode under the work directory
#singularmodes.save_mode(path=os.getcwd())