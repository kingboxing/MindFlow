#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:47:22 2023

@author: bojin
"""
from context import *

print('------------ Testing base flow function ------------')
start_time = time.time()
mesh=Mesh("./data/mesh/cylinder_26k.xml")
element=TaylorHood(mesh=mesh,order=(2,1))
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }
BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0)},
                    'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0)},
                    'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((1.0,0.0))},
                    'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0))},
                    'Outlet'  : {'Value': 'FreeOutlet'}
                    }

# initialise solver
solver = NewtonSolver(mesh=mesh, order=(2,1))
# set solver parameters
solver.parameters({'newton_solver':{'linear_solver': 'mumps','absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12}})
# mark boundary
for key in BoundaryLocations.keys():
     solver.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
# set boundary conditions
for key in BoundaryConditions.keys():
    solver.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])

# solve for Re=80
Re=80
# solve the problem
solver.solve(Re)

# compare results
datapath='./data/baseflow/bf_newton_cylinder_26k_re080'
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)
# print norm-inf
norm2=np.linalg.norm((element.w.vector()-solver.w.vector()).get_local(), ord=np.inf)

elapsed_time = time.time() - start_time
# print results
print('Results are printed as follows : ')
print('Re = %d     Error_2norm = %e     drag = %e    lift = %e' % (Re, norm2, solver.force(bodymark=5,direction=0) , solver.force(bodymark=5,direction=1)))

print('Elapsed Time = %e' % (elapsed_time))
print('------------ Testing completed ------------')