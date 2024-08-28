#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:47:22 2023

@author: bojin

Problem type : steady Navier-Stokes equations (base flow)

Details : steady cylinder flow at Re = 80
    five boundaries : top and bottom symmetric wall,
    no-slip cylinder surface (d = 1), velocity inlet (1, 0), outlet with -p*n+nu*grad(u)*n=0

Method : Newton method
"""
from context import *

print('------------ Testing base flow function ------------')
start_time = time.time()
mesh=Mesh("./data/mesh/cylinder_26k.xml")
element=TaylorHood(mesh=mesh,order=(2,1))
# initialise solver
solver = NewtonSolver(mesh=mesh, order=(2,1))
# store boundary locations and conditions
BoundaryLocations = {1 : {'name': 'Top',     'location':'on_boundary and near(x[1], 15.0, tol)'},
                     2 : {'name': 'Bottom',  'location':'on_boundary and near(x[1], -15.0, tol)'},
                     3 : {'name': 'Inlet',   'location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                     4 : {'name': 'Outlet',  'location':'on_boundary and near(x[0], 23.0, tol)'},
                     5 : {'name': 'Cylinder','location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                     }

solver.boundary.bc_list.update(BoundaryLocations)
solver.boundary.Symmetry(mark=1, norm=(0,1))
solver.boundary.Symmetry(mark=2, norm=(0,1))
solver.boundary.VelocityInlet(mark=3, vel=(1.0,0.0))
solver.boundary.FreeBoundary(mark=4)
solver.boundary.NoSlipWall(mark=5)

# # mark boundary and set boundary conditions
solver.set_boundary()
solver.set_boundarycondition()

# solve for Re=80
Re=80
# compare results
datapath='./data/baseflow/bf_newton_cylinder_26k_re080'
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)

if False:
    solver.initial(ic=element.w)
    
# set solver parameters
solver.parameters({'newton_solver':{'linear_solver': 'mumps','absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12}})
# solve the problem
solver.solve(Re)

# print norm-inf
norm2=np.linalg.norm((element.w.vector()-solver.flow.vector()).get_local(), ord=np.inf)

elapsed_time = time.time() - start_time
# print results
print('Results are printed as follows : ')
print('Re = %d     Error_2norm = %e     drag = %e    lift = %e' % (Re, norm2, solver.eval_force(mark=5,dirc=0) , solver.eval_force(mark=5,dirc=1)))

print('Elapsed Time = %e' % (elapsed_time))
print('------------ Testing completed ------------')