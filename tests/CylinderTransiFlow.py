#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:54:28 2023

@author: bojin

Problem type : unsteady Navier-Stokes equations

Details : unsteady cylinder flow at Re = 2000
    five boundaries : top and bottom symmetric wall,
    no-slip cylinder surface (d = 1), velocity inlet (1, 0), outlet with -p*n+nu*grad(u)*n=0 and p = 0

Method : IPCS method (split method)


"""
#%%
"""
passed the test
not yet compare correct results

"""
#%%
from context import *

print('------------ Testing IPCS solver ------------')
start_time = time.time()
mesh = Mesh("./data/mesh/cylinder_26k.xml")
element_base = TaylorHood(mesh=mesh)
#%% initialise solver
Re = 80
dt = 0.01
nstep = 100
solver = DNS_IPCS(mesh, Re, dt)
#%% store boundary locations and conditions
BoundaryLocations = {1: {'name': 'Top', 'location': 'on_boundary and near(x[1], 15.0, tol)'},
                     2: {'name': 'Bottom', 'location': 'on_boundary and near(x[1], -15.0, tol)'},
                     3: {'name': 'Inlet',
                         'location': 'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                     4: {'name': 'Outlet', 'location': 'on_boundary and near(x[0], 23.0, tol)'},
                     5: {'name': 'Cylinder',
                         'location': 'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                     }

solver.boundary.bc_list.update(BoundaryLocations)
solver.boundary.Symmetry(mark=1, norm=(0, 1))
solver.boundary.Symmetry(mark=2, norm=(0, 1))
solver.boundary.VelocityInlet(mark=3, vel=(1.0, 0.0))
solver.boundary.FreeBoundary(mark=4)
solver.boundary.NoSlipWall(mark=5)

# # mark boundary and set boundary conditions
solver.set_boundary()
solver.set_boundarycondition()

#%% initial condition
datapath = './data/baseflow/bf_newton_cylinder_26k_re' + str(Re).zfill(3)
data = TimeSeries(datapath)
data.retrieve(element_base.w.vector(), 0.0)
solver.initial(datapath, element_init=element_base)
#%% solve
lift = (solver.eval_force(mark=5, dirc=1),)
drag = (solver.eval_force(mark=5, dirc=0),)
# simulation
if comm_rank == 0:
    start_time = time.time()
    print('Results are printed as follows : ')
    print('Re = %d     dt  = %e     num_step = %d' % (Re, dt, nstep))

for i in range(0, nstep):
    time_i = i * dt
    if comm_rank == 0:
        print('time= %e(i= %d)    lift_coeff= %e    drag_coeff= %e' % (time_i, i, 2.0 * lift[i], 2.0 * drag[i]))
    solver.solve(inner_iter_max=15, tol=1e-6)
    lift += (solver.eval_force(mark=5, dirc=1, reuse=True),)
    drag += (solver.eval_force(mark=5, dirc=0, reuse=True),)
if comm_rank == 0:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('time %e' % (elapsed_time))
    print('Elapsed Time = %e' % (elapsed_time))
    print('------------ Testing completed ------------')
