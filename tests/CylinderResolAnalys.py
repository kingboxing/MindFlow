#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 23:59:16 2024

@author: bojin
"""

from context import *

print('------------ Testing Resolvent Analysis Solver ------------')

tracemalloc.start()
process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()
#%%
mesh=Mesh("./data/mesh/cylinder_26k.xml")
element=TaylorHood(mesh=mesh,order=(2,1))
# initialise solver
solver = ResolventAnalysis(mesh=mesh, order=(2,1))

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
solver.boundary.VelocityInlet(mark=3, vel=(0.0,0.0))
solver.boundary.FreeBoundary(mark=4)
solver.boundary.NoSlipWall(mark=5)

# mark boundary and set boundary conditions
solver.set_boundary()
solver.set_boundarycondition()
# solve for Re=80
Re=80

# retrieve results
datapath='./data/baseflow/bf_newton_cylinder_26k_re'+str(Re).zfill(3)
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)
# set baseflow
solver.set_baseflow(ic=element.w)
# solve
solver.param[solver.param['solver_type']]['which']='LM'

solver.solve(k=2, s=0.7*1j, Re=Re)

# print results
print('Results are printed as follows : ')
print(f'Re = {Re}\nEigenvalues = {solver.energy_amp}')
#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print('Elapsed Time = %e' % (elapsed_time))
print(f"Current memory usage: {current / (1024 * 1024):.2f} MB")
print(f"Peak memory usage: {peak / (1024 * 1024):.2f} MB")
print(f"Average CPU usage: {round(np.average(cpu_usage_diff),2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')