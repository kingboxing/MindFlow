#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:53:24 2023

@author: bojin
"""

from context import *
from src.LinAlg.Utils import sort_complex, load_complex

print('------------ Testing Eigen-decomposition Solver ------------')

process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()
#%%
mesh = Mesh("./data/mesh/cylinder_26k.xml")
element = TaylorHood(mesh=mesh, order=(2, 1))
# initialise solver
solver = EigenAnalysis(mesh=mesh, order=(2, 1))

# store boundary locations and conditions
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
solver.boundary.VelocityInlet(mark=3, vel=(0.0, 0.0))
solver.boundary.FreeBoundary(mark=4)
solver.boundary.NoSlipWall(mark=5)

# mark boundary and set boundary conditions
solver.set_boundary()
solver.set_boundarycondition()
# solve for Re=80
Re = 80

# retrieve results
datapath = './data/baseflow/bf_newton_cylinder_26k_re' + str(Re).zfill(3)
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)
# set baseflow
solver.set_baseflow(ic=element.w)
# solve
solver.param[solver.param['solver_type']]['which'] = 'LM'
solver.param[solver.param['solver_type']]['lusolver'] = 'superlu'
solver.param[solver.param['solver_type']]['symmetry'] = False
solver.param[solver.param['solver_type']]['ncv'] = 600

solver.solve(k=200, Re=Re)

# compare results
eigs = np.asarray(load_complex('./data/eigen/LMEigen_bf_newton_cylinder_26k_re080.txt'))
vals, ind = sort_complex(solver.vals)
Error_vals = np.linalg.norm(vals[:20] - eigs[:20], ord=np.inf)
# compare unstable eigs
Error_us = np.linalg.norm(np.loadtxt('./data/eigen/eigenvalues.txt') -
                          list(zip([Re], [np.abs(np.imag(vals[0]))],
                                   [np.real(vals[0])])), ord=np.inf)
datapath = './data/eigen/cylinder_eigenvecs_LR_Re' + str(Re).zfill(3)
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)
vecs = element.w.vector().get_local()
data.retrieve(element.w.vector(), 1.0)
vecs = vecs + element.w.vector().get_local() * 1j
Error_vecs = np.linalg.norm(np.abs(vecs) - np.abs(solver.vecs[:, ind[0]]), ord=np.inf)
# print results
print('Results are printed as follows : ')
print(
    f'Re = {Re}\nAll_ValError_inf_norm = {Error_vals}\nUs_ValError_inf_norm = {Error_us}\nVecError_2norm = {Error_vecs}')
#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
print('Elapsed Time = %e' % (elapsed_time))
print(f"Average CPU usage: {round(np.average(cpu_usage_diff), 2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')
