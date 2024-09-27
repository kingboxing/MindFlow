#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:45:33 2024

@author: bojin
"""

from context import *

print('------------ Testing state-space model ------------')
tracemalloc.start()
process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()
#%%
mesh = Mesh("./data/mesh/cylinder_26k.xml")
element = TaylorHood(mesh=mesh, order=(2, 1))
# initialise model
model = StateSpaceDAE2(mesh=mesh, order=(2, 1))

# define boundaries and conditions
model.boundary.define(mark=1, name='Top', locs='on_boundary and near(x[1], 15.0, tol)')
model.boundary.define(mark=2, name='Bottom', locs='on_boundary and near(x[1], -15.0, tol)')
model.boundary.define(mark=3, name='Inlet',
                      locs='on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))')
model.boundary.define(mark=4, name='Outlet', locs='on_boundary and near(x[0], 23.0, tol)')
model.boundary.define(mark=5, name='Cylinder',
                      locs='on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))')

model.boundary.Symmetry(mark=1, norm=(0, 1))
model.boundary.Symmetry(mark=2, norm=(0, 1))
model.boundary.VelocityInlet(mark=3, vel=(0.0, 0.0))
model.boundary.FreeBoundary(mark=4)
model.boundary.NoSlipWall(mark=5)

# mark boundary and set boundary conditions
model.set_boundary()
model.set_boundarycondition()
# solve for Re=80
Re = 80

# retrieve results
datapath = './data/baseflow/bf_newton_cylinder_26k_re' + str(Re).zfill(3)
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)
# set baseflow
model.set_baseflow(ic=element.w)
# assemble model
model.assemble_model(Re=Re)

vals, vecs = model.validate_eigs(k=2, param={'which': 'LR'})

vecs_m = vecs

# compare results
Error_vals = np.linalg.norm(np.loadtxt('./data/eigen/eigenvalues.txt') -
                            list(zip([Re], [np.abs(np.imag(vals[0]))],
                                     [np.real(vals[0])])), ord=np.inf)
datapath = './data/eigen/cylinder_eigenvecs_LR_Re' + str(Re).zfill(3)
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)
vecs = element.w.vector().get_local()
data.retrieve(element.w.vector(), 1.0)
vecs = vecs + element.w.vector().get_local() * 1j
# 
P_nbc = model.SSModel['Prol'][0]
P_nvel_bc = model.SSModel['Prol'][1]
P_npre_bc = model.SSModel['Prol'][2]
vecs = np.bmat([P_npre_bc.transpose() @ (P_nbc.transpose() @ vecs), P_nvel_bc.transpose() @ (P_nbc.transpose() @ vecs)])

Error_vecs = np.linalg.norm(np.abs(vecs) - np.abs(vecs_m[:, 0]), ord=np.inf)
# print results
print('Results are printed as follows : ')
print(f'Re = {Re}\nEigenvalues = {vals}\nValError_2norm = {Error_vals}\nVecError_2norm = {Error_vecs}')
#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print('Elapsed Time = %e' % (elapsed_time))
print(f"Current memory usage: {current / (1024 * 1024):.2f} MB")
print(f"Peak memory usage: {peak / (1024 * 1024):.2f} MB")
print(f"Average CPU usage: {round(np.average(cpu_usage_diff), 2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')
