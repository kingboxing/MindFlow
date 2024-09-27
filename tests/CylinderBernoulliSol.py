#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:14:16 2024

@author: bojin
"""

from context import *

print('------------ Testing Bernoulli Feedback Control Solver ------------')
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
# IO vec generator
vec_gen = VectorGenerator(model.element, model.boundary_condition)
input_vec, coord = vec_gen.unit_vector((0.6, 0.6), 0)
output_vec = vec_gen.point_vector((3.0, 0.0), 1)

# assemble model
model.assemble_model(input_vec=input_vec, output_vec=output_vec, Re=Re)
InitFeedback = BernoulliFeedback(model.SSModel)
transpose = True  # False for LQR, True for LQE
k0 = InitFeedback.solve(transpose=transpose)

vals_us, vecs_us = InitFeedback.validate_eigs(None, k=100, sigma=0.0, transpose=transpose)
vals, vecs = InitFeedback.validate_eigs(k0, k=100, sigma=0.0, transpose=transpose)

# Create the figure and axis
fig, ax = plt.subplots()
ax.scatter(np.real(vals_us), np.imag(vals_us), marker='o', color='red')
ax.scatter(np.real(vals), np.imag(vals), facecolors='none', marker='o', color='blue')
#ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-0.8, 0.2])
ax.set_ylim([-1, 1])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.show()
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
