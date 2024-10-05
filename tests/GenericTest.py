
from dolfin import *
import numpy as np
from src.BasicFunc.ElementFunc import TaylorHood
from src.BasicFunc.Boundary import SetBoundary, SetBoundaryCondition, BoundaryCondition, Boundary
from src.NSolver.SteadySolver import NewtonSolver
print('------------ Testing base flow function ------------')

#%%
mesh = Mesh("./data/mesh/cylinder_26k.xml")
element = TaylorHood(mesh=mesh, order=(2, 1))
# initialise solver
solver = NewtonSolver(mesh=mesh, order=(2, 1))
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
solver.boundary.VelocityInlet(mark=3, vel=(1.0, 0.0))
solver.boundary.FreeBoundary(mark=4)
solver.boundary.NoSlipWall(mark=5)

# # mark boundary and set boundary conditions
solver.set_boundary()
solver.set_boundarycondition()

# solve for Re=80
Re = 80
# compare results
datapath = './data/baseflow/bf_newton_cylinder_26k_re' + str(Re).zfill(3)
data = TimeSeries(datapath)
data.retrieve(element.w.vector(), 0.0)

# set solver parameters
solver.update_parameters(
    {'newton_solver': {'linear_solver': 'mumps', 'absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12}})
# solve the problem
solver.solve(Re)
vorti = solver.eval_vorticity()
# print norm-inf
norm2 = np.linalg.norm((element.w.vector() - solver.flow.vector()).get_local(), ord=np.inf)
# print results
print('Results are printed as follows : ')
print('Re = %d     Error_2norm = %e     drag = %e    lift = %e' % (
Re, norm2, solver.eval_force(mark=5, dirc=0), solver.eval_force(mark=5, dirc=1)))
print('------------ Testing completed ------------')