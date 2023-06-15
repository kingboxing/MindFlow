from __future__ import print_function
from fenics import *
import numpy as np
import time
from Boundary.Set_Boundary import Boundary
from FlowSolver.NS_Newton_Solver import *
from FlowSolver.FiniteElement import TaylorHood, SplitPV
from Plot.Functionmap import *
from Plot.Matplot import *

mesh=Mesh("mesh/cylinder_26thousand.xml")
element=TaylorHood(mesh=mesh)#SplitPV(mesh=mesh)#

savepath="base flow/cylinder_baseflow_newton_26thousand120"#+str(int(1/float(solver.nu))).zfill(3)
timeseries_r = TimeSeries(savepath)
timeseries_r.retrieve(element.w.vector(), 0.0)

# # Coordinates of all dofs in the mixed space
#maping=Functionmap(element.functionspace)
# # Coordinates of dofs of first subspace of the mixed space
#V0_dofs_w0 = maping.get_subcoor(element.functionspace.sub(0).sub(0))
#w_0=maping.get_subvalue(element.functionspace.sub(0).sub(0),element.w)
#
#V0_dofs_w1 = maping.get_subcoor(element.functionspace.sub(0).sub(1))
#w_1=maping.get_subvalue(element.functionspace.sub(0).sub(1),element.w)
#
#V0_dofs_w2 = maping.get_subcoor(element.functionspace.sub(1))
#w_2=maping.get_subvalue(element.functionspace.sub(1),element.w)
#
#contourf_cylinder(V0_dofs_w0[:,0],V0_dofs_w0[:,1],w_0,xlim=(-15,23),ylim=(-15,15))
#
#contourf_cylinder(V0_dofs_w1[:,0],V0_dofs_w1[:,1],w_1,xlim=(-15,23),ylim=(-15,15))
#
#contourf_cylinder(V0_dofs_w2[:,0],V0_dofs_w2[:,1],w_2,xlim=(-15,23),ylim=(-15,15))
#
vorticity=project(grad(element.w[1])[0]-grad(element.w[0])[1])
Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,20),ylim=(-5,5),vminf=-4,vmaxf=4,colormap='seismic',colorbar='off',axis='off',figsize=(8.8, 4))

#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-1,9),ylim=(-2,2),vminf=-3000,vmaxf=1000)