from __future__ import print_function
from fenics import *
from scipy.sparse import diags
import numpy as np
import time
from Boundary.Set_Boundary import Boundary
from FlowSolver.FiniteElement import TaylorHood
from FrequencyResponse.FrequencyResponse import InputOutput
from Plot.Functionmap import *
from Plot.Matplot import *

mesh=Mesh("./mesh/cylinder_26thousand.xml")
element=TaylorHood(mesh=mesh)
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }
# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
# coordinates of points on the cylinder
output_vec = InputOutput(mesh=mesh, boundary=boundary, nu=Constant(1.0/80.0), element=element)
cylinder_coorx, cylinder_coory = Boundary_Coor(boundary.get_domain(), 5)
cylinder_coorx = np.array(cylinder_coorx)
cylinder_coory = np.array(cylinder_coory)
# construct matrix to extract pressure elements, may duplicated
vec_pre = output_vec.PointPressure(coordinate=[cylinder_coorx[0], cylinder_coory[0]])
for i in range(len(cylinder_coorx)):
    vec_pre = np.logical_or(vec_pre, output_vec.PointPressure(coordinate=[cylinder_coorx[i], cylinder_coory[i]]))
vec_pre = vec_pre*1 # convert bool to int
matrix_pre = diags(vec_pre[0])
indices = matrix_pre.nonzero()
# non-square matrix to delete the zero tems in the results
matrix_pre = matrix_pre.tocsc()[indices[0],:]
# Coordinates of all dofs in the mixed space
maping=Functionmap(element.functionspace)
coor = maping.dofs_coor
V0_dofs_w0 = maping.get_subcoor(element.functionspace.sub(0).sub(0))
V0_dofs_w1 = maping.get_subcoor(element.functionspace.sub(0).sub(1))
V0_dofs_w2 = maping.get_subcoor(element.functionspace.sub(1))
# construct matrix to extract pressure elements
cylinder_coor = matrix_pre * coor
# path to the data
datapath='/mnt/F4E66F4EE66F0FE2/dropbox_data/FrequencyAnlysis/OscillatingAcc_system_state/cylinder_fre_respond_yoscillationacc_state_080_'+'imagpart'
timeseries_r = TimeSeries(datapath)
fre = timeseries_r.vector_times()
# # initialise the pressure matrix
# pressure = np.zeros((len(cylinder_coor),len(fre)))
# for i in range(len(fre)):
#     savepath='vel_v_fre/'+'vel_v_fre_'+'seq_'+str(i)+'.png'
#     title = 'Frequency = '+ str('{:.5f}'.format(fre[i])).zfill(11)
#     timeseries_r.retrieve(element.w.vector(), fre[i])
#     pressure[:,i] = matrix_pre * element.w.vector()[:]
# np.savetxt('cylinder_pointpressure_imag', pressure)
# np.savetxt('cylinder_coor', cylinder_coor)
# np.savetxt('cylinder_fre', fre)


for i in range(len(fre)):
    savepath_p = 'pressure_fre_imag/' + 'pressure_fre_' + 'seq_' + str(i) + '.png'
    savepath_u = 'vel_u_fre_imag/' + 'vel_u_fre_' + 'seq_' + str(i) + '.png'
    savepath_v = 'vel_v_fre_imag/' + 'vel_v_fre_' + 'seq_' + str(i) + '.png'

    title = 'Frequency = ' + str('{:.5f}'.format(fre[i])).zfill(11)
    timeseries_r.retrieve(element.w.vector(), fre[i])
    # Coordinates of dofs of first subspace of the mixed space
    w_0 = maping.get_subvalue(element.functionspace.sub(0).sub(0), element.w)
    w_1 = maping.get_subvalue(element.functionspace.sub(0).sub(1), element.w)
    w_2 = maping.get_subvalue(element.functionspace.sub(1), element.w)

    contourf_cylinder(V0_dofs_w0[:, 0], V0_dofs_w0[:, 1], w_0, xlim=(-15, 23), ylim=(-15, 15), figname=title,
                      savepath=savepath_u)
    contourf_cylinder(V0_dofs_w1[:, 0], V0_dofs_w1[:, 1], w_1, xlim=(-15, 23), ylim=(-15, 15), figname=title,
                      savepath=savepath_v)
    contourf_cylinder(V0_dofs_w2[:, 0], V0_dofs_w2[:, 1], w_2, xlim=(-15, 23), ylim=(-15, 15), figname=title,
                      savepath=savepath_p)