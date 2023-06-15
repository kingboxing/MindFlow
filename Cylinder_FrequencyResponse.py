"""
frequency respond : Y direction oscillation acceleration as input , lift as output
"""

from __future__ import print_function
import scipy.sparse.linalg as spla
from fenics import *
import os
import gc
import time as tm
import numpy as np
from scipy.io import mmwrite,mmread
from FrequencyAnalysis.FrequencyResponse import FrequencyAnalysis
from Boundary.Set_Boundary import Boundary
from FlowSolver.FiniteElement import TaylorHood
from Plot.Matplot import contourf_cylinder
from Plot.Functionmap import Functionmap, Boundary_Coor

#%%
mesh=Mesh("./mesh/cylinder_26thousand.xml")
element=TaylorHood(mesh=mesh)
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                   }
BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(1.0),       'Boundary': 'top'},
                    'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(1.0),       'Boundary': 'bottom'},
                    'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,1.0)), 'Boundary': 'inlet'},
                    'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'},
                    'Outlet'  : {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet'}
                    }
InputOutputConditions={ 'Input':  {'Variable': 'OscillateAcceleration', 'Direction': 1},
                        'Output': {'Variable': 'PointVelocity','Direction': 1, 'Coordinate': [2.75, 0.0]}}
                        
#InputOutputConditions={ 'Input':  {'Variable': 'GaussianForce_dual', 'Radius': 0.6, 'Angle': 70.0, 'Sigma': 0.1},
#                        'Output': {'Variable': 'PointVelocity', 'Direction': 1, 'Coordinate': [2.75, 0.0]}}
#InputOutputConditions={ 'Input':  {'Variable': 'GaussianForce_single', 'Radius': 0.6, 'Angle': 70.0, 'Sigma': 0.1},
#                        'Output': {'Variable': 'GaussianVelocity', 'Direction': [0,1], 'Coordinate': [2.75, 0.0], 'Sigma': 0.5}}
              
#InputOutputConditions={ 'Input':  {'Variable': 'OscillateDisplacement', 'Direction': 1},
#                        'Output': {'Variable': 'PointVelocity', 'Direction': [1,1], 'Coordinate': [2.75, 0.5]}}
# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
#cylinder_coorx, cylinder_coory, index = Boundary_Coor(boundary.get_domain(), 5)


Re=[60]#range(50,115,5)
fre=[1e8]#np.loadtxt('eigen_fre.txt')[:,1]
omega_range=[]#[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]+list(np.arange(0.01,1.5,0.01))+list(np.arange(1.5,10,0.1))+[10,1.0e2,1.0e3,1.0e4,1.0e5,1.0e6]
locs=list(np.arange(1.0,5.0,0.01))

resp=np.zeros((np.size(locs),np.size(Re)))+np.zeros((np.size(locs),np.size(Re)))*1j
fre_response = FrequencyAnalysis(mesh, boundary)
for key in BoundaryConditions.keys():
    fre_response.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
wi=element.add_functions()
wr=element.add_functions()

for re in Re:
    gain=[]
    nu = Constant(1.0/re)
    path = os.getcwd()+"/base flow/cylinder_baseflow_newton_26thousand"+str(re).zfill(3)
    omega_range=[fre[Re.index(re)]]
#    savepath = '/mnt/f/dropbox_data/FrequencyAnalysis/FrequencyResponse/GaussianForce_system_state/cylinder_fre_respond_gaussianforce_state_' + str(re).zfill(3)+'part1'
#    timeseries_r = TimeSeries(savepath + '_realpart')
#    timeseries_i = TimeSeries(savepath + '_imagpart')
    for om in omega_range:
        start_time = tm.time()
        print('Re = %d ; Omega = %.8f' %(re, om))
        omega = Constant(om)
        fre_response.solve(omega=omega, nu=nu, path=path,useUmfpack=False, options=InputOutputConditions, ReuseLU = False)
        gain.append(fre_response.gain)
#        np.savetxt('/mnt/f/dropbox_data/FrequencyAnalysis/FrequencyResponse/GaussianForce_system_state/cylinder_fre_respond_Gaussianforce_vel_' + str(re).zfill(3)+'part1', zip(omega_range, np.real(gain), np.imag(gain)))
#        print('FR = ', fre_response.gain)
        wr.vector()[:] = np.ascontiguousarray(np.real(fre_response.state))
        wi.vector()[:] = np.ascontiguousarray(np.imag(fre_response.state))
#        timeseries_r.store(wr.vector(), om)
#        timeseries_i.store(wi.vector(), om)
#        del fre_response.Linv
#        del fre_response.state
#        gc.collect()
        print("--- %s seconds ---" % (tm.time() - start_time))
        


# # Coordinates of all dofs in the mixed space
# maping=Functionmap(element.functionspace)
# # Coordinates of dofs of first subspace of the mixed space
# V0_dofs_w0 = maping.get_subcoor(element.functionspace.sub(0).sub(0))
# w_0=maping.get_subvalue(element.functionspace.sub(0).sub(0),wi)
#
# V0_dofs_w1 = maping.get_subcoor(element.functionspace.sub(0).sub(1))
# w_1=maping.get_subvalue(element.functionspace.sub(0).sub(1),wi)
#
# V0_dofs_w2 = maping.get_subcoor(element.functionspace.sub(1))
# w_2=maping.get_subvalue(element.functionspace.sub(1),wi)
#
# #contourf_cylinder(V0_dofs_w0[:,0],V0_dofs_w0[:,1],w_0)
# contourf_cylinder(V0_dofs_w0[:,0],V0_dofs_w0[:,1],-w_0*om)
#
# #contourf_cylinder(V0_dofs_w1[:,0],V0_dofs_w1[:,1],w_1)
# contourf_cylinder(V0_dofs_w1[:,0],V0_dofs_w1[:,1],-w_1*om)
#
# contourf_cylinder(V0_dofs_w2[:,0],V0_dofs_w2[:,1],w_2)