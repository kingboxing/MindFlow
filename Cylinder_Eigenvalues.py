from __future__ import print_function
from fenics import *
import os
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from Boundary.Set_Boundary import Boundary
from FrequencyAnalysis.EigenAnalysis import EigenAnalysis
from FlowSolver.FiniteElement import TaylorHood
from Plot.Matplot import contourf_cylinder
from Plot.Functionmap import Functionmap, Boundary_Coor


mesh=Mesh("./mesh/cylinder_26thousand.xml")
element=TaylorHood(mesh=mesh)
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.6, 0.6)) and between(x[1], (-0.6, 0.6))'},
                   }
BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top'},
                    'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom'},
                    'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet'},
                    'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'},
                    'Outlet'  : {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet'}
                    }

# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])

vals=[]
Re=[60]#range(50,115,5)
fre=[]
sig=[]
for re in Re:
    nu=Constant(1.0/re)
    path = os.getcwd()+"/base flow/cylinder_baseflow_newton_26thousand"+str(re).zfill(3)# './mean flow/cylinder_DNS_dt_001_IPCS_Re_100meanflow'#
    #path = './mean flow/cylinder_osci_omega1-5_self_consistent_26thousand060meanflow'    
    eigen = EigenAnalysis(mesh, boundary, nu=nu, path=path)#,dim='2.5D',lambda_z=Constant(0.0)
    for key in BoundaryConditions.keys():
        eigen.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
    """
    LM : largest magnitude
    SM : smallest magnitude
    LR : largest real part
    SR : smallest real part
    LI : largest imaginary part
    SI : smallest imaginary part
    """
    start_time = tm.time()
#    eigen.solve(k=1,sigma=0,which='LR')
#    vals=eigen.vals
    eigen.solve(k=2, sigma=0, which='LR',solver='Implicit',ReuseLU =False,inverse=False)
    end_time=tm.time()
    elapsed_time = end_time - start_time
    print('time %e' % (elapsed_time/60))
    fre.append(np.abs(np.imag(eigen.vals[0])))
    sig.append(np.real(eigen.vals[0]))
#np.savetxt('eigen_fre.txt', zip(Re,fre,sig))
#    start_time = tm.time()
#    eigen.solve(k=1,sigma=0,which='LR',v0=np.imag(eigen.vecs))
#    end_time=tm.time()
#    elapsed_time = end_time - start_time
#    print('time %e' % (elapsed_time/60))    
#    vals.append(eigen.vals[0])
#    savepath='cylinder_eigenvalues_Re'+str(re).zfill(3)+"baseflow"
#    np.savetxt(savepath,zip(np.real(eigen.vals.transpose()),np.imag(eigen.vals.transpose())))
#    plt.scatter(np.real(eigen.vals),np.imag(eigen.vals))
#    
#    savepath_vecs='cylinder_eigenvecs_LM_Re'+str(re).zfill(3)+"baseflow_2D"
#    timeseries_vecs=TimeSeries(savepath_vecs+'realpart')
#    for i in range(len(eigen.vals)):
#        element.w.vector()[:]=np.ascontiguousarray(np.real(eigen.vecs[:,i]))
#        timeseries_vecs.store(element.w.vector(), i)
#    timeseries_vecs=TimeSeries(savepath_vecs+'imagpart')
#    for i in range(len(eigen.vals)):
#        element.w.vector()[:]=np.ascontiguousarray(np.imag(eigen.vecs[:,i]))
#        timeseries_vecs.store(element.w.vector(), i)


#element.w.vector()[:]=np.ascontiguousarray(np.real(mode))
#vorticity=project(grad(element.w[1])[0]-grad(element.w[0])[1])
#Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-1,9),ylim=(-2,2))