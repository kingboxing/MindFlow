from __future__ import print_function
from fenics import *
import sys
import gc
import time as tm
import numpy as np
from FlowSolver.MeanFlow import *
from FlowSolver.FiniteElement import *
from Plot.Matplot import *
from SVD.MatrixCreator import *
from FlowSolver.NS_Newton_Solver import *
from Boundary.Set_Boundary import Boundary
def Bar_DivReynoldsStress(mesh,element,complexmode):
    element_split=SplitPV(mesh)
    mode0_r=element.add_functions()
    mode0_i=element.add_functions()
    
    modef0_r=element.add_functions()
    modef0_i=element.add_functions()
    
    mode0_r.vector()[:]=np.ascontiguousarray(np.real(complexmode))
    mode0_i.vector()[:]=np.ascontiguousarray(np.imag(complexmode))
    
    assign(modef0_r.sub(0),project( 
                                1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_r.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_i.sub(0))))
                                +1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_r.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_i.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
    assign(modef0_i.sub(0),project(
                                1.0*(-dot(mode0_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_r.sub(0))))
                                +1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_i.sub(0)))-dot(mode0_i.sub(0), nabla_grad(mode0_r.sub(0)))),element_split.functionspace_V,solver_type='gmres'))  
    return modef0_r

def Norm_UnitEnergy(weight=None,function=None):
    if weight is None:
        weight=identity(function.vector().size(0), dtype='float64', format='csr')
    try:
        vec=np.matrix(np.ascontiguousarray(function.vector().get_local()))
    except:
        vec=np.matrix(np.ascontiguousarray(function))
    else:
        pass
    energy=np.real(np.asscalar(vec*weight*vec.H))
    if energy >DOLFIN_EPS:
        try:
            norm_vec=function.vector()/np.sqrt(energy)
        except:
            norm_vec=function/np.sqrt(energy)
        else:
            pass
    return norm_vec, energy
    


mesh=Mesh("./mesh/cylinder_26thousand.xml")
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])

element = TaylorHood(mesh=mesh)
Re=60
omega=1.5
mag=0.1
Cn=0.5*mag
solver = NS_Newton_Solver(mesh=mesh, boundary=boundary, nu=Constant(1.0/Re))
solver.update_problem()
Af=np.loadtxt('/mnt/f/dropbox_data/FrequencyAnalysis/Self-consistent Model(base2mean)/Using Eigenmode/Re60_OSCI_omega1-5/self_consistent_Af_Re60eigenmodes')
fre=np.loadtxt('/mnt/f/dropbox_data/FrequencyAnalysis/Self-consistent Model(base2mean)/Using Eigenmode/Re60_OSCI_omega1-5/self_consistent_eigenvalue_Re60eigenmodes')
matrices = MatrixCreator(mesh=mesh, functionspace=element.functionspace)
Q = matrices.MatQf()
P = matrices.MatP()
weight=P*Q*P.T

mean=element.add_functions()
meanpath='/mnt/f/dropbox_data/FrequencyAnalysis/Self-consistent Model(base2mean)/Using Eigenmode/Re60_OSCI_omega1-5/meanflow_self_consistent_model_Re60eigenmodes_part0'
timeseries_mean=TimeSeries(meanpath)
timeseries_mean.retrieve(mean.vector(),timeseries_mean.vector_times()[74])

eigvecs_r=element.add_functions()
eigvecs_i=element.add_functions()
eigenpath='/mnt/f/dropbox_data/FrequencyAnalysis/Self-consistent Model(base2mean)/Using Eigenmode/Re60_OSCI_omega1-5/eigen_mode_self_consistent_model_Re60eigenmodes_part0'
timeseries_eig_r=TimeSeries(eigenpath+'realpart')
timeseries_eig_r.retrieve(eigvecs_r.vector(),timeseries_eig_r.vector_times()[0])
timeseries_eig_i=TimeSeries(eigenpath+'imagpart')
timeseries_eig_i.retrieve(eigvecs_i.vector(),timeseries_eig_i.vector_times()[0])

eigvecs=eigvecs_r.vector().get_local()+eigvecs_i.vector().get_local()*1j

assign(element.w, Bar_DivReynoldsStress(mesh,element, eigvecs))

force_1=np.matrix(element.w.vector().get_local())
energy=(force_1*weight*force_1.H)[0,0]
coe_eig=pow(Af[74]**2/energy,0.25)

##
respondvecs_r=element.add_functions()
respondvecs_i=element.add_functions()
respondpath='/mnt/f/dropbox_data/FrequencyAnalysis/Self-consistent Model(base2mean)/Using Eigenmode/Re60_OSCI_omega1-5/fre_response_consistent_model_Re60omega1.5eigenmodes_part0'
timeseries_respond_r=TimeSeries(respondpath+'realpart')
timeseries_respond_r.retrieve(respondvecs_r.vector(),timeseries_respond_r.vector_times()[73])
timeseries_respond_i=TimeSeries(respondpath+'imagpart')
timeseries_respond_i.retrieve(respondvecs_i.vector(),timeseries_respond_i.vector_times()[73])

respondvecs=respondvecs_r.vector().get_local()+respondvecs_i.vector().get_local()*1j

times=np.arange(0.0,1000,0.1)
fre_1=fre[75,1]
fre_2=omega
flow_reconstruct=element.add_functions()
lift=[]
drag=[]
flow_reconstruct.vector()[:]=np.ascontiguousarray(np.real(coe_eig*eigvecs))
assign(solver.w,flow_reconstruct)
lift_eig=solver.get_force(bodymark=5,direction=1)
flow_reconstruct.vector()[:]=np.ascontiguousarray(np.imag(coe_eig*eigvecs))
assign(solver.w,flow_reconstruct)
lift_eig+=solver.get_force(bodymark=5,direction=1)*1j

flow_reconstruct.vector()[:]=np.ascontiguousarray(np.real(respondvecs))
assign(solver.w,flow_reconstruct)
lift_resp=solver.get_force(bodymark=5,direction=1)
flow_reconstruct.vector()[:]=np.ascontiguousarray(np.imag(respondvecs))
assign(solver.w,flow_reconstruct)
lift_resp+=solver.get_force(bodymark=5,direction=1)*1j
for t in times:
    info('time= %e' %t)
    lift.append((2.0*np.real(lift_eig)*cos(fre_1*t)/sin(-0.2501)*sin(2.104)-2.0*np.imag(lift_eig)*sin(fre_1*t)/cos(-0.2501)*cos(2.104))+(2.0*np.real(lift_resp)*cos(fre_2*t)/sin(1.4045)*sin(0.3613)-2.0*np.imag(lift_resp)*sin(fre_2*t)/cos(1.4045)*cos(0.3613)))
    drag.append(0.0)
#    flow_reconstruct.vector()[:]=2.0*np.real(coe_eig*eigvecs)*cos(fre_1*t)-2.0*np.imag(coe_eig*eigvecs)*sin(fre_1*t)+2.0*np.real(respondvecs)*cos(fre_2*t)-2.0*np.imag(respondvecs)*sin(fre_2*t)
#    flow_reconstruct.vector()[:]=mean.vector().get_local()+flow_reconstruct.vector().get_local()
#    assign(solver.w,flow_reconstruct)
#    lift.append(solver.get_force(bodymark=5,direction=1))
np.savetxt('./cylinder_self_consistent_draglift_phaseshifted_reconstruct',zip(times,drag,lift))

#vorticity=project(grad(flow_reconstruct[1])[0]-grad(flow_reconstruct[0])[1])
#Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#vmax=np.max(np.abs(vorticity.vector().get_local()))
#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-vmax/2,vmaxf=vmax/2,colormap='seismic',colorbar='on',axis='off',figsize=(12.5, 5),colorbar_location={'fraction':0.08})
