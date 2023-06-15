from __future__ import print_function
from fenics import *
import numpy as np
import time
import gc
from FrequencyResponse.EigenAnalysis import EigenAnalysis
from SVD.SingularMode import SingularMode
from Boundary.Set_Boundary import Boundary
from FlowSolver.NS_Newton_Solver import *
from FlowSolver.FiniteElement import *
from FlowSolver.Error_RMSE import *
from SVD.MatrixCreator import *
from Plot.Matplot import *

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
    vec=np.matrix(np.ascontiguousarray(function.vector().get_local()))
    energy=np.asscalar(vec*weight*vec.H)
    if energy >DOLFIN_EPS:
        norm_vec=function.vector()/np.sqrt(energy)
    return norm_vec

## flag
Use_Singular_Mode=False
Restart=False
Re = 100
## collection of save path
if Use_Singular_Mode is False:
    mean_flow_savepath='meanflow_self_consistent_model_Re'+str(Re)+'eigenmodes'+'part0'
    nolinear_force_savepath='nonlinear_force_self_consistent_model_Re'+str(Re)+'eigenmodes'+'part0'
    eigval_savepath='self_consistent_eigenvalue_Re'+str(Re)+'eigenmodes'
    Af_savepath='self_consistent_Af_Re'+str(Re)+'eigenmodes'
    error_savepath='self_consistent_error_Re'+str(Re)+'eigenmodes'
    
    if Restart is True:
        mean_flow_readpath='meanflow_self_consistent_model_Re'+str(Re)+'eigenmodes'+'part0'
        nolinear_force_readpath='nonlinear_force_self_consistent_model_Re'+str(Re)+'eigenmodes'+'part0'
        
if Use_Singular_Mode is True:
    mean_flow_savepath='meanflow_self_consistent_model_Re'+str(Re)+'singularmodes'+'part0'
    nolinear_force_savepath='nonlinear_force_self_consistent_model_Re'+str(Re)+'singularmodes'+'part0'
    eigval_savepath='self_consistent_eigenvalue_Re'+str(Re)+'singularmodes'
    Af_savepath='self_consistent_Af_Re'+str(Re)+'singularmodes'
    error_savepath='self_consistent_error_Re'+str(Re)+'singularmodes'
    singularval_savepath='self_consistent_singularvalue_Re'+str(Re)+'singularmodes'
    
    if Restart is True:
        mean_flow_readpath='meanflow_self_consistent_model_Re'+str(Re)+'singularmodes'+'part0'
        nolinear_force_readpath='nonlinear_force_self_consistent_model_Re'+str(Re)+'singularmodes'+'part0'
    

## ...
mesh=Mesh("./mesh/cylinder_26thousand.xml")
element_split=SplitPV(mesh)
element=TaylorHood(mesh)    
matrices = MatrixCreator(mesh=mesh, functionspace=element.functionspace)
Q = matrices.MatQf()
P = matrices.MatP()
weight=P*Q*P.T

# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }
BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0)},
                    'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0)},
                    'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((1.0,0.0))},
                    'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0))},
                    'Outlet'  : {'Value': 'FreeOutlet'}
                    }

# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
# 1/Reynolds number & some functions to store results
nu = Constant(1.0 / Re)
f=element.add_functions()
w_pre=element.add_functions()
# initialise steady solver
solver = NS_Newton_Solver(mesh=mesh, boundary=boundary,nu=nu)
solver.solver_parameters(parameters={'newton_solver':{'linear_solver': 'mumps'}})
for key in BoundaryConditions.keys():
    solver.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
# initialise eigen solver
eigen = EigenAnalysis(mesh, boundary, nu)
for key in BoundaryConditions.keys():
    eigen.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])

if Use_Singular_Mode is True:
    # initialise SVD solver
    singularmodes=SingularMode(mesh=mesh, boundary=boundary, nu=nu)
    for key in BoundaryConditions.keys():
        singularmodes.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])

## initial values
a=0.5
dn=1.0
gamma=0.1
outer_loop=0
inner_loop=0
eigval=[]
error=[]
Af=[]    
Mean_flow=TimeSeries(mean_flow_savepath)
Nonlinear_force=TimeSeries(nolinear_force_savepath)
if Use_Singular_Mode is True:
    sigval=[]
if Restart is False:
    ## solve base flow 
    solver.solve(sourceterm=f.sub(0))
    assign(w_pre,solver.w)

if Restart is True:
    flow_restart=TimeSeries(mean_flow_readpath)
    flow_restart.retrieve(w_pre.vector(),flow_restart.vector_times()[-1])
    
    force_restart=TimeSeries(nolinear_force_readpath)
    force_restart.retrieve(f.vector(),flow_restart.vector_times()[-1])
    
    outer_loop=int(flow_restart.vector_times()[-1])/250
    inner_loop=int(flow_restart.vector_times()[-1])%250
    eig=np.loadtxt(eigval_savepath)
    eigval=list(eig[:,0]+eig[:,1]*1j)
    Af=list(np.loadtxt(Af_savepath))
    a=Af[-1]
    error=list(np.loadtxt(error_savepath))
    try:
        sigval
    except:
        pass
    else:
        sigval=list(np.loadtxt(singularval_savepath))
    
for j in np.arange(outer_loop,100):
    for i in np.arange(inner_loop,250):
        ## save results
        Mean_flow.store(w_pre.vector(), i+250*j)
        Nonlinear_force.store(f.vector(), i+250*j)
        
        np.savetxt(eigval_savepath,zip(np.real(eigval),np.imag(eigval)))
        np.savetxt(Af_savepath,zip(Af))
        np.savetxt(error_savepath,zip(error))
        ## start calculating
        eigen.solve(k=1,sigma=0,which='LR',baseflow=w_pre)
        assign(element.w, Bar_DivReynoldsStress(mesh,element, eigen.vecs))
        if Use_Singular_Mode is True:
            np.savetxt(singularval_savepath,zip(sigval))
            singularmodes.solve_SVD(k=1,omega=Constant(np.asscalar(np.imag(eigen.vals))),baseflow=w_pre)        
            assign(element.w, Bar_DivReynoldsStress(mesh,element, singularmodes.get_mode()[1]))
            sigval.append(np.real(singularmodes.get_mode()[2]))
        
        f.vector()[:]=-a*Norm_UnitEnergy(weight=weight,function=element.w)
        ## append results
        Af.append(a)
        eigval.append(eigen.vals[0])
        ## solve new base flow
        solver.solve(sourceterm=1.0*f.sub(0))
        ## error
        error.append(rmse(w_pre.vector().get_local(),solver.w.vector().get_local()))        
        ## relax factor
        w_pre.vector()[:]=(1-gamma)*w_pre.vector()[:]+gamma*solver.w.vector()[:]

        ## 
        print('energy=%e \t iteration= %d \t Error=%e \t Sigma=%e' %(a*a, i,error[-1],np.real(eigval[-1])))
        if error[-1] <= 1e-04:
            break
    ## new Af
    if Restart is True:
        inner_loop=0
    a=a/(1.0-dn*np.asscalar(np.real(eigen.vals)))**2
    if np.real(eigen.vals[0])<=1e-03:
        break













#vorticity=project(grad(solver.u[1])[0]-grad(solver.u[0])[1])
#Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-4,vmaxf=4,colormap='seismic',colorbar='off',axis='off',figsize=(10, 4))
#
#np.savetxt('self_consistent_eigenvalue_Re100',zip(np.real(eigval),np.imag(eigval)))
#timeseries_self = TimeSeries('self_consistent_Re100')
#timeseries_self.store(solver.w.vector(), 0.0)
#
#timeseries_self = TimeSeries('self_consistentforce_Re100')
#timeseries_self.store(element_split.u.vector(), 0.0)
#
#
#timeseries_mean = TimeSeries('./mean flow/cylinder_DNS_dt_001_IPCS_Re_100meanflow')
#mean=element.add_functions()
#timeseries_mean.retrieve(mean.vector(), 0.0)
#vorticity=project(grad(mean.sub(0)[1])[0]-grad(mean.sub(0)[0])[1])
#Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-4,vmaxf=4,colormap='seismic',colorbar='off',axis='off',figsize=(10, 4))
#
#timeseries_base = TimeSeries('./base flow/cylinder_baseflow_newton_26thousand100')
#base=element.add_functions()
#timeseries_base.retrieve(base.vector(), 0.0)
#vorticity=project(grad(base.sub(0)[1])[0]-grad(base.sub(0)[0])[1])
#Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-4,vmaxf=4,colormap='seismic',colorbar='off',axis='off',figsize=(10, 4))

#assign(element_split.u,modef0_r.sub(0))
#scale=[0]#[0.25,0.5,0.75,1.0]
#for i in scale: 
#    # solve the problem
#    solver.solve(sourceterm=i*element_split.u)
#    # print drag and lift
#    print('RE= %d     drag= %e    lift= %e' % (i, solver.get_force(bodymark=5,direction=0) , solver.get_force(bodymark=5,direction=1)))



#pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/FFT_samplingtime_696.5/fft_perturbation_field.npy')  
#time_step = 0.2
#time=np.arange(0.2,696.7,time_step)
#freqs = np.fft.fftfreq(len(time), time_step)*2*pi  
#pertur_fre_1st=pertur_fre[np.where(np.round(freqs,3)==1.046)[0][0],:]*1.0
#del pertur_fre
#gc.collect()   
#    
#mode0_r=element.add_functions()
#mode0_i=element.add_functions()
#
#mode0_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_1st))
#mode0_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_1st))
#
#modef0_r=element.add_functions()
#modef0_i=element.add_functions()
#
#assign(modef0_r.sub(0),project( 
#                            -1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_r.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_i.sub(0))))
#                            +-1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_r.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_i.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
#assign(modef0_i.sub(0),project(
#                            -1.0*(-dot(mode0_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_r.sub(0))))
#                            +-1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_i.sub(0)))-dot(mode0_i.sub(0), nabla_grad(mode0_r.sub(0)))),element_split.functionspace_V,solver_type='gmres'))  
#
#energy=np.matrix(P.transpose()*modef0_r.vector())*Q*np.matrix(P.transpose()*modef0_r.vector()).T
