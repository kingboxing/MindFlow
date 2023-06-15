from __future__ import print_function
from fenics import *
import os
import gc
import time as tm
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scila
import matplotlib.pyplot as plt
from Boundary.Set_Boundary import Boundary
from Boundary.Set_BoundaryCondition import BoundaryCondition
from FlowSolver.FiniteElement import TaylorHood
from SVD.MatrixCreator import MatrixCreator
from SVD.MatrixOperator import Matinv
from FrequencyAnalysis.FrequencyResponse import InputOutput
from Plot.Matplot import contourf_cylinder
from FrequencyAnalysis.EigenAnalysis import EigenAnalysis
from numpy.core.numeric import array

def sort_complex(a):
    b = array(a, copy=True)
    b.sort()
    index_sort=[]
    for i in b:
        index_sort.append(list(a).index(i))
    return b[::-1], index_sort[::-1]

mesh=refine(Mesh("./mesh/cylinder_74thousand_sym_60downstream_40upstream.xml"))
element=TaylorHood(mesh=mesh)
Re=100
nu=Constant(1.0/Re)
## get weight matrix
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 40.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -40.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and near(x[0], -60.0, tol)'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 60.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }
BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top'},
                    'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom'},
                    'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet'},
                    'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'}
                    }

# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
BC = BoundaryCondition(Functionspace=element.functionspace, boundary=boundary)
#for key in BoundaryConditions.keys():
#    BC.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
#matrices = MatrixCreator(mesh=mesh, functionspace=BC.functionspace,boundarycondition=BC.bcs) # initialise matrix creator
#P = matrices.MatP() # prolongation operator
#M = matrices.MatM()
#Mass = P*M*P.transpose()
### vector f
#InputOutputConditions={ 'Input':  {'Variable': 'GaussianForce_dual', 'Radius': 0.6, 'Angle': 70.0, 'Sigma': 0.1},
#                        'Output': {'Variable': 'PointVelocity', 'Direction': 1, 'Coordinate': [2.75, 0.0]}}
#IO_vec=InputOutput(mesh=mesh, boundary=boundary,element=element,bcs=BC.bcs)
#options=InputOutputConditions['Input']
#input_vec = eval("IO_vec."+IO_vec.dict_inputs[options['Variable']])
#options=InputOutputConditions['Output']
#output_vec = eval("IO_vec."+IO_vec.dict_outputs[options['Variable']])


"""
diagonalise with inverse eigenvalue method
"""
## 
BoundaryConditions['Outlet']  = {'FunctionSpace':  None,               'Value': 'FreeOutlet',        'Boundary': 'outlet'}
path = os.getcwd()+"/base flow/cylinder_baseflow_newton_74thousand_sym_refined_60downstream_40upstream"+str(Re).zfill(3)# './mean flow/cylinder_DNS_dt_001_IPCS_Re_100meanflow'#
eigen = EigenAnalysis(mesh, boundary, nu, path)#,dim='2.5D',lambda_z=Constant(0.0)
for key in BoundaryConditions.keys():
    eigen.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])

eigen.solve(k=50, sigma=-0.1, which='LR',solver='Explicit',inverse=False, ReuseLU=False)
vals=eigen.vals
vecs=eigen.vecs
vals_sort,ind_sort=sort_complex(vals)
info('vals[0]=%f+%f*1i' % (np.real(vals_sort[0]), np.imag(vals_sort[0])))
eigen.solve(k=50, sigma=-0.1, which='LR',solver='Explicit',inverse=True, ReuseLU=False)
valsT=eigen.vals
vecsT=eigen.vecs
valsT_sort,ind_sortT=sort_complex(valsT)

"""
save values and vectors
"""
re=Re
savepath='cylinder_eigenvalues_Re'+str(re).zfill(3)+"baseflow"
np.savetxt(savepath,zip(np.real(vals_sort.transpose()),np.imag(vals_sort.transpose())))

savepath='cylinder_inverse_eigenvalues_Re'+str(re).zfill(3)+"baseflow"
np.savetxt(savepath,zip(np.real(valsT_sort.transpose()),np.imag(valsT_sort.transpose())))

savepath_vecs='cylinder_eigenvecs_Re'+str(re).zfill(3)+"baseflow_2D"
timeseries_vecs=TimeSeries(savepath_vecs+'realpart')
for i in range(len(vals_sort)):
    element.w.vector()[:]=np.ascontiguousarray(np.real(vecs[:,ind_sort[i]]))
    timeseries_vecs.store(element.w.vector(), i)
timeseries_vecs=TimeSeries(savepath_vecs+'imagpart')
for i in range(len(vals_sort)):
    element.w.vector()[:]=np.ascontiguousarray(np.imag(vecs[:,ind_sort[i]]))
    timeseries_vecs.store(element.w.vector(), i)
    
savepath_vecs='cylinder_inverse_eigenvecs_Re'+str(re).zfill(3)+"baseflow_2D"
timeseries_vecs=TimeSeries(savepath_vecs+'realpart')
for i in range(len(valsT_sort)):
    element.w.vector()[:]=np.ascontiguousarray(np.real(vecsT[:,ind_sortT[i]]))
    timeseries_vecs.store(element.w.vector(), i)
timeseries_vecs=TimeSeries(savepath_vecs+'imagpart')
for i in range(len(valsT_sort)):
    element.w.vector()[:]=np.ascontiguousarray(np.imag(vecsT[:,ind_sortT[i]]))
    timeseries_vecs.store(element.w.vector(), i)

#vec_multiply=np.matrix(vecsT[:,ind_sortT].T)*np.matrix(vecs[:,ind_sort])
## k/(s-\lambda), calculate k=Lambda^-1*Scale*\Phi^-1*A^-1*f
#A_lu=spla.splu(eigen.A).solve
#k_coe=np.diag((valsT[ind_sortT[:]]))*np.diag((1.0/np.diag(vec_multiply)))*np.matrix(vecsT[:,ind_sortT[:]].T)*A_lu(input_vec.T)
#k_output=output_vec*np.matrix(vecs[:,ind_sort])
#k_above=k_output*np.diag(np.asarray(k_coe[:]).reshape(-1))
#
#b=eigen.M*np.matrix(vecs[:,ind_sort[:]])
#diag_lhs=np.diag((1.0/np.diag(vec_multiply)))*np.matrix(vecsT[:,ind_sortT[:]].T)*(A_lu(np.real(b).astype(eigen.A.dtype))+ 1j * A_lu(np.imag(b).astype(eigen.A.dtype)))
#np.savetxt('cylinder_fre_force_vel_v_Re'+str(Re).zfill(3)+'_eigenmodes_coefficients',zip(np.real(vals_sort),np.imag(vals_sort),np.real(k_above.T),np.imag(k_above.T)))

# frequency response
#num_mode=[50]
#omega_range=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]+list(np.arange(0.01,1.5,0.01))+list(np.arange(1.5,10,0.1))+[10,1.0e2,1.0e3,1.0e4,1.0e5,1.0e6]
#gain_matrix=np.zeros((len(omega_range)+1,len(num_mode)+1),dtype='complex')
#gain_matrix[1:,0]=np.asarray(omega_range)+np.asarray(omega_range)*1j
#gain_matrix[0,1:]=np.asarray(num_mode)+np.asarray(num_mode)*1j
#for num in num_mode:
#    gain=[]
#    index=num_mode.index(num)
#    for omega in omega_range:
#        eigenvals_matrix = np.diag(1.0/(omega*1j-vals_sort[0:num]))
#        gain.append((k_output[0,0:num]*eigenvals_matrix*k_coe[0:num,0])[0,0])
#    gain_matrix[1:,index+1]=np.asarray(gain)
#    plt.plot(np.log10(omega_range),20*np.log10(np.abs(gain_matrix[1:,index+1])))
#np.savetxt('cylinder_fre_force_vel_v_Re'+str(Re).zfill(3)+'_eigenmodes_realpart', np.real(gain_matrix))
#np.savetxt('cylinder_fre_force_vel_v_Re'+str(Re).zfill(3)+'_eigenmodes_imagpart', np.imag(gain_matrix))

"""
diagonalise with SVD
"""
### get eigenmodes
#eigenvec_r=element.add_functions()
#eigenvec_i=element.add_functions()
#eigenvec_abs=element.add_functions()
#savepath_vals = '/mnt/f/dropbox_data/FrequencyAnalysis/Eigenvals/Re100_2D_baseflow/cylinder_eigenvalues_Re'+str(Re).zfill(3)+'baseflow.txt'#'./cylinder_eigenvalues_Re'+str(Re).zfill(3)+'baseflow.txt'#'/media/bojin/ResearchData/DNSdata/'
#eigenvals = np.loadtxt(savepath_vals)
#ind=np.argmax(eigenvals[:,0])
#ind_sort=np.argsort(eigenvals[:,0])[::-1]
#eigenvals_sort=eigenvals[ind_sort,:]
#
#savepath_vecs='/mnt/f/dropbox_data/FrequencyAnalysis/Eigenvals/Re100_2D_baseflow/cylinder_eigenvecs_LM_Re'+str(Re).zfill(3)+'baseflow_2D'#'./cylinder_eigenvecs_LM_Re'+str(Re).zfill(3)+'baseflow_2D'#'/media/bojin/ResearchData/DNSdata/cylinder_eigenvecs_LM_Re'+str(re).zfill(3)+"baseflow_2D"
#timeseries_vecs_r=TimeSeries(savepath_vecs+'realpart')
#timeseries_vecs_i=TimeSeries(savepath_vecs+'imagpart')
#
#num=500
#eigvec=sp.lil_matrix((element.functionspace.dim(),num),dtype=np.complex128)
#for i in ind_sort:
#    j=list(ind_sort).index(i)
#    if j==num:
#        break    
#    timeseries_vecs_r.retrieve(eigenvec_r.vector(),i)
#    timeseries_vecs_i.retrieve(eigenvec_i.vector(),i)
#    eigenvec_abs.vector()[:]=np.abs(eigenvec_r.vector().get_local()+eigenvec_i.vector().get_local()*1j)
#    eigvec[:,j]=np.transpose(np.matrix(eigenvec_r.vector().get_local()))+np.transpose(np.matrix(eigenvec_i.vector().get_local()))*1j
#    info('%g' % j)
#
#eigvec_csc=eigvec.tocsc()
#del eigvec
#gc.collect()
#
#num_vec=500
#vec_rhs=eigen.M*eigvec_csc[:,0:num_vec]
##energy_1=(vec_rhs.H*vec_rhs).todense()
##vals=((vec_rhs.H*eigen.A*eigvec_csc[:,0]).todense())/energy_1
##k_coe=(vec_rhs.H*input_vec.transpose())/energy_1
#
#U, s, Vh =scila.svd(vec_rhs.todense(), full_matrices=False)
#vals=np.matrix(np.transpose(np.conj(Vh)))*np.matrix(scila.inv(scila.diagsvd(s, num_vec, num_vec)))*np.matrix(np.transpose(np.conj(U)))*((eigen.A*eigvec_csc[:,0:num_vec]).todense())
#
#k_coe=np.matrix(np.transpose(np.conj(Vh)))*np.matrix(scila.inv(scila.diagsvd(s, num_vec, num_vec)))*np.matrix(np.transpose(np.conj(U)))#*input_vec.transpose()
#element.w.vector()[:]=np.ascontiguousarray(np.transpose(np.real(k_coe[0,:])))



### test diagonalising matrix A
#u, vals, vt = spla.svds(eigvec_csc, k=450, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True)
#vals_reicp=np.zeros(len(vals))
#vals_reicp[0:len(vals[vals!=0.0])]=np.reciprocal(vals[vals!=0.0])
#vals_inverse = sp.diags(vals_reicp,shape=(len(vals),len(vals)))
#
#u_sp=sp.lil_matrix(u.shape,dtype=np.complex128)
#u_sp[0:u.shape[0],0:u.shape[1]]=u
#vt_sp=sp.lil_matrix(vt.shape,dtype=np.complex128)
#vt_sp[0:vt.shape[0],0:vt.shape[1]]=vt
#
#val_1=[]
#for i in range(450):
#    info('%g' % i)
#    val_1.append((vt_sp.H[i,:]*vals_inverse*u_sp.H*eigen.A*eigvec_csc[:,i]).todense())



## test singular values
#num_test=list(np.arange(50,1050,50))
#vals_test=np.zeros((len(num_test),10))
#for i in num_test:
#    vals = spla.svds(eigvec_csc[:,0:i-1], k=10, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=False)
#    vals_test[num_test.index(i),:]=vals
    
    
## get coefficients

#test=[500,600,700,800,900,1000,1100,1200]#list(np.arange(50,800,50))
#k_coe=sp.lil_matrix((element.functionspace.dim(),len(test)),dtype=np.complex128)
#for k_num in test:
#    info('start...')
#    u, vals, vt = spla.svds(eigvec_csc[:,0:k_num], k=int(k_num*0.8), ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True)
#    info('done...')
#    j=list(test).index(k_num)
#    vals_reicp=np.zeros(element.functionspace.dim())
#    vals_reicp[0:len(vals[vals!=0.0])]=np.reciprocal(vals[vals!=0.0])
#    vals_inverse = sp.diags(vals_reicp,shape=eigvec_csc.shape)
#    u_square=sp.lil_matrix((element.functionspace.dim(),element.functionspace.dim()),dtype=np.complex128)
#    u_square[0:u.shape[0],0:u.shape[1]]=u
#    vt_square=sp.lil_matrix((element.functionspace.dim(),element.functionspace.dim()),dtype=np.complex128)
#    vt_square[0:vt.shape[0],0:vt.shape[1]]=vt
#    k_coe[:,j]=((vt_square.H*(vals_inverse*(u_square.H*input_vec.transpose()))))
#    del u_square
#    del vt_square
#    del vals_inverse
#    del u
#    del vt
#    gc.collect()
