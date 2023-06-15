"""
Problem type : Perturbation equations in the frequency domain

Details : compute the first three global singular modes of the cylinder flow at Re = 45
    five boundaries : top and bottom symmetric wall,
    no-slip cylinder surface (d = 1), velocity inlet (0, 0), outlet with -p*n+nu*grad(u)*n=0

Method : Singular Value Decomposition
"""
from __future__ import print_function
from fenics import *
import os,sys
import time as tm
import numpy as np
import memory_profiler as mp
import scipy.sparse.linalg as spla
from RAPACK.SVD.interface import aslinearoperator
from scipy.io import mmread,mmwrite
from sksparse.cholmod import cholesky
from scipy.sparse.linalg import svds, eigs
from RAPACK.SVD.RSVD import randomized_svd
from RAPACK.Boundary.Set_Boundary import Boundary
from RAPACK.SVD.SingularMode import SingularMode
from RAPACK.SVD.MatrixOperator import MatInv, Mumpslu
from RAPACK.FlowSolver.FiniteElement import TaylorHood

mesh=Mesh("./mesh/cylinder_26thousand.xml")
path = os.getcwd()+"/mean flow/cylinder_DNS_dt_001_IPCS_Re_060meanflow"
element=TaylorHood(mesh=mesh)
bf=TimeSeries(path)
bf.retrieve(element.w.vector(),0.0)

#%%
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 25.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
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
# Reynolds number
Re = 60
nu = Constant(1.0 / Re)


# frequency of the singular modes
omega=Constant(1.0)

# initialise solver
singularmodes=SingularMode(mesh=mesh, boundary=boundary, omega=omega, nu=nu, baseflow=element.w,dim='2D')
# set boundary conditions
for key in BoundaryConditions.keys():
    singularmodes.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])

singularmodes.update_problem(MatK=None)
# start vector for iterative solver
vec_start = np.ones(singularmodes.functionspace.sub(0).dim())
factor = cholesky(singularmodes.Qf.tocsc(),ordering_method="natural")

Qf_lu = Mumpslu(singularmodes.Qf).solve
Qfinv = MatInv(singularmodes.Qf, A_lu=Qf_lu, trans='N',lusolver='mumps',echo=False)

D=spla.aslinearoperator(factor.L().transpose()*singularmodes.P.transpose()) * singularmodes.Linv * spla.aslinearoperator(singularmodes.P*singularmodes.Ibcs*factor.L())

#%%
def eigensvd(D):
    vals, vecs = spla.eigs(singularmodes.SVDexp, k=10, M=singularmodes.Qf, Minv=Qfinv, which='LR', v0=vec_start)
    return vals

def ssvd(D):
    u, s, vt = svds(D, k=10)
    s=np.sort(s)[::-1]
    return s
def rsvd(D):
    ur, sr, vtr = randomized_svd(D, 10, n_oversamples=5, n_iter=4)
    return sr
ssvdt=[]
rsvdt=[]
ssvdm=[]
rsvdm=[]

tt=tm.time()
mem=mp.memory_usage((eigensvd, (D,)),interval=1.0)
#mem=eigensvd(D)
tt0=tm.time()-tt
ssvdt.append(tt0)
ssvdm.append(np.max(mem)/1024)
info('SVD Time: %e \t SVD Mem: %e' %(tt0,np.max(mem)/1024))
#mmwrite('time_sparse_svd_'+str(inds)+'.mtx',np.matrix(ssvdt))
#mmwrite('mem_sparse_svd_'+str(inds)+'.mtx',np.matrix(ssvdm))


tt=tm.time()
mem=mp.memory_usage((ssvd, (D,)),interval=1.0)
#mem=ssvd(D)
tt1=tm.time()-tt
ssvdt.append(tt1)
ssvdm.append(np.max(mem)/1024)
info('SVD Time: %e \t SVD Mem: %e' %(tt1,np.max(mem)/1024))
#mmwrite('time_sparse_svd_'+str(inds)+'.mtx',np.matrix(ssvdt))
#mmwrite('mem_sparse_svd_'+str(inds)+'.mtx',np.matrix(ssvdm))

tt=tm.time()
mem=mp.memory_usage((rsvd, (D,)),interval=1.0)
#mem=rsvd(D)    
tt2=tm.time()-tt
rsvdt.append(tt2)
rsvdm.append(np.max(mem)/1024)
info('RSVD Time: %e \t RSVD Mem: %e' %(tt2,np.max(mem)/1024))
#mmwrite('time_random_svd_'+str(inds)+'.mtx',np.matrix(rsvdt))
#mmwrite('mem_random_svd_'+str(inds)+'.mtx',np.matrix(rsvdm))
#info('SVD Time: %e \t RSVD Time: %e \t Ratio: %e' %(np.average(ssvdt),np.average(rsvdt),np.average(rsvdt)/np.average(ssvdt)))