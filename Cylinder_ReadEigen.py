from __future__ import print_function
from fenics import *
import os
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from FlowSolver.FiniteElement import TaylorHood
from Plot.Matplot import contourf_cylinder
from Plot.Functionmap import Functionmap, Boundary_Coor

mesh=Mesh("./mesh/cylinder_26thousand.xml")
element=TaylorHood(mesh=mesh)
eigenvec_r=element.add_functions()
eigenvec_i=element.add_functions()
eigenvec_abs=element.add_functions()
Re=100
savepath_vals = '/mnt/f/dropbox_data/FrequencyAnalysis/Eigenvals/Re'+str(Re).zfill(3)+'_2D_baseflow/cylinder_eigenvalues_Re'+str(Re).zfill(3)+'baseflow.txt'#'/media/bojin/ResearchData/DNSdata/'
eigenvals = np.loadtxt(savepath_vals)
ind=np.argmax(eigenvals[:,0])
ind_sort=np.argsort(eigenvals[:,0])[::-1]
eigenvals_sort=eigenvals[ind_sort,:]

#savepath_vecs='/mnt/f/dropbox_data/FrequencyAnalysis/Eigenvals/Re'+str(Re).zfill(3)+'_2D_baseflow/cylinder_eigenvecs_LM_Re'+str(Re).zfill(3)+'baseflow_2D'#'/media/bojin/ResearchData/DNSdata/cylinder_eigenvecs_LM_Re'+str(re).zfill(3)+"baseflow_2D"
savepath_vecs='/mnt/f/dropbox_data/FrequencyAnalysis/FrequencyResponse/GaussianForce_system_state/cylinder_fre_respond_gaussianforce_state_ZEROat2-753_'+str(Re).zfill(3)
timeseries_vecs_r=TimeSeries(savepath_vecs)
timeseries_vecs_i=TimeSeries(savepath_vecs)

#i=ind_sort[1]#54#87#44#56#
timeseries_vecs_r.retrieve(eigenvec_r.vector(),0)
timeseries_vecs_i.retrieve(eigenvec_i.vector(),1)
eigenvec_abs.vector()[:]=np.abs(eigenvec_r.vector().get_local()+eigenvec_i.vector().get_local()*1j)


#location_x=list(np.arange(1,5.01,0.02))+list(np.arange(5.2,10.1,0.2))
#location_y=list(np.arange(-3,-0.04,0.02))+list(np.arange(-0.05,0.0,0.005))
#location_y=location_y+[0.0]#+list(np.sort(-np.array(location_y)))
#location_x=list(np.arange(0.5,23,0.01))
#location_y=[0.0]
#
#fre=list(ind_sort[[1,3,6,8]])
#fre_response_vel=np.zeros((len(fre)+2,len(location_x)*len(location_y)+1),dtype='complex')
#aa=[]
#for jx in range(len(location_x)):
#    for jy in range(len(location_y)):
#        aa.append(jy+jx*len(location_y)+1)
#
#for i in range(len(fre)):
#    fre_response_vel[i+2,0]=eigenvals[fre[i],0]+eigenvals[fre[i],1]*1j
#    timeseries_vecs_r.retrieve(eigenvec_r.vector(), fre[i])
#    timeseries_vecs_i.retrieve(eigenvec_i.vector(), fre[i])
#    for jx in range(len(location_x)):
#        for jy in range(len(location_y)):
#            vals=eigenvec_r(location_x[jx],location_y[jy])[1]+eigenvec_i(location_x[jx],location_y[jy])[1]*1j
#            fre_response_vel[i+2,jy+jx*len(location_y)+1]=vals
#            
#            fre_response_vel[0,jy+jx*len(location_y)+1]=location_x[jx]+location_x[jx]*1j
#            fre_response_vel[1,jy+jx*len(location_y)+1]=location_y[jy]+location_y[jy]*1j
        
#np.savetxt('cylinder_fre_force_vel_v_eigenmode_Re'+str(Re).zfill(3)+'_2D_centerline_realpart', np.real(fre_response_vel))
#np.savetxt('cylinder_fre_force_vel_v_eigenmode_Re'+str(Re).zfill(3)+'_2D_centerline_imagpart', np.imag(fre_response_vel))

markers={'type': 'circle','radius':[0.04],'facecolor':'k','edgecolor':'k','location_x':[2.753],'location_y':[0.0],'symmetric axis':(1,0)}
function=Function(FunctionSpace(mesh, "Lagrange", 2))
savepath='/mnt/c/Users/bjin1/Dropbox/Dropbox/PhD Documents/2018/10th week 5.11-5.18/task2/fre_response_AtZero_Re'+str(Re).zfill(3)
assign(function,eigenvec_abs.sub(0).sub(1))
val_max=np.max(np.abs(function.vector().get_local()))
Vdofs_x = function.function_space().tabulate_dof_coordinates().reshape((-1, 2))
contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],function.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-val_max,vmaxf=val_max,
                  colormap='seismic',colorbar='off',axis='on',figsize=(12.5, 5),patch=[markers],savepath=savepath+'_abs.png')