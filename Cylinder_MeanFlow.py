"""
Obtain mean flow from time series
"""


from __future__ import print_function
from fenics import *
import time as tm
import numpy as np
from FlowSolver.MeanFlow import *
from FlowSolver.FiniteElement import *
from Plot.Functionmap import *
from Plot.Matplot import *
from SVD.MatrixCreator import *

mesh=Mesh("./mesh/cylinder_26thousand.xml")
element = TaylorHood(mesh=mesh)
Re=60
omega=1.5
matrices = MatrixCreator(mesh=mesh, functionspace=element.functionspace)
Q = matrices.MatQf()
P = matrices.MatP()
weight=P*Q*P.T

mean=element.add_functions()
meanpath='./mean flow/cylinder_osci_omega1-5_dt_001_IPCS_26thousand060meanflow'  
timeseries_mean=TimeSeries(meanpath)
timeseries_mean.retrieve(mean.vector(),0.0)
#path='/mnt/f/DNSdata/DNS_OSCI/Re060/cylinder_osci_omega1-5_dt_001_IPCS_26thousand060'
#path='./nonlinear_force_self_consistent_model_Re60eigenmodes1part0'
path='/mnt/f/dropbox_data/FrequencyAnalysis/Self-consistent Model(base2mean)/Using Eigenmode/Re60_OSCI_omega1-5/meanflow_self_consistent_model_Re60eigenmodes_part0'
timeseries_test=TimeSeries(path)
times=timeseries_test.vector_times()
end_time=timeseries_test.vector_times()[-1]#74
energy=[]
test=element.add_functions()
for t in times:
    timeseries_test.retrieve(test.vector(),t)
    diff=np.matrix(test.vector().get_local()-mean.vector().get_local())
    energy.append((diff*weight*diff.H)[0,0])
np.savetxt('/mnt/f/dropbox_data/FrequencyAnalysis/Self-consistent Model(base2mean)/Using Eigenmode/Re60_OSCI_omega1-5/energy_diff_Re060',energy)
#start_time=timeseries_test.vector_times()[0]
#dt=0.1
#meanflow = MeanFlow(element=element, path=path, start=start_time, end=end_time, dt=dt)
#timeseries = TimeSeries("./mean flow/cylinder_osci_omega1-5_dt_001_IPCS_26thousand060meanflow")
#timeseries.store(meanflow.mean.vector(), 0.0)
#element.w.vector()[:]=meanflow.mean.vector().get_local()

#vorticity=project(grad(element.w[1])[0]-grad(element.w[0])[1])
#Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#vmax=np.max(np.abs(vorticity.vector().get_local()))
#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-vmax/2,vmaxf=vmax/2,colormap='seismic',colorbar='on',axis='off',figsize=(12.5, 5),colorbar_location={'fraction':0.08})


#mesh=Mesh("./mesh/cylinder_26thousand.xml")
#element = TaylorHood(mesh=mesh)
#base=element.add_functions()
#Re=50
#timeseries_base = TimeSeries("/mnt/F4E66F4EE66F0FE2/Dropbox/Code_reconstruction/RAPACK/base flow/cylinder_baseflow_newton_26thousand060")
#timeseries_base.retrieve(base.vector(), 0.0)
#
#timeseries_mean = TimeSeries("/mnt/F4E66F4EE66F0FE2/Dropbox/Code_reconstruction/RAPACK/DNSdata/controled_lift_impulse1dt_cylinder_dt_001_IPCS_26thousand060")
#time=np.arange(70,150.1,0.1)
#i=0
#for t in time:
#    timeseries_mean.retrieve(element.w.vector(), t)
#    element.w.vector()[:]=element.w.vector()[:]-base.vector()[:]
#    savepath='./figures/cylinder_osci_lift_perturbation_Re60_'+str(i).zfill(3)+'.png'
#    i=i+1
    #path = "/mnt/F4E66F4EE66F0FE2/Dropbox/Code_reconstruction/RAPACK/DNSdata/cylinder_dt_001_IPCS_26thousand"+str(Re).zfill(3)
    #start_time = 1103.79
    #end_time = 1198.94
    #dt = 0.01
    #meanflow = MeanFlow(element=element, path=path, start=start_time, end=end_time, dt=dt)
    #
    #timeseries = TimeSeries("./mean flow/cylinder_DNS_dt_001_IPCS_Re_"+str(Re).zfill(3)+"meanflow")
    #timeseries.store(meanflow.mean.vector(), 0.0)
    
    #timeseries_mean = TimeSeries("./mean flow/cylinder_DNS_dt_001_IPCS_Re_"+str(Re).zfill(3)+"meanflow")
    #timeseries_mean.retrieve(element.w.vector(), 0.0)
    ## Coordinates of all dofs in the mixed space
    #maping=Functionmap(element.functionspace)
    ## Coordinates of dofs of first subspace of the mixed space
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
    
#    vorticity=project(grad(element.w[1])[0]-grad(element.w[0])[1])
#    Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#    contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,20),ylim=(-5,5),savepath=savepath,vminf=-0.075,vmaxf=0.075,colormap='seismic',colorbar='off',axis='off',figsize=(8.8, 4))


## combine split functions into the mixed function
#mesh=Mesh("./mesh/cylinder_26thousand.xml")
#element1 = SplitPV(mesh=mesh)
#element2 = TaylorHood(mesh=mesh)
#base = element2.add_functions()
#baseflow = element1.add_functions()[0]
#Re=60
### base flow
#basepath = "/mnt/F4E66F4EE66F0FE2/Dropbox/Code_reconstruction/RAPACK/base flow/cylinder_baseflow_newton_26thousand"+str(Re).zfill(3)
#timeseries_base = TimeSeries(basepath)
#timeseries_base.retrieve(base.vector(), 0.0)
#assign(baseflow, base.sub(0))
### DNS path
##path = './DNSdata1/control_impulse001dt_cylinder_dt_001_IPCS_26thousand'+str(Re).zfill(3)
#path = "/mnt/F4E66F4EE66F0FE2/Dropbox/Code_reconstruction/RAPACK/DNSdata/controled_lift_impulse1dt_cylinder_dt_001_IPCS_26thousand"+str(Re).zfill(3)
#timeseries_vel = TimeSeries(path+'_velocity')
#timeseries_pre = TimeSeries(path+'_pressure')
#start_time = 0
#end_time = 200.0
#dt = 0.01
#time_series = np.arange(start_time,end_time+dt,dt)
#
###energy
#Q = assemble(inner(element1.tu,element1.v)*dx)
#per_energy=[]
### save path
#savepath = "/mnt/F4E66F4EE66F0FE2/Dropbox/Code_reconstruction/RAPACK/DNSdata/control_osci_lift_impulse1dt_cylinder_dt_001_IPCS_26thousand"+str(Re).zfill(3)
#timeseries_w = TimeSeries(path)
#for t in time_series[0:len(time_series)]:
#    t = round(t,2)
#    print('time= ', t)
#    timeseries_vel.retrieve(element1.u.vector(), t)
#    timeseries_pre.retrieve(element1.p.vector(), t)
#    
#    energy = Q*(element1.u.vector()-baseflow.vector())
#    per_energy.append(np.matrix((element1.u.vector()-baseflow.vector()).array())*np.matrix(energy.array()).T)
#    assign(element2.w.sub(0), element1.u)
#    assign(element2.w.sub(1), element1.p)
#    timeseries_w.store(element2.w.vector(), t)
#np.savetxt('./DNSdata/control_osci_lift_impulse1dt_cylinder_dt_001_IPCS_26thousand' + str(Re).zfill(3)+'perturbationenergy', zip(time_series,per_energy))