from __future__ import print_function
from fenics import *
import sys
import gc
import time as tm
import numpy as np
from FlowSolver.MeanFlow import *
from FlowSolver.FiniteElement import *
from Plot.Functionmap import *
from Plot.Matplot import *
from SVD.MatrixCreator import *


# mesh and element
Re = 60
mesh=Mesh("./mesh/cylinder_26thousand.xml")
element = TaylorHood(mesh=mesh)
# mean flow
meanpath='./mean flow/cylinder_osci_omega1-5_dt_001_IPCS_26thousand060meanflow'
timeseries_mean = TimeSeries(meanpath)
mean=element.add_functions()
timeseries_mean.retrieve(mean.vector(), 0.0)
# sampling time
time_step = 0.2
time=np.arange(229.8,1000.0,time_step)
#freqs = np.fft.fftfreq(len(time), time_step)*2*pi
# matrices
matrices = MatrixCreator(mesh=mesh, functionspace=element.functionspace)
Q = matrices.MatQf()
P = matrices.MatP()
## exec part
block=1
## u*grad(u)-bar(u*grad(u))
if block==1 or block==0:
    element_split=SplitPV(mesh)
    #dnspath='/media/bojin/ResearchData/dropbox_data/DNS/DNS_Newton_Re100/cylinder_DNS_dt_001_newton_Re_100'
    dnspath='/mnt/f/DNSdata/DNS_OSCI/Re060/cylinder_osci_omega1-5_dt_001_IPCS_26thousand060'
    timeseries_dns = TimeSeries(dnspath)
    dns=element.add_functions()
    pertur=[]
    nonlinear=[]
    for t in time:
        vorticity=project(grad(element.w[1])[0]-grad(element.w[0])[1])
        Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
        vmax=np.max(np.abs(vorticity.vector().get_local()))
        contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-vmax/2,vmaxf=vmax/2,colormap='seismic',colorbar='on',axis='off',figsize=(12.5, 5),colorbar_location={'fraction':0.08})

        j=list(time).index(t)
        info("time: %g" %j)
        timeseries_dns.retrieve(dns.vector(), t)
        pertur=element.add_functions()
        pertur.vector()[:]=dns.vector()-mean.vector()
        nonlinear.append(element.add_functions())
        assign(nonlinear[j].sub(0),project(dot(pertur.sub(0), nabla_grad(pertur.sub(0))),element_split.functionspace_V,solver_type='gmres'))
        
        if j==0:
            mean_per=0.5*nonlinear[j].vector()
        elif j==len(time)-1:
            mean_per=mean_per+0.5*nonlinear[j].vector()
        else:
            mean_per=mean_per+nonlinear[j].vector()
    mean_pertur=element.add_functions()
    mean_pertur.vector()[:]=mean_per/(len(time)-1)
    
    perturbations=np.zeros((len(time),element.functionspace.dim()))
    nonlinearpath='/mnt/f/DNSdata/DNS_OSCI/Re060/cylinder_osci_omega1-5_dt_001_IPCS_26thousand060_nonlinear_perturbation_mean'
    timeseries_nonlinear = TimeSeries(nonlinearpath)
    timeseries_nonlinear.store(mean_pertur.vector(), 0.0)
    for t in time:
        j=list(time).index(t)
        info("time: %g" %j)
        
        perturbations[j,:]=np.matrix((nonlinear[j].vector()-mean_pertur.vector()).get_local())
    np.save('/mnt/f/DNSdata/DNS_OSCI/Re060/perturbations.npy', perturbations)
        
    fx=project(mean_pertur[0])   
    Vdofs_x = fx.function_space().tabulate_dof_coordinates().reshape((-1, 2))
    vmax=np.max(np.abs(fx.vector().get_local()))
    contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],-fx.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-vmax,vmaxf=vmax,colormap='seismic',colorbar='on',axis='off',figsize=(12.5, 5),colorbar_location={'fraction': 0.08})

    fy=project(mean_pertur[1])      
    Vdofs_x = fy.function_space().tabulate_dof_coordinates().reshape((-1, 2))
    vmax=np.max(np.abs(fx.vector().get_local()))
    contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],-fy.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-vmax,vmaxf=vmax,colormap='seismic',colorbar='on',axis='off',figsize=(12.5, 5),colorbar_location={'fraction': 0.08})

## fft(u*grad(u)-bar(u*grad(u)))
if block==2 or block==0:
    perturbations = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/perturbations.npy')
    pertur_fre=np.zeros((len(time),element.functionspace.dim()),dtype=np.complex_)
    for i in np.arange(element.functionspace.dim()):
        info("count: %g" %i)
        pertur_fre[:,i] = np.fft.fft(perturbations[:,i])/len(time)
    np.save('/mnt/F4E66F4EE66F0FE2/dropbox_data/fft_perturbations.npy', pertur_fre)

## energy spectrum
if block==3 or block==0:
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/fft_perturbations.npy')
    energy = []
    for i in np.arange(len(time)):
        info("time: %g" %i)
        vec = np.matrix(P.transpose()*pertur_fre[i,:])
        energy.append(np.asscalar(vec*Q*vec.H))
    energy=np.real(np.asanyarray(energy))
    freqs = np.fft.fftfreq(len(energy), time_step)*2*pi
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], energy[idx])
    np.savetxt('/mnt/F4E66F4EE66F0FE2/dropbox_data/energy_nonlinear_pertur',zip(freqs[idx], energy[idx]))

## load SVD mode
if block==4 or block ==0:
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/fft_perturbations.npy')
    freqs = np.fft.fftfreq(len(time), time_step)*2*pi
    omega = (np.fft.fftfreq(len(time), time_step)*2*pi)[1:(len(time)/2-1)]
    omega = omega[np.where(omega<8.0)]
    # load sigular values and frequencies
    valuepath = '/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_SVDmodeEigen_Re'+str(Re)+'.txt'
    singularvalue = np.loadtxt(valuepath)
    # get path
    mode0path = []
    mode1path = []
    mode2path = []
    for ome in omega:
        mode0path.append('/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_mode0_Re'+str(Re)+'_Omega'+str(round(ome, 2)))
        mode1path.append('/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_mode1_Re'+str(Re)+'_Omega'+str(round(ome, 2)))
        mode2path.append('/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_mode2_Re'+str(Re)+'_Omega'+str(round(ome, 2)))
    # load first three modes
    mode_r=element.add_functions()
    mode_i=element.add_functions()
    mode0=np.zeros((len(omega),element.functionspace.dim()),dtype=np.complex_)
    mode1=np.zeros((len(omega),element.functionspace.dim()),dtype=np.complex_)
    mode2=np.zeros((len(omega),element.functionspace.dim()),dtype=np.complex_)
    for ome in omega:
        info('omega: %g' %ome)
        i=list(omega).index(ome)
        timeseries_mode = TimeSeries(mode0path[i])
        timeseries_mode.retrieve(mode_r.vector(), 0.0)
        timeseries_mode.retrieve(mode_i.vector(), 1.0)
        mode0[i,:]=np.matrix(mode_r.vector().get_local()+mode_i.vector().get_local()*1j)
        timeseries_mode = TimeSeries(mode1path[i])
        timeseries_mode.retrieve(mode_r.vector(), 0.0)
        timeseries_mode.retrieve(mode_i.vector(), 1.0)
        mode1[i,:]=np.matrix(mode_r.vector().get_local()+mode_i.vector().get_local()*1j)
        timeseries_mode = TimeSeries(mode2path[i])
        timeseries_mode.retrieve(mode_r.vector(), 0.0)
        timeseries_mode.retrieve(mode_i.vector(), 1.0)
        mode2[i,:]=np.matrix(mode_r.vector().get_local()+mode_i.vector().get_local()*1j)
    
    scale0=[]
    scale1=[]
    scale2=[]
    for ome in omega:
        info('omega: %g' %ome)
        i=list(freqs).index(ome)
        j=list(omega).index(ome)
        scale0.append(np.asscalar(np.matrix(P.transpose()*pertur_fre[i,:])*Q*np.matrix(P.transpose()*mode0[j,:]).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*mode0[j,:])*Q*np.matrix(P.transpose()*mode0[j,:]).H))*np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre[i,:])*Q*np.matrix(P.transpose()*pertur_fre[i,:]).H))))
        scale1.append(np.asscalar(np.matrix(P.transpose()*pertur_fre[i,:])*Q*np.matrix(P.transpose()*mode1[j,:]).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*mode1[j,:])*Q*np.matrix(P.transpose()*mode1[j,:]).H))*np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre[i,:])*Q*np.matrix(P.transpose()*pertur_fre[i,:]).H))))
        scale2.append(np.asscalar(np.matrix(P.transpose()*pertur_fre[i,:])*Q*np.matrix(P.transpose()*mode2[j,:]).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*mode2[j,:])*Q*np.matrix(P.transpose()*mode2[j,:]).H))*np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre[i,:])*Q*np.matrix(P.transpose()*pertur_fre[i,:]).H))))
    plt.plot(omega, abs((np.asarray(scale0))),omega, abs((np.asarray(scale1))),omega, abs((np.asarray(scale2))))
    np.savetxt('/mnt/F4E66F4EE66F0FE2/dropbox_data/nonlinear_force_fit',zip(omega,abs((np.asarray(scale0))),abs((np.asarray(scale1))),abs((np.asarray(scale2)))))
#field=[]
#time=np.arange(0,100.1,0.1)
#
#for t in time:
#    field.append(element.add_functions())
#    j=list(time).index(t)
#    for ome in omega:
#        i = list(omega).index(ome)
#        cosr=np.cos(ome*t)
#        sini=np.sin(ome*t)
#        field[j].vector()[:]=field[j].vector() + singularvalue[i][1]*0.01/1.6*(2*cosr*mode_r[i].vector()-2*sini*mode_i[i].vector())
#
##time=np.arange(0,100.1,0.1)
##for t in time:
##    j=list(time).index(t)
##    savepath='/mnt/d/dropbox_data/figures/cylinder_omegaall_Re'+str(Re)+'_'+str(j)+'.png'
#j=500
#vorticity=project(grad(field[j][1])[0]-grad(field[j][0])[1])
#Vdofs_x = vorticity.function_space().tabulate_dof_coordinates().reshape((-1, 2))
#contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],vorticity.vector(),xlim=(-2,20),ylim=(-10,10),colormap='seismic',colorbar='on',axis='off',figsize=(8.8, 8))
