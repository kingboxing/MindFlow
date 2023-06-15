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
from Boundary.Set_Boundary import Boundary
from SVD.SingularMode import SingularMode
from FrequencyResponse.MatrixAssemble import *
import scipy.sparse.linalg as spla

# mesh and element
Re = 100
mesh=Mesh("./mesh/cylinder_26thousand.xml")
element = TaylorHood(mesh=mesh)
element_split=SplitPV(mesh)
# mean flow
meanpath='/mnt/F4E66F4EE66F0FE2/Dropbox/Code_reconstruction/RAPACK/mean flow/cylinder_DNS_dt_001_IPCS_Re_100meanflow'
timeseries_mean = TimeSeries(meanpath)
mean=element.add_functions()
timeseries_mean.retrieve(mean.vector(), 0.0)
# sampling time
time_step = 0.2
time=np.arange(0.2,696.7,time_step)
#freqs = np.fft.fftfreq(len(time), time_step)*2*pi
# matrices
matrices = MatrixCreator(mesh=mesh, functionspace=element.functionspace)
Q = matrices.MatQf()
P = matrices.MatP()
## exec part
block=7
## u*grad(u)-bar(u*grad(u))
if block==1 or block==0:
    #dnspath='/media/bojin/ResearchData/dropbox_data/DNS/DNS_Newton_Re100/cylinder_DNS_dt_001_newton_Re_100'
    dnspath='/media/bojin/ResearchData/DNSdata/cylinder_dt_01_DNS_IPCS_26thousand_Re100'
    timeseries_dns = TimeSeries(dnspath)
    dns=element.add_functions()
    pertur=[]
    nonlinear=[]
    for t in time:
        j=list(time).index(t)
        info("time: %g" %j)
        timeseries_dns.retrieve(dns.vector(), t)
        pertur.append(element.add_functions())
        pertur[j].vector()[:]=dns.vector()-mean.vector()
       
        if j==0:
            mean_per=0.5*pertur[j].vector()
        elif j==len(time)-1:
            mean_per=mean_per+0.5*pertur[j].vector()
        else:
            mean_per=mean_per+pertur[j].vector()
    mean_pertur=element.add_functions()
    mean_pertur.vector()[:]=mean_per/(len(time)-1)
    
    perturbations=np.zeros((len(time),element.functionspace.dim()))
    perturpath='/media/bojin/ResearchData/dropbox_data/DNS/DNS_Newton_Re100/cylinder_DNS_dt_001_newton_Re_100_nonlinear_perturbation'
    timeseries_pertur = TimeSeries(perturpath)
    for t in time:
        j=list(time).index(t)
        info("time: %g" %j)
        timeseries_pertur.store(pertur[j].vector(),t)#-mean_pertur.vector(), t)
        perturbations[j,:]=np.matrix((pertur[j].vector().get_local()))#-mean_pertur.vector()).get_local())
    np.save('/mnt/F4E66F4EE66F0FE2/dropbox_data/perturbation_field.npy', perturbations)

## fft(u*grad(u)-bar(u*grad(u)))
if block==2 or block==0:
    perturbations = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/perturbation_field.npy')
    pertur_fre=np.zeros((len(time),element.functionspace.dim()),dtype=np.complex_)
    for i in np.arange(element.functionspace.dim()):
        info("count: %g" %i)
        pertur_fre[:,i] = np.fft.fft(perturbations[:,i])/len(time)
    np.save('/mnt/F4E66F4EE66F0FE2/dropbox_data/fft_perturbation_field.npy', pertur_fre)

## energy spectrum
if block==3 or block==0:
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/fft_perturbation_field.npy')
    energy = []
    for i in np.arange(len(time)):
        info("time: %g" %i)
        vec = np.matrix(P.transpose()*pertur_fre[i,:])
        energy.append(np.asscalar(vec*Q*vec.H))
    energy=np.real(np.asanyarray(energy))
    freqs = np.fft.fftfreq(len(energy), time_step)*2*pi
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], energy[idx])
    np.savetxt('/mnt/F4E66F4EE66F0FE2/dropbox_data/energy_velocity_pertur',zip(freqs[idx], energy[idx]))

## load SVD mode
if block==4 or block ==0:
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/fft_perturbation_field.npy')
    freqs = np.fft.fftfreq(len(time), time_step)*2*pi
    omega = (np.fft.fftfreq(len(time), time_step)*2*pi)[1:(len(time)/2-1)]
    omega = omega[np.where(omega<8.0)]
    # load sigular values and frequencies
    #valuepath = '/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_SVDmodeEigen_Re'+str(Re)+'.txt'
    #singularvalue = np.loadtxt(valuepath)
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
        timeseries_mode.retrieve(mode_r.vector(), 2.0)
        timeseries_mode.retrieve(mode_i.vector(), 3.0)
        mode0[i,:]=np.matrix(mode_r.vector().get_local()+mode_i.vector().get_local()*1j)
        timeseries_mode = TimeSeries(mode1path[i])
        timeseries_mode.retrieve(mode_r.vector(), 2.0)
        timeseries_mode.retrieve(mode_i.vector(), 3.0)
        mode1[i,:]=np.matrix(mode_r.vector().get_local()+mode_i.vector().get_local()*1j)
        timeseries_mode = TimeSeries(mode2path[i])
        timeseries_mode.retrieve(mode_r.vector(), 2.0)
        timeseries_mode.retrieve(mode_i.vector(), 3.0)
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
    np.savetxt('/mnt/F4E66F4EE66F0FE2/dropbox_data/perturbation_response_fit',zip(omega,abs((np.asarray(scale0))),abs((np.asarray(scale1))),abs((np.asarray(scale2)))))

##
if block==5 or block==0:
    # store boundary locations and conditions in two dicts
    BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                       'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                       'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                       'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                       'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                       }
    BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top'},
                        'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom'},
                        'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet'},
                        'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'},
                        'Outlet': {'FunctionSpace': 'V.sub(1)',          'Value': Constant(0.0), 'Boundary': 'outlet'}
                        }
    
    # define boundaries and mark them with integers
    boundary=Boundary(mesh)
    for key in BoundaryLocations.keys():
         boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
    # Reynolds number
    nu = Constant(1.0 / Re)
    # initialise solver
    singularmodes=SingularMode(mesh=mesh, boundary=boundary, nu=nu, path=meanpath,omega=Constant(3.139))
    # set boundary conditions
    for key in BoundaryConditions.keys():
        singularmodes.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
    
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/FFT_samplingtime_696.5/fft_perturbation_field.npy')
    freqs = np.fft.fftfreq(len(time), time_step)*2*pi  
    pertur_fre_1st=pertur_fre[np.where(np.round(freqs,3)==1.046)[0][0],:]*1.0
    pertur_fre_n1st=pertur_fre[np.where(np.round(freqs,3)==-1.046)[0][0],:]*1.0
    
    pertur_fre_2nd=pertur_fre[np.where(np.round(freqs,3)==2.093)[0][0],:]*1.0
    pertur_fre_n2nd=pertur_fre[np.where(np.round(freqs,3)==-2.093)[0][0],:]*1.0
    
    pertur_fre_3rd=pertur_fre[np.where(np.round(freqs,3)==3.139)[0][0],:]*1.0
    pertur_fre_4th=pertur_fre[np.where(np.round(freqs,3)==4.185)[0][0],:]*1.0
    pertur_fre_5th=pertur_fre[np.where(np.round(freqs,3)==5.231)[0][0],:]*1.0
    del pertur_fre
    gc.collect()    
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/FFT_samplingtime_696.5/fft_perturbations.npy')
    perturforce_fre_1st=pertur_fre[np.where(np.round(freqs,3)==1.046)[0][0],:]*1.0
    
    perturforce_fre_2nd=pertur_fre[np.where(np.round(freqs,3)==2.093)[0][0],:]*1.0
    perturforce_fre_n2nd=pertur_fre[np.where(np.round(freqs,3)==-2.093)[0][0],:]*1.0
    
    perturforce_fre_3rd=pertur_fre[np.where(np.round(freqs,3)==3.139)[0][0],:]*1.0
    perturforce_fre_4th=pertur_fre[np.where(np.round(freqs,3)==4.185)[0][0],:]*1.0
    perturforce_fre_5th=pertur_fre[np.where(np.round(freqs,3)==5.231)[0][0],:]*1.0
    del pertur_fre
    gc.collect()  
    #energy_vel = np.loadtxt('/mnt/F4E66F4EE66F0FE2/dropbox_data/energy_velocity_pertur')
    #fre_1st =  round(energy_vel[np.where(energy_vel[np.where(energy_vel[:,0]>0),1]==np.max(energy_vel[np.where(energy_vel[:,0]>0),1])),0],2)
    #fre_1st=1.05
    #mode0path = []
#    mode1path = []
#    mode2path = []
    #mode0path.append('/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_mode0_Re'+str(Re)+'_Omega'+str(round(fre_1st, 2)))
    #mode1path.append('/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_mode1_Re'+str(Re)+'_Omega'+str(round(fre_2st, 2)))
    #mode2path.append('/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_extend/cylinder_mode2_Re'+str(Re)+'_Omega'+str(round(fre_1st, 2)))
    mode0_r=element.add_functions()
    mode0_i=element.add_functions()
    moden0_r=element.add_functions()
    moden0_i=element.add_functions()
    
    mode1_r=element.add_functions()
    mode1_i=element.add_functions()
    moden1_r=element.add_functions()
    moden1_i=element.add_functions()
    
    mode2_r=element.add_functions()
    mode2_i=element.add_functions()
    
    mode3_r=element.add_functions()
    mode3_i=element.add_functions()
    
    mode4_r=element.add_functions()
    mode4_i=element.add_functions()
    
    modef0_r=element.add_functions()
    modef0_i=element.add_functions()
    
    modef1_r=element.add_functions()
    modef1_i=element.add_functions()
    modefn1_r=element.add_functions()
    modefn1_i=element.add_functions()
    
    modef2_r=element.add_functions()
    modef2_i=element.add_functions()
    
    modef3_r=element.add_functions()
    modef3_i=element.add_functions()
    
    modef4_r=element.add_functions()
    modef4_i=element.add_functions()
#    timeseries_mode = TimeSeries(mode0path[0])
#    timeseries_mode.retrieve(mode0_r.vector(), 2.0)
#    timeseries_mode.retrieve(mode0_i.vector(), 3.0)
    #timeseries_mode = TimeSeries(mode1path[0])
    #timeseries_mode.retrieve(mode1_r.vector(), 2.0)
    #timeseries_mode.retrieve(mode1_i.vector(), 3.0)
    #timeseries_mode = TimeSeries(mode2path[0])
    #timeseries_mode.retrieve(mode2_r.vector(), 2.0)
    #timeseries_mode.retrieve(mode2_i.vector(), 3.0)
    ## first harmonic 
    mode0_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_1st))
    mode0_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_1st))
    moden0_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_n1st))
    moden0_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_n1st))
    
    mode1_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_2nd))
    mode1_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_2nd))
    moden1_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_n2nd))
    moden1_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_n2nd))
    
    mode2_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_3rd))
    mode2_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_3rd))
    
    mode3_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_4th))
    mode3_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_4th))
    
    mode4_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_5th))
    mode4_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_5th))

    modef0_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_1st))
    modef0_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_1st))
    
    modef1_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_2nd))
    modef1_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_2nd)) 
    modefn1_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_n2nd))
    modefn1_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_n2nd)) 
    
    modef2_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_3rd))
    modef2_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_3rd))   

    modef3_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_4th))
    modef3_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_4th))  
    
    modef4_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_5th))
    modef4_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_5th))  
    nonlinear_r=[]
    nonlinear_r.append(element.add_functions())
    assign(nonlinear_r[0].sub(0),project(    
                                        +0.0*(dot(mode0_r.sub(0), nabla_grad(mode0_r.sub(0)))-dot(mode0_i.sub(0), nabla_grad(mode0_i.sub(0))))
                                        +0.0*(dot(moden0_r.sub(0), nabla_grad(mode2_r.sub(0)))-dot(moden0_i.sub(0), nabla_grad(mode2_i.sub(0))))
                                        +0.0*(dot(mode2_r.sub(0), nabla_grad(moden0_r.sub(0)))-dot(mode2_i.sub(0), nabla_grad(moden0_i.sub(0))))
                                        +0.0*(dot(moden1_r.sub(0), nabla_grad(mode3_r.sub(0)))-dot(moden1_i.sub(0), nabla_grad(mode3_i.sub(0))))
                                        +0.0*(dot(mode3_r.sub(0), nabla_grad(moden1_r.sub(0)))-dot(mode3_i.sub(0), nabla_grad(moden1_i.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
    #
    nonlinear_i=[]
    nonlinear_i.append(element.add_functions())
    assign(nonlinear_i[0].sub(0),project(
                                        +0.0*(dot(mode0_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_r.sub(0))))
                                        +0.0*(dot(moden0_r.sub(0), nabla_grad(mode2_i.sub(0)))+dot(moden0_i.sub(0), nabla_grad(mode2_r.sub(0))))
                                        +0.0*(dot(mode2_r.sub(0), nabla_grad(moden0_i.sub(0)))+dot(mode2_i.sub(0), nabla_grad(moden0_r.sub(0))))
                                        +0.0*(dot(moden1_r.sub(0), nabla_grad(mode3_i.sub(0)))+dot(moden1_i.sub(0), nabla_grad(mode3_r.sub(0))))
                                        +0.0*(dot(mode3_r.sub(0), nabla_grad(moden1_i.sub(0)))+dot(mode3_i.sub(0), nabla_grad(moden1_r.sub(0)))),element_split.functionspace_V,solver_type='gmres'))  
    #
    force_2nd=P.transpose()*(nonlinear_r[0].vector().get_local()+nonlinear_i[0].vector().get_local()*1j)
    scale0=np.asscalar(np.matrix(force_2nd)*Q*np.matrix(P.transpose()*perturforce_fre_3rd).H)/(np.real(np.asscalar(np.matrix(P.transpose()*perturforce_fre_3rd)*Q*np.matrix(P.transpose()*perturforce_fre_3rd).H)))
    #singularmodes.omega=2.09
    singularmodes.update_problem()
    response_2nd_r=element.add_functions()
    response_2nd_i=element.add_functions()
    response=singularmodes.L_lu(singularmodes.P.dot(singularmodes.M.dot(P.transpose()*perturforce_fre_3rd)))
    response_2nd_r.vector()[:]=np.ascontiguousarray(np.real(response))
    response_2nd_i.vector()[:]=np.ascontiguousarray(np.imag(response))
    response_2nd=response/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*response)*Q*np.matrix(P.transpose()*response).H)))
    scale1=np.asscalar(np.matrix(P.transpose()*response_2nd)*Q*np.matrix(P.transpose()*pertur_fre_3rd).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre_3rd)*Q*np.matrix(P.transpose()*pertur_fre_3rd).H)))
    ## second harmonic
#    nonlinear_r.append(element.add_functions())
#    assign(nonlinear_r[1].sub(0),project(dot(mode0_r.sub(0), nabla_grad(response_2nd_r.sub(0)))+dot(response_2nd_r.sub(0), nabla_grad(mode0_r.sub(0)))-dot(mode0_i.sub(0), nabla_grad(response_2nd_i.sub(0)))-dot(response_2nd_i.sub(0), nabla_grad(mode0_i.sub(0))),element_split.functionspace_V,solver_type='gmres'))
#    nonlinear_i.append(element.add_functions())
#    assign(nonlinear_i[1].sub(0),project(dot(mode0_i.sub(0), nabla_grad(response_2nd_r.sub(0)))+dot(response_2nd_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode0_r.sub(0), nabla_grad(response_2nd_i.sub(0)))+dot(response_2nd_i.sub(0), nabla_grad(mode0_r.sub(0))),element_split.functionspace_V,solver_type='gmres'))
#    force_3rd=P.transpose()*(nonlinear_r[1].vector().get_local()+nonlinear_i[1].vector().get_local()*1j)
#    singularmodes.omega=3.14
#    singularmodes.update_problem()
#    response_3rd_r=element.add_functions()
#    response_3rd_i=element.add_functions()
#    response=singularmodes.L_lu(singularmodes.P.dot(singularmodes.M.dot(force_3rd)))
#    response_3rd_r.vector()[:]=np.ascontiguousarray(np.real(response))
#    response_3rd_i.vector()[:]=np.ascontiguousarray(np.imag(response))
#    response_3rd=response/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*response)*Q*np.matrix(P.transpose()*response).H)))
#    scale2=np.asscalar(np.matrix(P.transpose()*pertur_fre_3rd)*Q*np.matrix(P.transpose()*response_3rd).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre_3rd)*Q*np.matrix(P.transpose()*pertur_fre_3rd).H)))

    #savepath='/mnt/F4E66F4EE66F0FE2/Dropbox/PhD Documents/2018/3rd week 3.14-3.23/report/rebuild_modefzero_r_v.png'
    #fx=project(-nonlinear_i[0][0])      
    #Vdofs_x = fx.function_space().tabulate_dof_coordinates().reshape((-1, 2))
    #vmaxmin=np.max(np.abs(np.imag(fx.vector().get_local())))
    #contourf_cylinder(Vdofs_x[:,0],Vdofs_x[:,1],fx.vector(),xlim=(-2,23),ylim=(-5,5),vminf=-vmaxmin,vmaxf=vmaxmin,colormap='seismic',colorbar='off',axis='off',figsize=(10, 4))

##
if block==6 or block==0:
    # store boundary locations and conditions in two dicts
    BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                       'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                       'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                       'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                       'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                       }
    BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top'},
                        'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom'},
                        'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet'},
                        'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'},
                        }
    
    # define boundaries and mark them with integers
    boundary=Boundary(mesh)
    for key in BoundaryLocations.keys():
         boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
    # Reynolds number
    nu = Constant(1.0 / Re)
    # initialise solver
    singularmodes1=SingularMode(mesh=mesh, boundary=boundary, nu=nu, path=meanpath,omega=Constant(1.046))
    # set boundary conditions
    for key in BoundaryConditions.keys():
        singularmodes1.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
    
        # initialise solver
    singularmodes2=SingularMode(mesh=mesh, boundary=boundary, nu=nu, path=meanpath,omega=Constant(2.093))
    # set boundary conditions
    for key in BoundaryConditions.keys():
        singularmodes2.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
        # initialise solver
    singularmodes3=SingularMode(mesh=mesh, boundary=boundary, nu=nu, path=meanpath,omega=Constant(3.139))
    # set boundary conditions
    for key in BoundaryConditions.keys():
        singularmodes3.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
    
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/FFT_samplingtime_696.5/fft_perturbation_field.npy')
        
    freqs = np.fft.fftfreq(len(time), time_step)*2*pi  
    pertur_fre_1st=pertur_fre[np.where(np.round(freqs,3)==1.046)[0][0],:]*1.0
    
    pertur_fre_2nd=pertur_fre[np.where(np.round(freqs,3)==2.093)[0][0],:]*1.0
    
    pertur_fre_3rd=pertur_fre[np.where(np.round(freqs,3)==3.139)[0][0],:]*1.0
    del pertur_fre
    gc.collect() 
    
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/FFT_samplingtime_696.5/fft_perturbations.npy')
    perturforce_fre_1st=pertur_fre[np.where(np.round(freqs,3)==1.046)[0][0],:]*1.0

    del pertur_fre
    gc.collect()      
    mode_r=element.add_functions()
    mode_i=element.add_functions() 
    
    mode0_r=element.add_functions()
    mode0_i=element.add_functions()
    
    mode1_r=element.add_functions()
    mode1_i=element.add_functions()
    
    mode2_r=element.add_functions()
    mode2_i=element.add_functions()
    
    modef0_r=element.add_functions()
    modef0_i=element.add_functions()
    
    modeff0_r=element.add_functions()
    modeff0_i=element.add_functions()
    
    modef1_r=element.add_functions()
    modef1_i=element.add_functions()
    
    modef2_r=element.add_functions()
    modef2_i=element.add_functions()
    
    mode0_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_1st))
    mode0_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_1st))
#    
    mode1_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_2nd))
    mode1_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_2nd))
#    
    mode2_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_3rd))
    mode2_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_3rd))
    
    modeff0_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_1st))
    modeff0_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_1st))
    ##1.038
#    mode0path ='/media/bojin/ResearchData/dropbox_data/FrequencyAnlysis/Cylinder_SVD_Re100/Cylinder_SVD_Re100_round3/cylinder_mode0_Re'+str(Re)+'_Omega'+str(round(1.046, 3))
#    timeseries_mode = TimeSeries(mode0path)
#    timeseries_mode.retrieve(mode_r.vector(), 2.0)
#    timeseries_mode.retrieve(mode_i.vector(), 3.0)
#    mode=-(mode_r.vector().get_local()+mode_i.vector().get_local()*1j)
#    scale0=np.asscalar(np.matrix(P.transpose()*mode)*Q*np.matrix(P.transpose()*pertur_fre_1st).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*mode)*Q*np.matrix(P.transpose()*mode).H))*np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre_1st)*Q*np.matrix(P.transpose()*pertur_fre_1st).H)))
    
#    mode0_r.vector()[:]=np.ascontiguousarray(np.real(mode))
#    mode0_i.vector()[:]=np.ascontiguousarray(np.imag(mode))
    singularmodes1.update_problem()  
    singularmodes2.update_problem()  
    singularmodes3.update_problem() 
    scale0=[]
    scale1=[]
    scale2=[]
    scale2f=[]
    for i in range(5):
        assign(modef1_r.sub(0),project( 
                                    1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_r.sub(0)))-dot(mode0_i.sub(0), nabla_grad(mode0_i.sub(0))))
                                    +1.0*(dot(mode0_r.sub(0), nabla_grad(mode2_r.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode2_i.sub(0))))
                                    +1.0*(dot(mode2_r.sub(0), nabla_grad(mode0_r.sub(0)))+dot(mode2_i.sub(0), nabla_grad(mode0_i.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
        assign(modef1_i.sub(0),project(
                                    1.0*(dot(mode0_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode0_r.sub(0))))
                                    +1.0*(dot(mode0_r.sub(0), nabla_grad(mode2_i.sub(0)))-dot(mode0_i.sub(0), nabla_grad(mode2_r.sub(0))))
                                    +1.0*(-dot(mode2_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode2_i.sub(0), nabla_grad(mode0_r.sub(0)))),element_split.functionspace_V,solver_type='gmres'))  
           
        response_2nd=-singularmodes2.L_lu(singularmodes2.P.dot(singularmodes2.M.dot(P.transpose()*(modef1_r.vector().get_local()+modef1_i.vector().get_local()*1j))))
        mode1_r.vector()[:]=np.ascontiguousarray(np.real(response_2nd))
        mode1_i.vector()[:]=np.ascontiguousarray(np.imag(response_2nd))
        scale0.append(np.asscalar(np.matrix(P.transpose()*response_2nd)*Q*np.matrix(P.transpose()*pertur_fre_2nd).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*response_2nd)*Q*np.matrix(P.transpose()*response_2nd).H))*np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre_2nd)*Q*np.matrix(P.transpose()*pertur_fre_2nd).H))))

        
        assign(modef2_r.sub(0),project(
                                    1.0*(dot(mode0_r.sub(0), nabla_grad(mode1_r.sub(0)))-dot(mode0_i.sub(0), nabla_grad(mode1_i.sub(0))))
                                    +1.0*(dot(mode1_r.sub(0), nabla_grad(mode0_r.sub(0)))-dot(mode1_i.sub(0), nabla_grad(mode0_i.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
     
        assign(modef2_i.sub(0),project( 
                                    1.0*(dot(mode0_r.sub(0), nabla_grad(mode1_i.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode1_r.sub(0))))
                                    +1.0*(dot(mode1_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode1_i.sub(0), nabla_grad(mode0_r.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
        
        response_3rd=-singularmodes3.L_lu(singularmodes3.P.dot(singularmodes3.M.dot(P.transpose()*(modef2_r.vector().get_local()+modef2_i.vector().get_local()*1j))))
        mode2_r.vector()[:]=np.ascontiguousarray(np.real(response_3rd))
        mode2_i.vector()[:]=np.ascontiguousarray(np.imag(response_3rd)) 
        scale1.append(np.asscalar(np.matrix(P.transpose()*response_3rd)*Q*np.matrix(P.transpose()*pertur_fre_3rd).H)/np.sqrt(np.real(np.asscalar(np.matrix(P.transpose()*response_3rd)*Q*np.matrix(P.transpose()*response_3rd).H))*np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre_3rd)*Q*np.matrix(P.transpose()*pertur_fre_3rd).H))))
                              
        assign(modef0_r.sub(0),project(
                                    1.0*(dot(mode0_r.sub(0), nabla_grad(mode1_r.sub(0)))+dot(mode0_i.sub(0), nabla_grad(mode1_i.sub(0))))
                                    +1.0*(dot(mode1_r.sub(0), nabla_grad(mode0_r.sub(0)))+dot(mode1_i.sub(0), nabla_grad(mode0_i.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
     
        assign(modef0_i.sub(0),project( 
                                    1.0*(dot(mode0_r.sub(0), nabla_grad(mode1_i.sub(0)))-dot(mode0_i.sub(0), nabla_grad(mode1_r.sub(0))))
                                    +1.0*(-dot(mode1_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(mode1_i.sub(0), nabla_grad(mode0_r.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
    
        ff=modef0_r.vector().get_local()+modef0_i.vector().get_local()*1j
        scale2f.append(np.asscalar(np.matrix(P.transpose()*ff)*Q*np.matrix(P.transpose()*perturforce_fre_1st).H)/(np.real(np.asscalar(np.matrix(P.transpose()*perturforce_fre_1st)*Q*np.matrix(P.transpose()*perturforce_fre_1st).H))))
        
        response_1st=-singularmodes1.L_lu(singularmodes1.P.dot(singularmodes1.M.dot(P.transpose()*(modef0_r.vector().get_local()+modef0_i.vector().get_local()*1j))))
        mode0_r.vector()[:]=np.ascontiguousarray(np.real(response_1st))
        mode0_i.vector()[:]=np.ascontiguousarray(np.imag(response_1st))
        scale2.append(np.asscalar(np.matrix(P.transpose()*response_1st)*Q*np.matrix(P.transpose()*pertur_fre_1st).H)/(np.real(np.asscalar(np.matrix(P.transpose()*pertur_fre_1st)*Q*np.matrix(P.transpose()*pertur_fre_1st).H))))

##
if block==7 or block==0:
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/FFT_samplingtime_696.5/fft_perturbation_field.npy')
        
    freqs = np.fft.fftfreq(len(time), time_step)*2*pi  
    pertur_fre_1st=pertur_fre[np.where(np.round(freqs,3)==1.046)[0][0],:]*1.0
    
    pertur_fre_2nd=pertur_fre[np.where(np.round(freqs,3)==2.093)[0][0],:]*1.0
    
    del pertur_fre
    gc.collect() 
    
    pertur_fre = np.load('/mnt/F4E66F4EE66F0FE2/dropbox_data/FFT_samplingtime_696.5/fft_perturbations.npy')
    perturforce_fre_1st=pertur_fre[np.where(np.round(freqs,3)==1.046)[0][0],:]*1.0

    del pertur_fre
    gc.collect()
    
    mode0_r=element.add_functions()
    mode0_i=element.add_functions()
    
    mode1_r=element.add_functions()
    mode1_i=element.add_functions()
    
    modef0_r=element.add_functions()
    modef0_i=element.add_functions()
    
    mode0_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_1st))
    mode0_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_1st))
#    
    mode1_r.vector()[:]=np.ascontiguousarray(np.real(pertur_fre_2nd))
    mode1_i.vector()[:]=np.ascontiguousarray(np.imag(pertur_fre_2nd))
#    
    
    modef0_r.vector()[:]=np.ascontiguousarray(np.real(perturforce_fre_1st))
    modef0_i.vector()[:]=np.ascontiguousarray(np.imag(perturforce_fre_1st))
    
    f_r=(inner(dot(mode0_r.sub(0), nabla_grad(element.tu)),element.v)+inner(dot(element.tu, nabla_grad(mode0_r.sub(0))),element.v))*dx
    f_i=(-inner(dot(mode0_i.sub(0), nabla_grad(element.tu)),element.v)-inner(dot(element.tu, nabla_grad(mode0_i.sub(0))),element.v))*dx
    matass = MatrixAssemble()    
    
        # store boundary locations and conditions in two dicts
    BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                       'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                       'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                       'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                       'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                       }
    BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'top'},
                        'Bottom'  : {'FunctionSpace': 'V.sub(0).sub(1)',   'Value': Constant(0.0),       'Boundary': 'bottom'},
                        'Inlet'   : {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'inlet'},
                        'Cylinder': {'FunctionSpace': 'V.sub(0)',          'Value': Constant((0.0,0.0)), 'Boundary': 'cylinder'},
                        }
    
    # define boundaries and mark them with integers
    boundary=Boundary(mesh)
    for key in BoundaryLocations.keys():
         boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
    # Reynolds number
    nu = Constant(1.0 / Re)
    # initialise solver
    singularmodes1=SingularMode(mesh=mesh, boundary=boundary, nu=nu, path=meanpath,omega=Constant(1.046))
    singularmodes2=SingularMode(mesh=mesh, boundary=boundary, nu=nu, path=meanpath,omega=Constant(1.046))
    for key in BoundaryConditions.keys():
        singularmodes1.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
  
    F_r=matass.assemblematrix(f_r,singularmodes1.bcs)
    F_i=matass.assemblematrix(f_i,singularmodes1.bcs)
    F=F_r+F_i.multiply(1j)
#    singularmodes1.update_problem()
    singularmodes1.update_problem()
    M=singularmodes1.M
    A=P.transpose() *(F) * P
    b=M.dot(P.transpose().dot(singularmodes1.L*(mode0_r.vector().get_local()+mode0_i.vector().get_local()*1j)))
    u2=spla.spsolve(A, b)
    modeu2_r=element.add_functions()
    modeu2_i=element.add_functions()
    modeu2_r.vector()[:]=np.ascontiguousarray(np.real(P*u2))
    modeu2_i.vector()[:]=np.ascontiguousarray(np.imag(P*u2))

    assign(modef0_r.sub(0),project(
                                1.0*(dot(mode0_r.sub(0), nabla_grad(modeu2_r.sub(0)))+dot(mode0_i.sub(0), nabla_grad(modeu2_i.sub(0))))
                                +1.0*(dot(modeu2_r.sub(0), nabla_grad(mode0_r.sub(0)))+dot(modeu2_i.sub(0), nabla_grad(mode0_i.sub(0)))),element_split.functionspace_V,solver_type='gmres'))
 
    assign(modef0_i.sub(0),project( 
                                1.0*(dot(mode0_r.sub(0), nabla_grad(modeu2_i.sub(0)))-dot(mode0_i.sub(0), nabla_grad(modeu2_r.sub(0))))
                                +1.0*(-dot(modeu2_r.sub(0), nabla_grad(mode0_i.sub(0)))+dot(modeu2_i.sub(0), nabla_grad(mode0_r.sub(0)))),element_split.functionspace_V,solver_type='gmres'))

    ff=modef0_r.vector().get_local()+modef0_i.vector().get_local()*1j  
    response_1st=-singularmodes1.L_lu(singularmodes1.P.dot(singularmodes1.M.dot(P.transpose()*(modef0_r.vector().get_local()+modef0_i.vector().get_local()*1j))))

                    