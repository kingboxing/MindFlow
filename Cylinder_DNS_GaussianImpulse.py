"""
Give system state after impulse Gaussian force which is implemented changing as Gaussian procedure

Problem type : unsteady Navier-Stokes equations

Details : unsteady cylinder flow at Re = 80
    five boundaries : top and bottom symmetric wall,
    no-slip cylinder surface (d = 1), velocity inlet (1, 0), outlet with -p*n+nu*grad(u)*n=0 and p = 0

Method : IPCS method (split method)
"""
from __future__ import print_function
from fenics import *
import time as tm
import numpy as np
from Boundary.Set_Boundary import Boundary
from FlowSolver.DNS_Solver_MPI import *

def dirac_function(a,c,x):
    dirac=1.0/(a*sqrt(pi))*np.exp(-(x-c)**2/a**2)
    return dirac

def gaussianforce(radius, angle, sig):
    center_x = radius*cos(angle/180.0*pi)
    center_y = radius*sin(angle/180.0*pi)
    sigma2 = sig**2
    force = Expression(
            'pow(2.0*sigma2*pi,-1)*exp(-(pow(x[0]-center_x,2)+pow(x[1]-center_y,2))*pow(2.0*sigma2,-1))',
            degree=2, sigma2=sigma2, center_x=center_x, center_y=center_y)
    return force

radius = 0.6
angle = 70
sig = 0.1
force_upper = gaussianforce(radius, angle, sig)
force_lower = gaussianforce(radius, -angle, sig)
x_component = Constant(cos(angle/180.0*pi))*(force_upper-force_lower)
y_component = Constant(sin(angle/180.0*pi))*(force_upper+force_lower)
force = (x_component, y_component)

dt=1e-6
t_all=1e-3
mag=1e-2
tt=np.round(np.arange(0,t_all+dt,dt),6)
signal=[0.0]+list(mag*dirac_function(2e-4,t_all/2,tt))+[0.0]
np.trapz(signal)*dt
#%%

mesh=Mesh("./mesh/cylinder_26thousand.xml")
element_base = TaylorHood(mesh=mesh)
element_init = TaylorHood(mesh=mesh)
# store boundary locations and conditions in two dicts
BoundaryLocations = {'Top'      : {'Mark': 1, 'Location':'on_boundary and near(x[1], 15.0, tol)'},
                   'Bottom'     : {'Mark': 2, 'Location':'on_boundary and near(x[1], -15.0, tol)'},
                   'Inlet'      : {'Mark': 3, 'Location':'on_boundary and x[0] < 0.0 + tol and not (between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5)))'},
                   'Outlet'     : {'Mark': 4, 'Location':'on_boundary and near(x[0], 23.0, tol)'},
                   'Cylinder'   : {'Mark': 5, 'Location':'on_boundary and between(x[0], (-0.5, 0.5)) and between(x[1], (-0.5, 0.5))'},
                   }
BoundaryConditions = {'Top'   : {'FunctionSpace': 'V.sub(1)',   'Value': Constant(0.0)},
                    'Bottom'  : {'FunctionSpace': 'V.sub(1)',   'Value': Constant(0.0)},
                    'Inlet'   : {'FunctionSpace': 'V',          'Value': Constant((1.0,0.0))},
                    'Cylinder': {'FunctionSpace': 'V',          'Value': Constant((0.0,0.0))}
                    }
BoundaryConditions_p = {
                    'Outlet'  : {'Value': 'FreeOutlet'}
                    }
# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
    boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
# Reynolds number
Re = 60
nu = Constant(1.0 / Re)

path = "./base flow/cylinder_baseflow_newton_26thousand"+str(Re).zfill(3)
timeseries_base = TimeSeries(path)
timeseries_base.retrieve(element_base.w.vector(), 0.0)

# initialise solver
solver = DNS_IPCS_Solver(mesh=mesh, boundary=boundary, nu=nu, dt=dt)
assign(solver.u_pre, element_base.w.sub(0))
assign(solver.p_pre, element_base.w.sub(1))

savepath='./control_gaussianvel_impulse1E-2_cylinder_dt_1E-6_IPCS_26thousand'+str(Re).zfill(3)
timeseries_result = TimeSeries(savepath)

# set boundary conditions
for key in BoundaryConditions.keys():
    solver.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
for key in BoundaryConditions_p.keys():
    solver.set_boundarycondition(BoundaryConditions_p[key], BoundaryLocations[key]['Mark'])

if comm_rank==0:
    start_time = tm.time()
    myfile = open('./control_gaussianvel_impulse1E-2_cylinder_dt_1E-6_IPCS_26thousand'+str(Re).zfill(3)+'_dragandlift', 'a')
for i in range(0,np.size(signal)):
    time_i=i*dt
    solver.sourceterm = (Constant(signal[i])*force[0], Constant(signal[i])*force[1])
    solver.solve(method='direct',iterative_num=40, iterative_tol=1e-6)
    
    lift=(solver.get_force(bodymark=5, direction=1))
    drag=(solver.get_force(bodymark=5, direction=0))
    if comm_rank==0:
        myfile.write("%e\t %e\t %e\t %e\n" % (time_i, drag, lift, solver.u_pre(2.75,0)[1]))
        myfile.flush()
    if comm_rank==0:
        print('time= %e(i= %d)    lift= %e    drag= %e' % (time_i, i, 2.0 * lift, 2.0 * drag))  
        
assign(element_init.w.sub(0), solver.u_pre)
assign(element_init.w.sub(1), solver.p_pre)
timeseries_result.store(element_init.w.vector(),0.0)
if comm_rank==0:
    myfile.close()
    end_time=tm.time()
    elapsed_time = end_time - start_time
    print('time %e' % (elapsed_time))
    
