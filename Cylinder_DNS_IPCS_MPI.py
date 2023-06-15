"""
Problem type : unsteady Navier-Stokes equations

Details : unsteady cylinder flow at Re = 2000
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
from FlowSolver.DataConvert import *

mesh=Mesh("./mesh/cylinder_26thousand.xml")

## data convert, combine result
#Re=180
#path1='/mnt/F4E66F4EE66F0FE2/dropbox_data/cylinder_dt_001_DNS_IPCS_26thousand_Re'+str(Re).zfill(3)
#element_1 = SplitPV(mesh=mesh)
#element_2 = TaylorHood(mesh=mesh)
#timeseries_v1 = TimeSeries(path1+'_velocity')
#timeseries_p1 = TimeSeries(path1+'_pressure')
#timeseries_2 = TimeSeries(path1)
#time_series=np.arange(250.0,400.1,0.1)
#for time in time_series:
#    info('time: %g' %time)
#    timeseries_v1.retrieve(element_1.u.vector(), time)
#    timeseries_p1.retrieve(element_1.p.vector(), time)
#    assign(element_2.w.sub(0), element_1.u)
#    assign(element_2.w.sub(1), element_1.p)
#    timeseries_2.store(element_2.w.vector(), time)

# DNS
element_base = TaylorHood(mesh=mesh)
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
#BoundaryConditions_p = {
#                    'Outlet'  : {'FunctionSpace': 'Q',          'Value': Constant(0.0)}
#                    }
BoundaryConditions_p = {
                    'Outlet'  : {'Value': 'FreeOutlet'}
                    }               
                    
# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
    boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
# Reynolds number
Re = 100
nu = Constant(1.0 / Re)
dt = 0.01
# initial conditions
path = '/mnt/F4E66F4EE66F0FE2/dropbox_data/cylinder_dt_001_DNS_IPCS_26thousand_Re'+str(Re).zfill(3)
#path='./base flow/cylinder_baseflow_newton_26thousand100'
timeseries_r = TimeSeries(path)
timeseries_r.retrieve(element_base.w.vector(),400.0)
# initialise solver
solver = DNS_IPCS_Solver(mesh=mesh, boundary=boundary, nu=nu, dt=dt, path=None, noise=False,restart=0.0)
assign(solver.u_pre, element_base.w.sub(0))
assign(solver.p_pre, element_base.w.sub(1))

# save path, use HDF5File while parallelism computing
savepath='/mnt/F4E66F4EE66F0FE2/dropbox_data/cylinder_dt_001_DNS_IPCS_26thousand_Re'+str(Re).zfill(3)
timeseries_vel = TimeSeries(savepath+'_velocity')
timeseries_pre = TimeSeries(savepath+'_pressure')

# set boundary conditions
for key in BoundaryConditions.keys():
    solver.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
for key in BoundaryConditions_p.keys():
    solver.set_boundarycondition(BoundaryConditions_p[key], BoundaryLocations[key]['Mark'])
# record time and set file to store lift&drag
if comm_rank==0:
    start_time = tm.time()
    myfile = open('/mnt/F4E66F4EE66F0FE2/dropbox_data/cylinder_dt_001_DNS_IPCS_26thousand_Re'+str(Re).zfill(3)+'_dragandlift', 'a')
# time: 0-200s
for i in range(0,11):
    time_i=i*dt
    lift=(solver.get_force(bodymark=5, direction=1))
    drag=(solver.get_force(bodymark=5, direction=0))
    if comm_rank==0:
        myfile.write("%e\t %e\t %e\n" % (time_i, drag, lift))
        myfile.flush()
    if i%10 == 0:
        timeseries_vel.store(solver.u_pre.vector(), time_i)
        timeseries_pre.store(solver.p_pre.vector(), time_i)
    if comm_rank==0:
        print('time= %e(i= %d)    lift= %e    drag= %e' % (time_i, i, 2.0 * lift, 2.0 * drag))
    solver.solve(method='direct',iterative_num=5, iterative_tol=1e-6)
if comm_rank==0:
    myfile.close()
    end_time=tm.time()
    elapsed_time = end_time - start_time
    print('time %e' % (elapsed_time))
