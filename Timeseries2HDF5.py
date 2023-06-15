from __future__ import print_function
from fenics import *
import numpy as np
import os

mesh=Mesh("./RAPACK/mesh/cylinder_highRe_263thousand.xml")
path = os.getcwd()+"/DNSdata_Re2000_refined/cylinder_DNS_IPCS_263thousand_Re"+str(2000).zfill(4)
hdf = HDF5File(mesh.mpi_comm(), 'cylinder_DNS_IPCS_263thousand_Re'+str(2000).zfill(4)+'.h5','w')

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
u = Function(V)
p = Function(Q)

paths0 = {'Velocity':    path+"_velocity",
         'Pressure':    path+"_pressure"}
paths1 = {'Velocity':    path+"_velocity_part1",
         'Pressure':    path+"_pressure_part1"}
paths2 = {'Velocity':    path+"_velocity_part2",
         'Pressure':    path+"_pressure_part2"}
paths3 = {'Velocity':    path+"_velocity_part3",
         'Pressure':    path+"_pressure_part3"}
paths4 = {'Velocity':    path+"_velocity_part4",
         'Pressure':    path+"_pressure_part4"}
paths5 = {'Velocity':    path+"_velocity_part5",
         'Pressure':    path+"_pressure_part5"}
paths6 = {'Velocity':    path+"_velocity_part6",
         'Pressure':    path+"_pressure_part6"}
#i=0
#timeseries_v = TimeSeries(mesh.mpi_comm(), eval("paths"+str(i)+"['Velocity']"))
#timeseries_p = TimeSeries(mesh.mpi_comm(), eval("paths"+str(i)+"['Pressure']"))
#time = timeseries_v.vector_times()
#timeseries_v.retrieve(u.vector(), 0.0)
#timeseries_p.retrieve(p.vector(), 0.0)
#hdf.write(u, 'Velocity', 0.0)
#hdf.write(p, 'Pressure', 0.0)
         
times = []
for i in range(0,7):
    timeseries_v = TimeSeries(mesh.mpi_comm(), eval("paths"+str(i)+"['Velocity']"))
    timeseries_p = TimeSeries(mesh.mpi_comm(), eval("paths"+str(i)+"['Pressure']"))
    time = timeseries_v.vector_times()
    for t in time[0:-1]:
        timeseries_v.retrieve(u.vector(), t)
        timeseries_p.retrieve(p.vector(), t)
        hdf.write(u, 'Velocity', t)
        hdf.write(p, 'Pressure', t)
        times.append(t)
        print(t)
timeseries_v.retrieve(u.vector(), time[-1])
timeseries_p.retrieve(p.vector(), time[-1])
hdf.write(u, 'Velocity', time[-1])
hdf.write(p, 'Pressure', time[-1])
times.append(time[-1])
hdf.write(np.array(times), 'Time')
