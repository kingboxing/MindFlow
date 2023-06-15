"""
Problem type : steady Navier-Stokes equations (base flow)

Details : steady cylinder flow at Re = 45 and Re = 100
    five boundaries : top and bottom symmetric wall,
    no-slip cylinder surface (d = 1), velocity inlet (1, 0), outlet with -p*n+nu*grad(u)*n=0

Method : Newton method
"""
from __future__ import print_function
from fenics import *
import numpy as np
import time
from Boundary.Set_Boundary import Boundary
from FlowSolver.NS_Newton_Solver import *
from FlowSolver.FiniteElement import TaylorHood

mesh=Mesh("./mesh/cylinder_26thousand.xml")
element=TaylorHood(mesh=mesh,order=(2,1))
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
#Expression(('-10*x[1]', '10*x[0]'),degree=2)

# define boundaries and mark them with integers
boundary=Boundary(mesh)
for key in BoundaryLocations.keys():
     boundary.set_boundary(location=BoundaryLocations[key]['Location'], mark=BoundaryLocations[key]['Mark'])
# initialise solver
solver = NS_Newton_Solver(mesh=mesh, boundary=boundary,order=(2,1))
# set solver parameters
solver.solver_parameters(parameters={'newton_solver':{'linear_solver': 'mumps','absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12}})
# set boundary conditions
for key in BoundaryConditions.keys():
    solver.set_boundarycondition(BoundaryConditions[key], BoundaryLocations[key]['Mark'])
#%%
# Reynolds number
#Re = list(np.arange(100,300,50))+list(np.arange(275,325,25))#list(np.arange(275,350,25))+list(np.arange(330,390,10))+list(np.arange(385,505,5))#[100]#
Re=[60]#range(100,500,25)+range(500,1050,50)#range(100,100050,25)    
for rr in Re:
    nu = Constant(1.0 / rr)
    # solve the problem
    solver.solve(nu=nu)
    # print drag and lift
    print('Re= %d     drag= %e    lift= %e' % (rr, solver.get_force(bodymark=5,direction=0) , solver.get_force(bodymark=5,direction=1)))
    assign(element.w,solver.w)
# # change parameters and Reynolds number, solve the problem again
# solver.solver_parameters(parameters={'newton_solver':{'linear_solver': 'mumps','absolute_tolerance': 1e-8, 'relative_tolerance': 1e-8}})
# solver.solve(nu=Constant(1.0/45))
# # print drag and lift
# print('drag= %e    lift= %e' % (solver.get_force(bodymark=5,direction=0) , solver.get_force(bodymark=5,direction=1)))
#
    # save the base flow when Re = 100
    if rr>500:
        savepath='./base flow/cylinder_baseflow_newton_26thousand_highorder_'+str(rr).zfill(3) #'./base flow/cylinder_baseflow_newton_150thousand_Re4000'#+str(int(1/float(solver.nu))).zfill(3)
        timeseries_r = TimeSeries(savepath)
        timeseries_r.store(solver.w.vector(), 0.0)
