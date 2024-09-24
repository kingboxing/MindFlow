#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:24:05 2024

@author: bojin
"""

from ..Deps import *

class Params:
    """
    
    """
    def __init__(self):
        self.params={}
        
    def params_default(self):
        nprm=NonlinearVariationalSolver.default_parameters()
        lprm=LinearVariationalSolver.default_parameters()
        
    def solver_default(self):
        info('-'*78)
        list_linear_solver_methods()
        info('-'*62)
        list_krylov_solver_preconditioners()
        
    def NewtonSolver(self, params={}):
        self.newton_solver = {'linear_solver':            'mumps',
                              'absolute_tolerance':       1e-12, 
                              'relative_tolerance':       1e-12,
                              'convergence_criterion':    'residual',
                              'error_on_nonconvergence':  True,
                              'krylov_solver':            self.krylov_solver,
                              'lu_solver':                self.lu_solver,
                              'maximum_iterations':       50,
                              'preconditioner':           'default',
                              'relaxation_parameter':     None,
                              'report':                   True
                            }
        self.newton_solver.update(params)
        
    def SnesSolver(self,params={}):
        self.snes_solver = {'absolute_tolerance':            1e-10,
                            'error_on_nonconvergence':       True,
                            'krylov_solver':                 self.krylov_solver,
                            'line_search':                   'basic',
                            'linear_solver':                 'default',
                            'lu_solver':                     self.lu_solver,
                            'maximum_iterations':            50,
                            'maximum_residual_evaluations':  2000,
                            'method':                        'default',
                            'preconditioner':                'default',
                            'relative_tolerance':             1e-09,
                            'report':                         True,
                            'sign':                           'default',
                            'solution_tolerance':             1e-16
                            }
        self.snes_solver.update(params)
        
    def KrylovSolver(self, params={}):
        self.krylov_solver={'absolute_tolerance':        1e-10,
                            'divergence_limit':          1e5,
                            'error_on_nonconvergence':   True,
                            'maximum_iterations':        1000,
                            'monitor_convergence':       True,
                            'nonzero_initial_guess':     False,
                            'relative_tolerance':        1e-8,
                            'report':                    False              
                            }
        self.krylov_solver.update(params)
        
    def LUSolver(self,params={}):
        self.lu_solver={'report':    True, 
                        'symmetric': False, 
                        'verbose':   False,
                        'reuse_factorization': True # not sure if it worked even para.add() method is not used
                        }
        self.lu_solver.update(params)
        
    def NonLinearSolver(self,params={}):
        
        self.params.update({'newton_solver':    self.newton_solver,
                            'nonlinear_solver': 'newton',
                            'print_matrix':     False,
                            'print_rhs':        False,
                            'snes_solver':      self.snes_solver,
                            'symmetric':        False
                            })
        self.params.update(params)
        
    def LinearSolver(self,params={}):
        self.params.update({'linear_solver':{
                            'krylov_solver':  self.krylov_solver,
                            'linear_solver':  'default',
                            'lu_solver':      self.lu_solver,
                            'preconditioner': 'default',
                            'print_matrix':    False,
                            'print_rhs':       False,
                            'symmetric':       False
                            }})
        self.params.update(params)
    
    def IPCSolver(self, params={}):
        self.ipcs_solver = {'method':                   'lu', 
                            'linear_solver':            'mumps',
                            'lu_solver':                self.lu_solver,
                            'krylov_solver':            self.krylov_solver,
                            
                            'maximum_iterations':       20,
                            'convergence_criterion':    'linf',
                            'absolute_tolerance':       1e-7,
                            'relative_tolerance':       1e-7,
                            'relaxation_parameter':     1.0,
                            'report':                   False
                            }
        
    
    
    

class FreqOptions:
    """
    """
    
    def __init__(self):
        pass
    

class ContOptions:
    """
    """
    
    def __init__(self):
        pass