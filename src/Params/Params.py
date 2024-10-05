#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:24:05 2024

@author: bojin

A centralized class to manage default parameters for different modules, classes, or functions.
This class allows other parts of the package to access, modify, and reset default values.

"""

from ..Deps import *
from ..LinAlg.Utils import dict_deep_update


class DefaultParameters:
    def __init__(self):
        """
        Initialize the default parameters for various modules, classes, or functions.
        """
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        self._initializer_map = {'eigen_decompose': self._initialize_eigen_decompose_defaults,
                                 'eigen_solver': self._initialize_eigen_solver_defaults,
                                 'frequency_response': self._initialize_freq_defaults,
                                 'resolvent_solver': self._initialize_resolvent_defaults,
                                 'state_space_model': self._initialize_ssmodel_defaults,
                                 'bernoulli_feedback': self._initialize_bernoulli_defaults,
                                 'riccati_pymess': self._initialize_riccati_defaults,
                                 'lqe_pymess': self._initialize_lqe_defaults,
                                 'lqr_pymess': self._initialize_lqr_defaults,
                                 'newton_solver': self._initialize_newton_defaults,
                                 'IPCS_solver': self._initialize_IPCS_defaults,
                                 }

        for key in self._initializer_map:
            self._initializer_map[key]()

    def _initialize_eigen_decompose_defaults(self):
        default = {'method': 'lu',
                   'lusolver': 'mumps',
                   'echo': False,
                   'which': 'LM',
                   'v0': None,
                   'ncv': None,
                   'maxiter': None,
                   'tol': 0,
                   'return_eigenvectors': True,
                   'OPpart': None
                   }
        self.parameters['eigen_decompose'] = default

    def _initialize_eigen_solver_defaults(self):
        """
        symmetry : bool, optional
            whether assemble matrices in the symmetry way. Default is True.
        solve_type : str, optional
            Solver type ('implicit' or 'explicit'). Default is 'implicit'.
        inverse : bool, optional
            Whether to compute the inverse of the RHS eigenvectors. Default is False.
        BCpart : ‘r’ or ‘i’/None, optional
            The homogenous boundary conditions are applied in the real (matrix NS) or imag (matrix M) part of system. The default is None.
        """

        defaults_eigs = copy.deepcopy(self.parameters['eigen_decompose'])
        defaults_diff = {'symmetry': True,
                         'solve_type': 'implicit',
                         'inverse': False,
                         'BCpart': None
                         }
        defaults_eigs.update(defaults_diff)
        defaults = {'solver_type': 'eigen_solver',
                    'eigen_solver': defaults_eigs,
                    }

        self.parameters['eigen_solver'] = defaults

    def _initialize_freq_defaults(self):
        defaults = {'solver_type': 'frequency_response',
                    'frequency_response': {'method': 'lu',
                                           'lusolver': 'mumps',
                                           'echo': False}
                    }
        self.parameters['frequency_response'] = defaults

    def _initialize_resolvent_defaults(self):
        defaults_eigs = copy.deepcopy(self.parameters['eigen_decompose'])
        defaults_diff = {'symmetry': True,
                         'BCpart': None
                         }
        defaults_eigs.update(defaults_diff)
        defaults = {'solver_type': 'resolvent_solver',
                    'resolvent_solver': defaults_eigs,
                    }
        self.parameters['resolvent_solver'] = defaults

    def _initialize_bernoulli_defaults(self):
        defaults_eigs = copy.deepcopy(self.parameters['eigen_decompose'])
        defaults_diff = {'k': 2,
                         'sigma': 0.0,
                         'which': 'LR',  # compute unstable eigenvalues
                         }
        defaults_eigs.update(defaults_diff)
        defaults = {'solver_type': 'bernoulli_feedback',
                    'bernoulli_feedback':  defaults_eigs,
                    }

        self.parameters['bernoulli_feedback'] = defaults

    def _initialize_riccati_defaults(self):
        defaults_pymess = {'solver_type': 'riccati_solver',
                           'riccati_solver': {'type': 0,  #mess.MESS_OP_NONE
                                              'lusolver': 3,  #mess.MESS_DIRECT_UMFPACK_LU
                                              'adi': {'memory_usage': 2,  #mess.MESS_MEMORY_HIGH
                                                      'paratype': 3,  #mess.MESS_LRCFADI_PARA_ADAPTIVE_V
                                                      'output': 0,
                                                      'res2_tol': 1e-8,
                                                      'maxit': 2000
                                                      },
                                              'nm': {'output': 1,
                                                     'singleshifts': 0,
                                                     'linesearch': 1,
                                                     'res2_tol': 1e-5,
                                                     'maxit': 30,
                                                     'k0': None  #Initial Feedback
                                                     }
                                              }
                           }

        self.parameters['riccati_pymess'] = defaults_pymess

    def _initialize_lqe_defaults(self):
        self.parameters['lqe_pymess'] = copy.deepcopy(self.parameters['riccati_pymess'])
        self.parameters['lqe_pymess']['solver_type'] = 'lqe_solver'

    def _initialize_lqr_defaults(self):
        self.parameters['lqr_pymess'] = copy.deepcopy(self.parameters['riccati_pymess'])
        self.parameters['lqr_pymess']['solver_type'] = 'lqr_solver'
        self.parameters['lqr_pymess']['riccati_solver']['type'] = 1  # mess.MESS_OP_TRANSPOSE

    def _initialize_ssmodel_defaults(self):
        defaults = {'solver_type': 'state_space_model',
                    'state_space_model': {}
                    }
        self.parameters['state_space_model'] = defaults

    def _initialize_newton_defaults(self):
        """
        pending for test
        """

        defaults_lu_solver = {'report': True,
                              'symmetric': False,
                              'verbose': False,
                              }
        defaults_krylov_solver = {'absolute_tolerance': 1e-10,
                                  'divergence_limit': 1e5,
                                  'error_on_nonconvergence': True,
                                  'maximum_iterations': 1000,
                                  'monitor_convergence': True,
                                  'nonzero_initial_guess': False,
                                  'relative_tolerance': 1e-8,
                                  'report': False
                                  }
        defaults = {'solver_type': 'newton_solver',
                    'bc_reset': False,
                    'newton_solver': {'linear_solver': 'mumps',
                                      'absolute_tolerance': 1e-12,
                                      'relative_tolerance': 1e-12,
                                      'convergence_criterion': 'residual',
                                      'error_on_nonconvergence': True,
                                      'maximum_iterations': 50,
                                      'preconditioner': 'default',
                                      'relaxation_parameter': None,
                                      'report': True,
                                      'krylov_solver': defaults_krylov_solver,
                                      'lu_solver': defaults_lu_solver,
                                      }
                    }

        self.parameters['newton_solver'] = defaults

    def _initialize_IPCS_defaults(self):
        """
        pending for test
        """
        defaults = {'solver_type': 'IPCS_solver',
                    'IPCS_solver': {},
                    }
        self.parameters['IPCS_solver'] = defaults

    def get_defaults(self, module):
        """
        Get default parameters for a specific module or class.

        Parameters
        ----------
        module : str
            The name of the module or class to retrieve defaults for.

        Returns
        -------
        dict
            The dictionary of default parameters for the specified module or class.
        """
        return self.parameters.get(module, {})

    def update_defaults(self, module, updates):
        """
        Update default parameters for a specific module or class.

        Parameters
        ----------
        module : str
            The name of the module or class to update defaults for.
        updates : dict
            A dictionary containing parameter updates.

        Returns
        -------
        None
        """
        if module in self.parameters:
            self.parameters[module] = dict_deep_update(self.parameters[module], updates)
        else:
            self.parameters[module] = updates

    def reset_defaults(self, module):
        """
        Reset the default parameters for a specific module or class.

        Parameters
        ----------
        module : str
            The name of the module or class to reset defaults for.

        Returns
        -------
        None
        """
        if module in self.parameters:
            self._initializer_map[module]()


#%%
class Params:
    """
    
    """

    def __init__(self):
        self.params = {}

    def params_default(self):
        nprm = NonlinearVariationalSolver.default_parameters()
        lprm = LinearVariationalSolver.default_parameters()

    def solver_default(self):
        info('-' * 78)
        list_linear_solver_methods()
        info('-' * 62)
        list_krylov_solver_preconditioners()

    def NewtonSolver(self, params={}):
        self.newton_solver = {'linear_solver': 'mumps',
                              'absolute_tolerance': 1e-12,
                              'relative_tolerance': 1e-12,
                              'convergence_criterion': 'residual',
                              'error_on_nonconvergence': True,
                              'krylov_solver': self.krylov_solver,
                              'lu_solver': self.lu_solver,
                              'maximum_iterations': 50,
                              'preconditioner': 'default',
                              'relaxation_parameter': None,
                              'report': True
                              }
        self.newton_solver.update(params)

    def SnesSolver(self, params={}):
        self.snes_solver = {'absolute_tolerance': 1e-10,
                            'error_on_nonconvergence': True,
                            'krylov_solver': self.krylov_solver,
                            'line_search': 'basic',
                            'linear_solver': 'default',
                            'lu_solver': self.lu_solver,
                            'maximum_iterations': 50,
                            'maximum_residual_evaluations': 2000,
                            'method': 'default',
                            'preconditioner': 'default',
                            'relative_tolerance': 1e-09,
                            'report': True,
                            'sign': 'default',
                            'solution_tolerance': 1e-16
                            }
        self.snes_solver.update(params)

    def KrylovSolver(self, params={}):
        self.krylov_solver = {'absolute_tolerance': 1e-10,
                              'divergence_limit': 1e5,
                              'error_on_nonconvergence': True,
                              'maximum_iterations': 1000,
                              'monitor_convergence': True,
                              'nonzero_initial_guess': False,
                              'relative_tolerance': 1e-8,
                              'report': False
                              }
        self.krylov_solver.update(params)

    def LUSolver(self, params={}):
        self.lu_solver = {'report': True,
                          'symmetric': False,
                          'verbose': False,
                          'reuse_factorization': True  # not sure if it worked even para.add() method is not used
                          }
        self.lu_solver.update(params)

    def NonLinearSolver(self, params={}):
        self.params.update({'newton_solver': self.newton_solver,
                            'nonlinear_solver': 'newton',
                            'print_matrix': False,
                            'print_rhs': False,
                            'snes_solver': self.snes_solver,
                            'symmetric': False
                            })
        self.params.update(params)

    def LinearSolver(self, params={}):
        self.params.update({'linear_solver': {
            'krylov_solver': self.krylov_solver,
            'linear_solver': 'default',
            'lu_solver': self.lu_solver,
            'preconditioner': 'default',
            'print_matrix': False,
            'print_rhs': False,
            'symmetric': False
        }})
        self.params.update(params)

    def IPCSolver(self, params={}):
        self.ipcs_solver = {'method': 'lu',
                            'linear_solver': 'mumps',
                            'lu_solver': self.lu_solver,
                            'krylov_solver': self.krylov_solver,

                            'maximum_iterations': 20,
                            'convergence_criterion': 'linf',
                            'absolute_tolerance': 1e-7,
                            'relative_tolerance': 1e-7,
                            'relaxation_parameter': 1.0,
                            'report': False
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
