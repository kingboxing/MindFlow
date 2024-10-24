#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `DefaultParameters` class, which serves as a centralized manager for default parameters across various modules, classes, or functions within the FERePack package.

The class allows other parts of the package to access, modify, and reset default parameter values, ensuring consistency and flexibility in parameter management.
It initializes default parameters for different solvers and analyses, such as eigenvalue solvers, frequency response solvers, resolvent solvers, and Riccati equation solvers.

Classes
-------
- DefaultParameters:
    Manages default parameters for different modules, classes, or functions.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `DefaultParameters` and accessing or updating default parameters for specific modules.

```python
from FERePack.Params.Params import DefaultParameters

# Initialize the default parameters manager
params_manager = DefaultParameters()

# Get default parameters for a specific module
eigen_solver_defaults = params_manager.get_defaults('eigen_solver')

# Update default parameters for a module
new_defaults = {'solver_type': 'custom_solver', 'custom_param': 42}
params_manager.update_defaults('eigen_solver', new_defaults)

# Reset default parameters for a module
params_manager.reset_defaults('eigen_solver')
```

Notes
--------

- The DefaultParameters class is essential for maintaining consistent parameter configurations across different components of the FERePack package.
- The class initializes default parameters using internal methods and provides interfaces to get, update, and reset parameters for specific modules.
"""

from ..Deps import *
from ..LinAlg.Utils import dict_deep_update


class DefaultParameters:
    """
    Manages default parameters for different modules, classes, or functions.

    This class provides a centralized management system for default parameters used throughout the FERePack package. It initializes default parameters for various components and allows other parts of the package to access, modify, and reset these defaults.

    Attributes
    ----------
    parameters : dict
        A dictionary containing default parameters for different modules or classes.

    Methods
    -------
    get_defaults(module)
        Retrieve the default parameters for a specified module or class.
    update_defaults(module, updates)
        Update the default parameters for a specified module or class with provided values.
    reset_defaults(module)
        Reset the default parameters for a specified module or class to their initial values.

    Notes
    -----
    - The class initializes default parameters using private methods for each module or class.
    - Default parameters can be accessed and modified dynamically, providing flexibility in configuring solvers and analyses.
    """
    def __init__(self):
        """
        Initialize the DefaultParameters class.

        This constructor initializes the `parameters` dictionary and calls the `_initialize_parameters` method to set up default parameters for various modules, classes, or functions.
        """
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initialize default parameters for all supported modules.

        This method creates a mapping between module names and their respective initializer methods. It then calls each initializer to set up the default parameters.
        """
        self._initializer_map = {'eigen_decompose':     self._initialize_eigen_decompose_defaults,
                                 'eigen_solver':        self._initialize_eigen_solver_defaults,
                                 'frequency_response':  self._initialize_freq_defaults,
                                 'resolvent_solver':    self._initialize_resolvent_defaults,
                                 'state_space_model':   self._initialize_ssmodel_defaults,
                                 'bernoulli_feedback':  self._initialize_bernoulli_defaults,
                                 'nmriccati_pymess':    self._initialize_nmriccati_pymess_defaults,
                                 'newton_solver':       self._initialize_newton_defaults,
                                 'IPCS_solver':         self._initialize_IPCS_defaults,
                                 'nmriccati_mmess':     self._initialize_nmriccati_mmess_defaults,
                                 'radiriccati_mmess':   self._initialize_radiriccati_mmess_defaults,
                                 }

        for key in self._initializer_map:
            self._initializer_map[key]()

    def _initialize_eigen_decompose_defaults(self):
        """
        Initialize default parameters for eigen-decomposition functions.

        Parameters initialized include (see scipy.sparse.linalg.eigs):
            - method: str, the method used to solve the linear problems (e.g., 'lu')
            - lusolver: str, the LU decomposition solver (e.g., 'mumps')
            - echo: bool, whether prints the number of times the linear system has been solved (default False)
            - which: str (default 'LM')
            - v0: numpy array (default None)
            - ncv: int (default None)
            - maxiter: int (default None)
            - tol: float (default 0)
            - return_eigenvectors: bool (default True)
            - OPpart: None or str (default None)
        """
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
        Initialize default parameters for the eigen-solver class of linearized Navier-Stokes equations.

        Parameters initialized include:
            - feedback_pencil: dict, optional,
                - a dict with U, V to formulate A + U * V'
            - symmetry : bool, optional
                - whether assemble matrices in the symmetry way. Default is True.
            - solve_type : str, optional
                - Solver type ('implicit' or 'explicit'). Default is 'implicit'.
            - inverse : bool, optional
                - Whether to compute the inverse of the RHS eigenvectors. Default is False.
            - BCpart : ‘r’ or ‘i’/None, optional
                - The homogenous boundary conditions are applied in the real (matrix NS) or imag (matrix M) part of system. The default is None.
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
                    'feedback_pencil': None, # either None or a dict with U, V
                    }

        self.parameters['eigen_solver'] = defaults

    def _initialize_freq_defaults(self):
        """
        Initialize default parameters for the frequency response class.

        Parameters initialized include:
            - method: str, the method used to solve the linear problems (e.g., 'lu')
            - lusolver: str, the LU decomposition solver (e.g., 'mumps')
            - echo: bool, whether prints the number of times the linear system has been solved (default False)
            - symmetry: bool, whether assemble matrices in the symmetry way. Default is True.
            - BCpart: ‘r’ or ‘i’/None, the homogenous boundary conditions are applied in the real (matrix NS) or imag (matrix M) part of system. The default is None.
            - feedback_pencil: dict, a dict with U, V to formulate A + U * V'
        """
        defaults = {'solver_type': 'frequency_response',
                    'frequency_response': {'method': 'lu',
                                           'lusolver': 'mumps',
                                           'echo': False,
                                           'symmetry': True,
                                           'BCpart': None},
                    'feedback_pencil': None,
                    }
        self.parameters['frequency_response'] = defaults

    def _initialize_resolvent_defaults(self):
        """
        Initialize default parameters for the resolvent solver class.

        Parameters initialized include:
            - symmetry: bool, whether assemble matrices in the symmetry way. Default is True.
            - BCpart: ‘r’ or ‘i’/None, the homogenous boundary conditions are applied in the real (matrix NS) or imag (matrix M) part of system. The default is None.
            - feedback_pencil: dict, a dict with U, V to formulate A + U * V'
        """
        defaults_eigs = copy.deepcopy(self.parameters['eigen_decompose'])
        defaults_eigs.update({'symmetry': True, 'BCpart': None})
        defaults = {'solver_type': 'resolvent_solver',
                    'resolvent_solver': defaults_eigs,
                    'feedback_pencil': None,
                    }
        self.parameters['resolvent_solver'] = defaults

    def _initialize_ssmodel_defaults(self):
        """
        Initialize default parameters for the state-space model class.

        Parameters initialized include:

        """
        defaults = {'solver_type': 'state_space_model',
                    'state_space_model': {}
                    }
        self.parameters['state_space_model'] = defaults

    def _initialize_bernoulli_defaults(self):
        """
        Initialize default parameters for the Bernoulli solver class.

        Parameters initialized include:
            - k: int, number of unstable eigenvalues
            - sigma: float, shift values
            - which: str, eigenvalue computation settings
        """
        defaults_eigs = copy.deepcopy(self.parameters['eigen_decompose'])
        defaults_diff = {'k': 2,
                         'sigma': 0.0,
                         'which': 'LR',  # compute unstable eigenvalues
                         }
        defaults_eigs.update(defaults_diff)
        defaults = {'solver_type': 'bernoulli_feedback',
                    'bernoulli_feedback': defaults_eigs,
                    }

        self.parameters['bernoulli_feedback'] = defaults

    def _initialize_nmriccati_pymess_defaults(self):
        """
        Initialize default parameters for the Riccati solver class using the Newton method and PyMESS package.

        Parameters initialized include (see details in PyMESS documentation):
            - method: 'nm'
                - Method for solving Riccati equation.
            - backend: 'python'
                - Platform backend to use.
            - type: operation type
            - lusolver: direct solver type
                - MESS_DIRECT_DEFAULT_LU 0, the same as UMFPACK
                - MESS_DIRECT_SPARSE_LU 1, too long time to check
                - MESS_DIRECT_LAPACKLU 2,
                - MESS_DIRECT_UMFPACK_LU: 3, normal
                - MESS_DIRECT_SUPERLU_LU : 4, check error occured, kernel died
                - MESS_DIRECT_CSPARSE_LU 5, too long time to check
                - MESS_DIRECT_BANDED_LU 6,
                - MESS_DIRECT_MKLPARDISO_LU 7
            - delta: real and negativ, common choice -0.02
            - adi.memory_usage: memory usage option
            - adi.paratype: Shift parameter type
            - adi.output: 0 or 1 Turn output in ADI method on/off.
            - adi.res2_tol: real relative 2-norm stopping tolerance in ADI method
            - adi.maxit: maximum number of iterations in ADI method
            - nm.output: 0 or 1 Turn output in Newton method on/off.
            - nm.singleshifts: 0 or 1 Turn single shift strategy on or off.
            - nm.linesearch: 0 or 1 Turn line search on or off.
            - nm.res2_tol: real Stopping criteria, 2-norm of relative residual.
            - nm.maxit: maximum number of iterations in Newton method.
            - nm.k0: array_like real matrix Initial Feedback.
        """
        defaults_pymess = {'solver_type':   'riccati_solver',
                           'method':        'nm',
                           'backend':       'python',
                           'riccati_solver': {'type':       0,  #mess.MESS_OP_NONE
                                              'lusolver':   3,  #mess.MESS_DIRECT_UMFPACK_LU
                                              'delta':      -0.02,
                                              'adi': {'memory_usage': 2,  #mess.MESS_MEMORY_HIGH
                                                      'paratype': 3,  #mess.MESS_LRCFADI_PARA_ADAPTIVE_V
                                                      'output': 0,
                                                      'res2_tol': 1e-10,
                                                      'maxit': 2000
                                                      },
                                              'nm': {'output': 1,
                                                     'singleshifts': 0,
                                                     'linesearch': 1,
                                                     'res2_tol': 1e-8,
                                                     'maxit': 30,
                                                     'k0': None  #Initial Feedback
                                                     }
                                              }
                           }

        self.parameters['nmriccati_pymess'] = defaults_pymess

    def _initialize_nmriccati_mmess_defaults(self):
        """
        Initialize default parameters for the Riccati solver class using the Newton method with the M-M.E.S.S. (Matlab) backend.
        
        Parameters initialized include (see detail in MMESS documentation):
            - norm                      possible  values: 2, 'fro'
                                        use 2-norm (2) or Frobenius norm ('fro') to
                                        compute residual and relative change norms
                                        (optional, default: 'fro')
    
            - LDL_T                     possible  values: false, true
                                        use LDL^T formulation for the RHS and
                                        solution
                                        (optional, default: false)
    
            - eqn.type                  possible  values: 'N', 'T'
                                        determining whether (N) or (T) is solved
                                        (optional, default fallback: 'N')
    
            - eqn.haveE                 possible  values: false, true
                                        if haveE = false: matrix E is assumed to be the identity
                                        (optional, default: false)
    
            - eqn.haveUV                possible  values: false, true
                                        if haveUV = true: U = [U1, U2] and V = [V1, V2]
                                        if K or DeltaK are accumulated during the iteration they
                                        use only U2 and V2. U1 and V1 can be used for an external
                                        rank-k update of the operator.
                                        The size of U1 and V1 can be given via eqn.sizeUV1.
                                        (optional, default: false if no U and V are given)
    
            - eqn.sizeUV1               possible values: nonnegative integer
                                        if a stabilizing feedback is given via U and
                                        V, eqn.sizeUV1  indicates the number of columns/rows of U and V
                                        (optional, default: size(eqn.U, 2))
    
            - adi.maxiter               possible  values: integer > 0
                                        maximum iteration number
                                        (optional, default: 100)
    
            - adi.res_tol               possible  values: scalar >= 0
                                        stopping tolerance for the relative
                                        residual norm; if res_tol = 0 the relative
                                        residual norm is not evaluated
                                        (optional, default: 0)
    
            - adi.rel_diff_tol          possible  values: scalar >= 0
                                        stopping tolerance for the relative
                                        change of the solution Z;
                                        if res_tol = 0 the relative
                                        change is not evaluated
                                        (optional, default: 0)
    
            - adi.info                  possible  values: 0, 1
                                        turn on (1) or off (0) the status output in
                                        every iteration step
                                        (optional, default: 0)
    
            - adi.compute_sol_fac       possible  values: false, true
                                        turn on (1) or off (0) the computation of
                                        the factored solution; turn off if only the
                                        feedback matrix K is of interest
                                        (optional, default: true)
    
            - adi.accumulateK           possible  values: false, true
                                        accumulate the feedback matrix K during the
                                        iteration (optional, default: false)
    
            - adi.accumulateDeltaK      possible  values: false, true
                                        accumulate the update DeltaK of the
                                        feedback matrix K during the iteration
                                        (optional, default: false)
    
            - nm.K0                     possible values: dense 'T': (m1 x n) matrix
                                        or 'N': (m2 x n) matrix
                                        initial stabilizing feedback K
                                        (optional)
    
            - nm.maxiter                possible  values: integer > 0
                                        maximum NM iteration number
                                        (optional, default: 20)
    
            - nm.res_tol                possible values: scalar >= 0
                                        stopping tolerance for the relative NM
                                        residual norm; if res_tol = 0 the relative
                                        residual norm is not evaluated
                                        (optional, default: 0)
    
            - nm.rel_diff_tol           possible values: scalar >= 0
                                        stopping tolerance for the relative
                                        change of the NM solution Z;
                                        if res_tol = 0 the relative
                                        change is not evaluated
                                        (optional, default: 0)
    
            - nm.info                   possible  values: 0, 1
                                        turn on (1) or off (0) the status output in
                                        every NM iteration step
                                        (optional, default: 0)
    
            - nm.accumulateRes          possible  values: false, true
                                        accumulate the relative NM residual norm
                                        during the inner ADI iteration
                                        (optional, default: false)
    
            - nm.linesearch             possible  values: false, true
                                        if turned of (false) NM makes full steps; if
                                        turned on (true) a step size 0<=lambda<=2 is
                                        computed (optional, default: false)
    
            - nm.inexact                possible  values: false, 'linear',
                                        'superlinear', 'quadratic'
                                        the inner ADI uses an adaptive relative ADI
                                        residual norm; with
                                        ||R||: relative NM residual norm,
                                        tau: - nm.tau,
                                        j: NM iteration index;
                                        'linear': tau * ||R||,
                                        'superlinear':  ||R|| / (j^3 + 1),
                                         'quadratic': tau / sqrt(||R||)
                                        (optional, default: false)
    
            - nm.tau                    possible  values: scalar >= 0
                                        factor for inexact inner ADI iteration
                                        tolerance (optional, default: 1)
    
            - nm.projection.freq        possible  values: integer >= 0
                                        frequency of the usage of Galerkin
                                        projection acceleration in NM
                                        (optional, default: 0)
    
            - nm.projection.ortho       possible  values: false, true
                                        implicit (false) or explicit (true)
                                        orthogonalization of Z in LRNM
                                        i.e., explicit orthogonalization via orth().
                                        (optional, default: true, ignored in framework KSM)
    
            - nm.projection.meth        method for solving projected Lyapunov or
                                        Riccati equation. Depending on 'type' possible
                                        values are: 'lyapchol', 'lyap_sgn_fac', 'lyap',
                                        'lyapunov', 'lyap2solve', 'icare', 'care',
                                        'care_nwt_fac', 'mess_dense_nm'
                                        (optional, default: best available solver for the
                                        type depending on the presence of the control
                                        toolbox/package and version of Matlab/Octave used.)
                                        Remark: some solver are disallowed/excluded when
                                        - LDL_T is 'true'.
    
            - shifts.num_desired        possible  values: integer > 0
                                        number of shifts that should be computed
                                        2*num_desired < num_Ritz + num_hRitz is required
                                        (optional, default: 25)
    
            - shifts.num_Ritz           possible  values: integer > 0
                                        number of Arnoldi steps w.r.t. F for
                                        heuristic shift computation
                                        num_Ritz < n is required
                                        (optional, default: 50)
    
            - shifts.num_hRitz          possible  values: integer > 0
                                        number of Arnoldi steps w.r.t. inv(F) for
                                        heuristic shift computation
                                        num_hRitz < n is required
                                        (optional, default: 25)
    
            - shifts.method             possible  values:
                                        'heuristic', ('heur', 'Penzl', 'penzl')
                                        for Penzl's heuristics.
                                        'wachspress', ('Wachspress')
                                        for asymptotically optimal Wachspress
                                        selection.
                                        'projection'
                                        for adaptively updated projection shifts.
                                        method for shift computation
                                        in case of 'projection' new shifts are
                                        computed during the iteration steps,
                                        otherwise the shifts are reused cyclically
                                        (optional, default: 'heuristic')
    
            - shifts.period             possible  values: integer > 0
                                        number of NM iterations that should pass
                                        until new shifts are computed
                                        (optional, default: 1)
    
            - shifts.b0                 (n x 1) array
                                        start vector for Arnoldi algorithm for
                                        heuristic shift computation
                                        (optional, default: ones(n, 1))
        """
        defaults_mmess = {'solver_type':    'riccati_solver',
                          'method':         'nm',
                          'backend':        'matlab',
                          'riccati_solver': { 'norm':   2, # possible  values: 2, 'fro'
                                              'LDL_T':  False, # use LDL^T formulation for the RHS and solution
                                              'eqn':    { 'type':    'N', # possible values: 'N', 'T'
                                                          'haveE':   True,
                                                          'haveUV':  False,
                                                          'sizeUV1': 'default'
                                                        },
                                              'adi':    { 'maxiter':            100,
                                                          'res_tol':            1e-12,
                                                          'rel_diff_tol':       1e-16,
                                                          'info':               0,
                                                          'compute_sol_fac':    True,
                                                          'accumulateK':        False,
                                                          'accumulateDeltaK':   False
                                                        },
                                              'nm':     { 'K0':             'default', # i.e. None
                                                          'maxiter':        20,
                                                          'res_tol':        1e-10,
                                                          'rel_diff_tol':   1e-16,
                                                          'info':           0,
                                                          'accumulateRes':  True,
                                                          'linesearch':     True,
                                                          'inexact':        'superlinear',
                                                          'tau':            0.1,
                                                          'projection':     { 'freq':   0,
                                                                              'ortho':  True,
                                                                              'meth':   'default',
                                                                            },
                                                          'res':            { 'maxiter': 10,
                                                                              'tol':     1e-6,
                                                                              'orth':    0,
                                                                            }
                                                        },
                                              'shifts': { 'num_desired':    25,
                                                          'num_Ritz':       50,
                                                          'num_hRitz':      25,
                                                          'method':         'heuristic',
                                                          'period':         1,
                                                          'b0':             'default',  # default: ones(n, 1), (n x 1) array, start vector for Arnoldi algorithm for heuristic shift computation
                                                         },
                                            }
                          }

        self.parameters['nmriccati_mmess']=defaults_mmess

    def _initialize_radiriccati_mmess_defaults(self):
        """
        Initialize default parameters for the Riccati solver class using the RADI method with the M-M.E.S.S. (Matlab) backend.

        Parameters initialized include (see detail in MMESS documentation):
            - norm                      possible  values: 2, 'fro'
                                        use 2-norm (2) or Frobenius norm ('fro') to
                                        compute residual and relative change norms
                                        (optional, default: 'fro')

            - LDL_T                     possible  values: false, true
                                        use LDL^T formulation for the RHS and
                                        solution
                                        (optional, default: false)

            - eqn.type                  possible  values: 'N', 'T'
                                        determining whether (N) or (T) is solved
                                        (optional, default fallback: 'N')

            - eqn.haveE                 possible  values: false, true
                                        if haveE = false: matrix E is assumed to be the identity
                                        (optional, default: false)

            - eqn.haveUV                possible  values: false, true
                                        if haveUV = true: U = [U1, U2] and V = [V1, V2]
                                        if K or DeltaK are accumulated during the iteration they
                                        use only U2 and V2. U1 and V1 can be used for an external
                                        rank-k update of the operator.
                                        The size of U1 and V1 can be given via eqn.sizeUV1.
                                        (optional, default: false if no U and V are given)

            - eqn.sizeUV1               possible values: nonnegative integer
                                        if a stabilizing feedback is given via U and
                                        V, eqn.sizeUV1  indicates the number of columns/rows of U and V
                                        (optional, default: size(eqn.U, 2))

            - radi.Z0                   possible values: dense (n x m4) matrix
                                        initial stabilizing solution factor
                                        X0 = Z0*inv(Y0)*Z0', this factor has to
                                        result in a positive semi-definite Riccati
                                        residual W0 (optional, default: zeros(n, m4))

            - radi.Y0                   possible values: dense (m4 x m4) matrix
                                        initial stabilizing solution factor
                                        X0 = Z0*inv(Y0)*Z0', this factor has to
                                        result in a positive semi-definite
                                        Riccati residual W0 (optional, default: eye(m4))

            - radi.W0                   possible values: dense (n x m5) matrix
                                        initial Riccati residual factor such that
                                        R(X0) = W0 * W0', if
                                        - radi.compute_res = true, this factor is
                                        computed out of Z0 and Y0
                                        Note: In case of Bernoulli stabilization
                                        the W0 is given by the right hand-side C'
                                        for 'T' and B for 'N' and is automatically
                                        set if - radi.compute_res = false
                                        (optional, default: C' for 'T' or B for
                                        'N')

            - radi.T0                   possible values: dense (m5 x m5) matrix
                                        initial Riccati residual factor such that
                                        R(X0) = W0 * T0 * W0', if
                                        - radi.compute_res = true, this factor is
                                        computed out of Z0 and Y0
                                        (required for LDL^T formulation if
                                        - radi.W0 was explicitly set)

            - radi.K0                   possible values: dense 'T': (m1 x n)
                                        matrix, 'N':  (m2 x n) matrix
                                        initial K (corresponding to Z0 and Y0)
                                        Note: If K0 is given without Z0, only the
                                        resulting stabilizing feedback is computed.
                                        Also it has to correspond to W0.
                                        (optional, default: E*Z0*inv(Y0)*Z0'*C' for
                                        'N' or E'*Z0*inv(Y0)*Z0'*B for 'T')

            - radi.compute_sol_fac      possible values: false, true
                                        turn on (true) or off (false) to compute the
                                        solution of the Riccati equation and use it
                                        internally for computations, or only
                                        the stabilizing feedback
                                        (optional, default: true)

            - radi.get_ZZt              possible values: false, true
                                        turn on (true) or off (false) to compute only
                                        the low-rank decomposition X = Z*Z'
                                        without the middle term Y
                                        (optional, default: true)

            - radi.compute_res          possible values: false, true
                                        turn on (1) or off (0) to compute the
                                        residual corresponding to the initial
                                        solution factors Z0, Y0, if 0 then the
                                        right hand-side is used as residual if
                                        there is no W0 (optional, default: true)

            - radi.maxiter              possible  values: integer > 0
                                        maximum RADI iteration number
                                        (optional, default: 100)

            - radi.res_tol              possible  values: scalar >= 0
                                        stopping tolerance for the relative
                                        RADI residual norm; if res_tol = 0 the
                                        relative residual norm is not evaluated
                                        (optional, default: 0)

            - radi.rel_diff_tol         possible  values: scalar >= 0
                                        stopping tolerance for the relative
                                        change of the RADI solution Z;
                                        if res_tol = 0 the relative
                                        change is not evaluated
                                        (optional, default: 0)

            - radi.info                 possible  values: 0, 1
                                        turn on (1) or off (0) the status output in
                                        every RADI iteration step
                                        (optional, default: 0)

            - radi.trunc_tol            possible values: scalar > 0
                                        tolerance for rank truncation of the
                                        low-rank solutions (aka column compression)
                                        (optional, default: eps*n)

            - radi.trunc_info           possible values: 0, 1
                                        verbose mode for column compression
                                        (optional, default: 0)

            - shifts.method             possible  values:
                                        'precomputed',
                                        'penzl','heur', (basic MMESS routine)
                                        'projection' (basic MMESS routine)
                                        'gen-ham-opti' (special for RADI)
                                        method for shift computation
                                        (optional, default: 'gen-ham-opti')

            - shifts.history            possible values: integer * size(W0, 2) > 0
                                        parameter for accumulating the history
                                        of shift computations
                                        (optional, default: 6 * columns of residual)

            - shifts.info               possible  values: 0, 1
                                        turn output of used shifts before the first
                                        iteration step on (1) or off (0) (optional, default: 0)

        """

        # change default in above doc
        defaults_mmess = {'solver_type': 'riccati_solver',
                          'method': 'radi',
                          'backend': 'matlab',
                          'riccati_solver': {'norm': 2,  # possible  values: 2, 'fro'
                                             'LDL_T': False,  # use LDL^T formulation for the RHS and solution
                                             'eqn': {'type': 'N',  # possible values: 'N', 'T'
                                                     'haveE': True,
                                                     'haveUV': False,
                                                     'sizeUV1': 'default'
                                                     },
                                             'shifts': {    'history':              'default',
                                                            'method':               'gen-ham-opti',
                                                            'info':                 0,
                                                            'naive_update_mode':    False,
                                                            'num_desired':          25
                                                        },
                                             'radi': {  'Z0':               'default',
                                                        'Y0':               'default',
                                                        'W0':               'default',
                                                        'T0':               'default',
                                                        'K0':               'default',
                                                        'compute_sol_fac':  True,
                                                        'get_ZZt':          True,
                                                        'compute_res':      True,
                                                        'maxiter':          300,
                                                        'res_tol':          1e-10,
                                                        'rel_diff_tol':     0,
                                                        'info':             0,
                                                        'trunc_tol':        'default',
                                                        'trunc_info':       'default',
                                                    }
                                             }
                          }
        self.parameters['radiriccati_mmess'] = defaults_mmess


    def _initialize_newton_defaults(self):
        """
        Initialize default parameters for the Newton method in DNS solvers.

        Parameters initialized include (see details in FEniCS documentation):
            - pending for implementation

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
        Initialize default parameters for the IPCS (Incremental Pressure Correction Scheme) method in DNS solvers.

        Parameters initialized include:
            - pending for implementation.
        """
        defaults = {'solver_type': 'IPCS_solver',
                    'IPCS_solver': {},
                    }
        self.parameters['IPCS_solver'] = defaults

    def get_defaults(self, module):
        """
        Retrieve the default parameters for a specified module or class.

        Parameters
        ----------
        module : str
            The name of the module or class to retrieve defaults for.

        Returns
        -------
        dict
            A dictionary containing the default parameters for the specified module or class.

        Examples
        --------
        ```python
        defaults = params_manager.get_defaults('eigen_solver')
        ```
        """
        return self.parameters.get(module, {})

    def update_defaults(self, module, updates):
        """
        Update the default parameters for a specified module or class with provided values.

        Parameters
        ----------
        module : str
            The name of the module or class to update defaults for.
        updates : dict
            A dictionary containing parameter updates.

        Returns
        -------
        None

        Examples
        --------
        ```python
        new_defaults = {'solver_type': 'custom_solver', 'custom_param': 42}
        params_manager.update_defaults('eigen_solver', new_defaults)
        ```
        """
        if module in self.parameters:
            self.parameters[module] = dict_deep_update(self.parameters[module], updates)
        else:
            self.parameters[module] = updates

    def reset_defaults(self, module):
        """
        Reset the default parameters for a specified module or class to their initial values.

        Parameters
        ----------
        module : str
            The name of the module or class to reset defaults for.

        Returns
        -------
        None

        Examples
        --------
        ```python
        params_manager.reset_defaults('eigen_solver')
        ```
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

