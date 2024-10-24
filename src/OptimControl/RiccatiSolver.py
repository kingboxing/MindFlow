#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `GRiccatiDAE2Solver` class for solving generalized Riccati equations for index-2 systems using the M.E.S.S. (Matrix Equation Sparse Solver) library.

The class supports both Python (PyM.E.S.S.) and MATLAB (M-M.E.S.S.) backends and allows for solving the Riccati
equation using methods like Newton's method (NM) and the Rational Arnoldi (RADI) method. It also provides
functionality to compute performance metrics such as the H2 norm of the system.

Classes
-------
- GRiccatiDAE2Solver:
    Solves generalized Riccati equations for index-2 systems and computes performance metrics.

Dependencies
------------
- FEniCS
- NumPy
- SciPy
- M.E.S.S. library (Python or MATLAB version)
- joblib
- multiprocessing

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `GRiccatiDAE2Solver` with a state-space model, setting
the desired method and backend, solving the Riccati equation, and computing performance metrics.

```python
from FERePack.OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from FERePack.OptimControl import StateSpaceDAE2

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0

# Initialize the state-space model
model = StateSpaceDAE2(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
model.set_boundary(bc_list)
model.set_boundarycondition(bc_list)

# Set base flow
model.set_baseflow(ic=base_flow_function)

# Define input and output vectors
input_vec = ...  # Define your input vector (actuation)
output_vec = ...  # Define your output vector (measurement)

# Assemble the state-space model
model.assemble_model(input_vec=input_vec, output_vec=output_vec)

# Initialize the Riccati solver with the model
riccati_solver = GRiccatiDAE2Solver(model, method='nm', backend='python')

# Solve the Riccati equation
riccati_solver.solve_riccati()

# Access the solution factor (Z) of the Riccati equation
solution_factor = riccati_solver.facZ

# Compute the squared H2 norm of the closed-loop system
h2_norm_squared = riccati_solver.sys_energy(MatQ)
```

Notes
--------

- The class handles both continuous-time algebraic Riccati equations and provides options to choose different solution methods and backends.
- It supports both low-rank and full-rank solutions, depending on the method used.
- The computed solution can be used for controller synthesis, stability analysis, and performance evaluation.

"""

from ..Deps import *

from ..OptimControl.SystemModel import StateSpaceDAE2
from ..LinAlg.Utils import distribute_numbers, deep_set_attr, assemble_dae2, cholesky_sparse, invert_diag_block_matrix
from ..Params.Params import DefaultParameters
from ..Interface.Py2Mat import python2matlab, matlab2python, start_matlab


class GRiccatiDAE2Solver:
    """
    Solves a generalized Riccati equation for an index-2 system and computes various performance metrics like the H2 norm.

    Supports both Python (PyM.E.S.S.) and MATLAB (M-M.E.S.S.) backends.

    Parameters
    ----------
    model : dict or StateSpaceDAE2
        State-space model or dictionary containing system matrices.
    method : str, optional
        Method used to solve generalized Riccati equations ('nm' or 'radi'). Default is 'nm'.
    backend : str, optional
        Backend used to solve generalized Riccati equations ('python' or 'matlab'). Default is 'python'.

    Attributes
    ----------
    eqn : dict
        Dictionary containing system matrices and parameters.
    param : dict
        Default parameters for the Riccati solver.
    facZ : numpy.ndarray or dict
        Solution factor of the Riccati equation.
    status : dict
        Status information from the solver.
    sys_energy : function
        Function to compute the squared H2 norm of the system.
    gain_energy : function
        Function to compute the contribution of the estimator/regulator on the H2 norm.

    Methods
    -------
    solve_riccati(engine=None)
        Solve the generalized Riccati equation.
    sys_energy(MatQ, pid=None, chunk_size=5000)
        Compute the squared H2 norm of the closed-loop system.
    gain_energy(K)
        Compute the contribution of each estimator/regulator on the H2 norm.
    _feedback_pencil(U, V, mode)
        Assemble the feedback pencil for the system.

    Notes
    -----
    - The class supports solving the Riccati equation using either Newton's method or the RADI method.
    - It can interface with both Python and MATLAB versions of the M.E.S.S. library.
    - The solution factor `facZ` can be used to reconstruct the solution of the Riccati equation.
    - The computed feedback can be applied to the system to achieve stabilization or performance improvements.
    """

    def __init__(self, model, method='nm', backend='python'):
        """
        Initialize the Riccati solver with a state-space model.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model or dictionary containing system matrices.
        method : str, optional
            Method used to solve generalized Riccati equations ('nm' or 'radi'). Default is 'nm'.
        backend : str, optional
            Backend used to solve generalized Riccati equations ('python' or 'matlab'). Default is 'python'.
        """
        self._default_parameters(method, backend)
        self._extract_model(model)
        self.status = None
        self.facZ = None
        self.sys_energy = self._squared_h2norm_system
        self.gain_energy = self._squared_h2norm_gain

    def _default_parameters(self, method, backend):
        """
        Set default parameters based on the method and backend.

        Parameters
        ----------
        method : str
            Method used to solve generalized Riccati equations ('nm' or 'radi').
        backend : str
            Backend used to solve generalized Riccati equations ('python' or 'matlab').
        """
        if method == 'nm' and backend == 'python':
            self.param = DefaultParameters().parameters['nmriccati_pymess']
            self.param['mess_options'] = mess.Options()
        elif method == 'nm' and backend == 'matlab':
            self.param = DefaultParameters().parameters['nmriccati_mmess']
        elif method == 'radi' and backend == 'matlab':
            self.param = DefaultParameters().parameters['radiriccati_mmess']

    def _extract_model(self, model):
        """
        Assign the state-space model to `self.eqn`.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model.

        Raises
        ------
        TypeError
            If the input `model` is not a valid state-space model.
        """
        self.eqn = {}
        if isinstance(model, dict):
            self.eqn.update({k: model[k] for k in ['B', 'C', 'M', 'A', 'G']})
            if self.param['backend'] == 'matlab' and 'U' in model and 'V' in model:
                self.eqn.update({'U': model['U'], 'V': model['V']})
                self.param['riccati_solver']['eqn'].update({'haveUV': True, 'sizeUV1': 'default'})

        elif isinstance(model, StateSpaceDAE2):
            self.eqn.update({'B': model.B, 'C': model.C, 'M': model.M, 'A': model.A, 'G': model.G})
            if self.param['backend'] == 'matlab' and hasattr(model, 'U') and hasattr(model, 'V'):
                self.eqn.update({'U': model.U, 'V': model.V})
                self.param['riccati_solver']['eqn'].update({'haveUV': True, 'sizeUV1': 'default'})
        else:
            raise TypeError('Invalid type for state-space model.')

    def _apply_parameters(self):
        """
        Apply the Riccati solver parameters to the solver.
        """
        if self.param['method'] == 'nm' and self.param['backend'] == 'python':
            param = self.param['riccati_solver']
            if 'lusolver' in param:
                mess.direct_select(param.pop('lusolver'))
            deep_set_attr(self.param['mess_options'], param)
        elif self.param['backend'] == 'matlab':
            self.opts = python2matlab(self.param['riccati_solver'])

    def _apply_matlab_system(self):
        """
        Formulate system matrices when using the MATLAB backend.

        Returns
        -------
        eqn_mat : dict
            Dictionary containing the system matrices in MATLAB format.
        """
        A_full, E_full = assemble_dae2(self.eqn)
        # Update state-space model with assembled matrices
        eqn_mat = {'E_': E_full, 'A_': A_full, 'B': self.eqn['B'], 'C': self.eqn['C']}
        if self.param['riccati_solver']['eqn']['haveUV']:
            eqn_mat.update({'U': self.eqn['U'], 'V': self.eqn['V']})
        if self.param['riccati_solver']['LDL_T']:
            eqn_mat.update({'Q': self.eqn['Q'], 'R': self.eqn['R']})

        return python2matlab(eqn_mat)

    def _apply_matlab_solver(self, eng):
        """
        Solve the Riccati equation using the MATLAB backend.

        Parameters
        ----------
        eng : matlab.engine.MatlabEngine
            MATLAB engine instance.
        """
        self._apply_parameters()
        eqn_mat = self._apply_matlab_system()
        if self.param['method'] == 'nm':
            output = eng.GRiccatiDAE2NMSolver(eqn_mat, self.opts)
        elif self.param['method'] == 'radi':
            output = eng.GRiccatiDAE2RADISolver(eqn_mat, self.opts)

        out = matlab2python(output)
        # assume all dense matrices
        facZ = out.pop('Z', None)
        facY = out.pop('Y', None)  # possible diagonal block matrix
        facD = out.pop('D', None)  # possible diagonal matrix
        fedK = out.pop('K', None)
        self.facZ = {'Z': facZ, 'Y': facY, 'D': facD, 'K': fedK}
        self.status = out

    def solve_riccati(self, engine=None):
        """
        Solve the generalized Riccati equation.

        The solution depends on the method and backend being used:
            - Provides the factor Z of the solution X = Z * Z^T when using method = 'nm'.
            - Provides the factor Z of the solution X = Z * inv(Y) * Z^T when using method = 'radi' with param.radi.compute_sol_fac = true and not only initial K0.
            - Provides the factor Z of the solution X = Z * D * Z^T when using method = 'radi' with param.LDL_T = true and not only initial K0..

        Parameters
        ----------
        engine : matlab.engine.MatlabEngine, optional
            MATLAB engine instance to solve the Riccati equation. Required if backend is 'matlab'.
            If None, a new MATLAB engine is started. Default is None.

        Notes
        -----
        - For 'nm' method with 'python' backend, uses PyM.E.S.S. to solve the Riccati equation.
        - For 'nm' or 'radi' methods with 'matlab' backend, uses M-M.E.S.S. via MATLAB engine.
        """
        if self.param['method'] == 'nm' and self.param['backend'] == 'python':
            self._apply_parameters()
            eqn = mess.EquationGRiccatiDAE2(self.param['mess_options'], self.eqn['M'], self.eqn['A'], self.eqn['G'],
                                            self.eqn['B'], self.eqn['C'], self.param['riccati_solver']['delta'])
            self.facZ, self.status = mess.lrnm(eqn, self.param['mess_options'])
        elif self.param['backend'] == 'matlab':
            if engine is None:
                eng = start_matlab()
            else:
                eng = engine
            self._apply_matlab_solver(eng)
            if engine is None:
                eng.quit()

    def _solution_factor(self):
        """
        Return the solution factor for different solvers.

        Returns
        -------
        facZ : numpy.ndarray
            Solution factor of the Riccati equation.

        Notes
        -----
        - Handles different cases based on the method and parameters used.
        - If the solution is not available, returns None.
        """
        if not isinstance(self.facZ, dict):
            return self.facZ

        facZ = self.facZ['Z']

        if self.param['riccati_solver']['LDL_T'] and self.facZ['D'] is not None:
            facD = self.facZ['D']
        elif self.param['method'] == 'radi' and not self.param['riccati_solver']['radi']['get_ZZt'] and self.facZ['Y'] is not None:
            facD = invert_diag_block_matrix(self.facZ['Y'], maxsize=3000)
        else:
            facD = None
        return facZ if facD is None else facZ @ cholesky_sparse(facD, maxsize=3000)  # np.linalg.cholesky(facD)

    def _squared_h2norm_system(self, MatQ, pid=None, chunk_size=5000):
        """
        Compute the squared H2 norm of the closed-loop system.

        Parameters
        ----------
        MatQ : scipy.sparse matrix
            Square-root/Cholesky factor of the weight matrix used to evaluate the H2 norm.
        pid : int, optional
            Number of parallel processes. Default uses available cores minus one.
        chunk_size : int, optional
            Maximum number of columns to process per iteration. Default is 5000.

        Returns
        -------
        float
            Squared H2 norm.

        Notes
        -----
        - Uses parallel processing to speed up computation.
        - The H2 norm is formulated as 'MatQ @ facZ @ facZ^T @ MatQ^T', where 'MatQ = M^(1/2)'
        - The H2 norm is computed as the Frobenius norm of `MatQ @ facZ`.
        """
        #%% pending
        facZ = self._solution_factor()
        if facZ is None:
            print('No solution factor returned since RADI method is used with only initial K0 provided.')
            return np.nan

        def h2norm_partial(start, iend):
            return np.sum(np.square(MatQ @ facZ[:, start:iend]))

        #%%
        # the number of cpus
        num_cores = multiprocessing.cpu_count()
        # if pid is None then pid = max_num_cpu - 1
        pid = pid or num_cores - 1
        # the number of columns
        n = facZ.shape[1]

        if n <= chunk_size or pid == 1:
            return h2norm_partial(0, n)
        else:
            chunk_alloc = distribute_numbers(n, pid)
            h2norms = np.zeros(pid)  # square of h2 norm from each processing

            # Parallel computation of H2 norm
            def h2norm_parallel(proc):
                # start index for this process
                size = chunk_alloc[proc]
                ind_s = int(np.sum(chunk_alloc[:proc]))
                # computes 5000 columns at the same time over all processes
                iters = int(np.ceil(1.0 * size * pid / chunk_size))  # second division for mempry saving
                # pass if size is zero
                if iters:
                    chunk_alloc_2nd = distribute_numbers(size, iters)
                    sub_h2norm = 0
                    for j in range(iters):
                        # pass if chunk_alloc_2nd[j] is zero
                        if chunk_alloc_2nd[j]:
                            # end indices
                            ind_e = ind_s + chunk_alloc_2nd[j]
                            # h2norm for this part of facZ[:,ind_s:ind_z]
                            sub_h2norm += h2norm_partial(ind_s, ind_e)
                            # start indices
                            ind_s = ind_e
                    h2norms[proc] = sub_h2norm

            # Parallel computation of H2 norm
            jb.Parallel(n_jobs=pid, require='sharedmem')(jb.delayed(h2norm_parallel)(proc) for proc in range(pid))
            return np.sum(h2norms)

    def _squared_h2norm_gain(self, K):
        """
        Compute the contribution of each estimator/regulator on the H2 norm.

        Parameters
        ----------
        K : numpy.ndarray
            Feedback matrix.

        Returns
        -------
        numpy.ndarray
            Diagonal elements representing the contribution of each estimator.

        Notes
        -----
        - Useful for analyzing the impact of estimation errors on control performance.
        - In LQE problem, K is the full-information controller, the result shows propagated estimation error energy
        through the LQR gain matrix, which reflects the impact of estimation errors on the performance of the control
        effort (i.e. in LQG problem). It measures the control sensitivity to estimation errors: higher values suggest
        that the control actions are more sensitive to inaccuracies in the estimated state, thus indicating a greater
        impact of estimation errors on control performance. In simpler terms, it answers: How much does the control
        effort suffer due to errors in state estimation?
        """
        facZ = self._solution_factor()
        if facZ is None:
            print('No solution factor returned since RADI method is used with only initial K0 provided.')
            return np.nan

        if K.shape[1] != facZ.shape[0]:
            K = K.T
        norm_T = K @ facZ

        return np.diag(norm_T @ norm_T.T)

    def _feedback_pencil(self, U, V, mode):
        """
        Assemble the feedback pencil for the system.

        Parameters
        ----------
        U : numpy.ndarray
            Feedback matrix U (n x m3).
        V : numpy.ndarray
            Feedback matrix V (n x m3).
        mode : int
            Mode of assembling feedback pencil.
            - If mode = 0, feedback is accumulated directly to the state matrix.
            - If mode = 1 and backend is 'matlab', feedback is updated to U and V matrices in `self.eqn`.

        Notes
        -----
        - Modifies the system matrices based on the feedback provided.
        - Supports different modes of applying feedback depending on the backend.
        - Matrix A can have the form A = Ã + U * V^T if U (eqn.U) and V (eqn.V) are provided U and V are dense (n x m3) matrices and should satisfy m3 << n.
        - In LQE (e.g. sol_type = 'N') problem, V could always be the measurement matrix C, so U can be only added directly.
        - In LQR (e.g. sol_type = 'T') problem, U could always be the actuation matrix B, so V can be only added directly.
        - Otherwise, U, V stacked, and dimensions increase

        Raises
        ------
        ValueError
            If an invalid mode or backend is provided.
        """
        # get solver type
        backend = self.param['backend']
        if backend == 'python':
            sol_type = ['N', 'T'][self.param['riccati_solver']['type']]
        elif backend == 'matlab':
            sol_type = self.param['riccati_solver']['eqn']['type']
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        # Modify U and V based on the solve type for negative feedback
        if sol_type == 'N':
            U = -U
        elif sol_type == 'T':
            V = -V
        else:
            raise ValueError('Unrecognized solve type {}'.format(sol_type))

        if mode == 0:
            # Accumulate feedback directly to the state matrix A
            self.eqn['A'] += sp.csr_matrix(U) @ sp.csr_matrix(V.T)
            self.eqn['A'].eliminate_zeros()
            self.eqn['A'].sort_indices()  # sort data indices, prevent unmfpack error -8
        elif mode == 1 and self.param['backend'] == 'matlab':
            # Update feedback matrices U and V in MATLAB.
            self.param['riccati_solver']['eqn'].update({'haveUV': True, 'sizeUV1': 'default'})
            if 'U' in self.eqn and 'V' in self.eqn:
                # update existing U and V in the equation.
                mark = {'N': ['V', 'U'], 'T': ['U', 'V']}[sol_type]
                dict_uv = {'V': V, 'U': U}
                if dict_uv[mark[0]].shape[1] < self.eqn[mark[0]].shape[1] and np.array_equal(dict_uv[mark[0]], self.eqn[mark[0]][:, -dict_uv[mark[0]].shape[1]:]):
                    # if U or V exists, only add V or U
                    zeros = np.zeros((dict_uv[mark[1]].shape[0], self.eqn[mark[1]].shape[1] - dict_uv[mark[1]].shape[1]))
                    UV_update = np.hstack((zeros, dict_uv[mark[1]])) + self.eqn[mark[1]]
                    self.eqn.update({mark[1]: UV_update})
                else:  # otherwise U, V stack, and dimensions increase
                    self.eqn.update({
                                        'U': np.hstack((self.eqn['U'], U)),
                                        'V': np.hstack((self.eqn['V'], V))
                                    })
            else:
                # First initialization
                self.eqn.update({'U': U, 'V': V})
        else:
            raise ValueError(f'Invalid mode for feedback accumulation: {mode}')