#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:13:08 2024

@author: bojin

Generalized Riccati Equation Solver

This module provides the GRiccatiDAE2Solver class for solving generalized Riccati equations
for index-2 systems using the M.E.S.S. library.
"""

from ..Deps import *

from ..OptimControl.SystemModel import StateSpaceDAE2
from ..LinAlg.Utils import distribute_numbers, deep_set_attr, assemble_dae2, cholesky_sparse, invert_diag_block_matrix
from ..Params.Params import DefaultParameters
from ..Interface.Py2Mat import python2matlab, matlab2python, start_matlab


class GRiccatiDAE2Solver:
    """
    Solve a generalized Riccati equation for an Index-2 system and computes various performance metrics
    like the H2 norm.
    Supports both Python (Py M.E.S.S.) and MATLAB (M M.E.S.S.) backends.
    """

    def __init__(self, model, method='nm', backend='python'):
        """
        Initialize the Riccati solver with a state-space model.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model or dictionary containing system matrices.
        method : str, optional
            Method used to solve generalized Riccati equations. Default is 'nm'.
        backend : str, optional
            Backend used to solve generalized Riccati equations. Default is 'python'.
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
        Assign the state-space model to self.eqn.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model.
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
        formulate system matrices while using Matlab backend.
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
        solve Riccati Equation using Matlab backend.
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

            Provides the factor Z of the solution X = Z * Z^T when using method = 'nm'.

            Provides the factor Z of the solution X = Z * inv(Y) * Z^T when using method = 'radi' with param.radi.compute_sol_fac = true and not only initial K0.

            Provides the factor Z of the solution X = Z * D * Z^T when using method = 'radi' with param.LDL_T = true.
        Parameters
        ----------
        engine : object from matlab.engine.start_matlab()
            matlab engine to solve the Riccati equation.
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
        return solution factor for different solver.

        pending for further development.
        """
        if not isinstance(self.facZ, dict):
            return self.facZ

        facZ = self.facZ['Z']

        if self.param['riccati_solver']['LDL_T'] and self.facZ['D'] is not None:
            facD = self.facZ['D']
        elif self.param['method'] == 'radi' and not self.param['riccati_solver']['radi']['get_ZZt'] and self.facZ['Y'] is not None:
            facD = invert_diag_block_matrix(self.facZ['Y'], maxsize=30)
        else:
            facD = None
        return facZ if facD is None else facZ @ cholesky_sparse(facD, maxsize=30)  # np.linalg.cholesky(facD)

    def _squared_h2norm_system(self, MatQ, pid=None, chunk_size=5000):
        """
        Compute the squared H2 norm of the closed-loop system.
        H2-Norm = Q * Z * Z^T * Q^T, where Q = M^(1/2).

        Parameters
        ----------
        MatQ : scipy.sparse matrix
            Factor of the weight matrix used to evaluate the H2 norm.
        pid : int, optional
            Number of parallel processes. Default is None (uses available cores).
        chunk_size : int, optional
            Maximum number of columns to process per iteration. Default is 5000.

        Returns
        -------
        float
            Squared H2 norm.
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
        Compute the contribution of each estimator/regulator on the H2 Norm.
        e.g. in LQE problem, K is the full-information controller, the result shows propagated estimation error energy
        through the LQR gain matrix, which reflects the impact of estimation errors on the performance of the control
        effort (i.e. in LQG problem). It measures the control sensitivity to estimation errors: higher values suggest
        that the control actions are more sensitive to inaccuracies in the estimated state, thus indicating a greater
        impact of estimation errors on control performance. In simpler terms, it answers: How much does the control
        effort suffer due to errors in state estimation?

        Parameters
        ----------
        K : numpy array
            Feedback matrix.

        Returns
        -------
        numpy array
            Diagonal matrix representing the contribution of each estimator.
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
        Assemble the feedback pencil.
        Matrix A can have the form A = Ã + U*V' if U (eqn.U) and V (eqn.V) are provided U and V are dense (n x m3)
        matrices and should satisfy m3 << n

        In LQE (e.g. sol_type = 'N') problem, V could always be the measurement matrix C, so U can be only added directly
        In LQR (e.g. sol_type = 'T') problem, U could always be the actuation matrix B, so V can be only added directly
        Otherwise, U, V stacked, and dimensions increase

        Parameters
        ----------
        U : dense (n x m3) matrix
            for feedback pencil.
        V : dense (n x m3) matrix
            for feedback pencil.
        mode : int
            Mode of assembling feedback pencil. Default is 0.
            - If mode = 0, feedback is accumulated directly to the state matrix.
            - If mode = 1 and backend = 'matlab', feedback is updated to U and V matrices in self.eqn.

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