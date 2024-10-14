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
from ..LinAlg.Utils import distribute_numbers, dict_deep_update, deep_set_attr, assemble_dae2, assemble_sparse
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
        n = self.eqn['G'].shape[1]
        A_full = assemble_sparse([[self.eqn['A'], self.eqn['G']], [self.eqn['G'].T, None]])
        E_full = assemble_sparse([[self.eqn['M'], None], [None, sp.csr_matrix((n, n))]])
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
            eqn = mess.EquationGRiccatiDAE2(self.param['mess_options'], self.eqn['M'], self.eqn['A'], self.eqn['G'], self.eqn['B'], self.eqn['C'], self.param['riccati_solver']['delta'])
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
        facD = None
        if self.facZ['Y'] is not None:
            facD = np.linalg.inv(self.facZ['Y'])
        elif self.facZ['D'] is not None:
            facD = self.facZ['D']
        return facZ if facD is None else facZ @ np.linalg.cholesky(facD)

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

class GRiccatiDAE2NMSolver_Pymess:
    """
    This class solves a generalized Riccati equation for an Index-2 system and computes
    various performance metrics such as H2 norm.
    """

    def __init__(self, ssmodel):
        """
        Initialize the Riccati solver with a state-space model.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model or dictionary containing the system matrices.

        Description
        ----------
        MESS_DIRECT_DEFAULT_LU 0, the same as UMFPACK
        
        MESS_DIRECT_SPARSE_LU 1, too long time to check
        
        MESS_DIRECT_LAPACKLU 2, 
        
        MESS_DIRECT_UMFPACK_LU: 3, normal
            
        MESS_DIRECT_SUPERLU_LU : 4, check error occured, kernel died
        
        MESS_DIRECT_CSPARSE_LU 5, too long time to check
        
        MESS_DIRECT_BANDED_LU 6,
        
        MESS_DIRECT_MKLPARDISO_LU 7,


        """
        self.Model = ssmodel
        self._apply_default_param()
        self.facZ = None
        self.status = None

    def _apply_default_param(self):
        """
        Set default parameters for the Riccati solver.
        """
        self.param = DefaultParameters().parameters['nmriccati_pymess']
        self.param['mess_options'] = mess.Options()
        param_default = self.param['riccati_solver']
        self.update_riccati_params(param_default)

    def update_riccati_params(self, param):
        """
        Update the Riccati solver parameters.

        Parameters
        ----------
        param : dict
            A dictionary containing solver parameters to update.
        """
        self.param['riccati_solver'] = dict_deep_update(self.param['riccati_solver'], param)
        if 'lusolver' in param:
            mess.direct_select(param.pop('lusolver'))
        deep_set_attr(self.param['mess_options'], param)

    def _assign_model(self, ssmodel):
        """
        Assign the state-space model.

        Parameters
        ----------
        ssmodel : dict or StateSpaceDAE2
            State-space model.

        Raises
        ------
        TypeError
            If the input is not a valid state-space model.
        """

        if isinstance(ssmodel, dict):
            self._M = ssmodel['M']
            self._A = ssmodel['A']
            self._G = ssmodel['G']
            self._B = ssmodel['B']
            self._C = ssmodel['C']
        elif isinstance(self.ssmodel, StateSpaceDAE2):
            self._M = ssmodel.M
            self._A = ssmodel.A
            self._G = ssmodel.G
            self._B = ssmodel.B
            self._C = ssmodel.C
        else:
            raise TypeError('Invalid type for state-space model.')

    def solve_riccati(self, delta=-0.02):
        """
        Solve the generalized Riccati equation.
        Provides the factor Z of the solution X = Z * Z^T.

        Parameters
        ----------
        delta : float, optional
            A real, negative scalar for shift-invert. Default is -0.02.

        Returns
        -------
        status : M.E.S.S. status object
            Status of the Riccati equation solver.
        """

        self._assign_model(self.Model)
        eqn = mess.EquationGRiccatiDAE2(self.param['mess_options'], self._M, self._A, self._G, self._B, self._C, delta)
        self.facZ, self.status = mess.lrnm(eqn, self.param['mess_options'])
        return self.status

    def squared_h2norm(self, MatQ, pid=None, chunk_size=5000):
        """
        Compute the squared H2 norm of the solution.
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
        # the number of cpus
        num_cores = multiprocessing.cpu_count()
        # if pid is None then pid = max_num_cpu - 1
        pid = pid or num_cores - 1
        # the number of columns
        n = self.facZ.shape[1]

        def h2norm_partial(start, end):
            return np.sum(np.square(MatQ @ self.facZ[:, start:end]))

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


    def normvec_T(self, K):
        """
        Compute the contribution of each estimator/regulator on the H2 Norm?

        Parameters
        ----------
        K : numpy array
            Feedback matrix.

        Returns
        -------
        numpy array
            Diagonal matrix representing the contribution of each estimator.
        """

        if K.shape[1] != self.facZ.shape[0]:
            K = K.T
        norm_T = K @ self.facZ

        return np.diag(norm_T @ norm_T.T)
