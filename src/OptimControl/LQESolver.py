#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `LQESolver` class for solving the Linear Quadratic Estimation (LQE) problem for index-2 systems,
utilizing the M.E.S.S. (Matrix Equation Sparse Solver) library for solving generalized Riccati equations.

The LQE problem involves designing an optimal estimator (observer) for a given system, minimizing the estimation error covariance.
The `LQESolver` class extends the `GRiccatiDAE2Solver` to solve the estimator Riccati equation and compute the Kalman filter gain.

Classes
-------
- LQESolver:
    Solves the LQE problem for index-2 systems and computes the Kalman filter gain.

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
Typical usage involves creating an instance of `LQESolver` with a state-space model, setting sensor noise and disturbance parameters,
solving the LQE problem, and computing the estimator gain.

```python
from FERePack.OptimControl.LQESolver import LQESolver
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

# Initialize the LQE solver with the model
lqe_solver = LQESolver(model, method='nm', backend='python')

# Set sensor noise parameters
lqe_solver.sensor_noise(alpha=1.0)

# Set disturbance parameters
disturbance_matrix = ...  # Define your disturbance matrix
lqe_solver.disturbance(B=disturbance_matrix, beta=0.1)

# Solve the LQE problem
lqe_solver.solve()

# Compute the estimator (Kalman filter) gain
estimator_gain = lqe_solver.estimator()
```

Notes
--------

- The LQESolver class solves the estimator Riccati equation to compute the optimal observer gain.
- It supports iterative solving using accumulation methods for large-scale systems.
- The class can interface with both Python and MATLAB versions of the M.E.S.S. library.
- Sensor noise and disturbance parameters can be configured to model realistic measurement and process noises.

"""

from ..Deps import *
from ..OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from ..OptimControl.BernoulliSolver import BernoulliFeedback
from ..LinAlg.Utils import distribute_numbers
from ..Params.Params import DefaultParameters
from ..Interface.Py2Mat import start_matlab


class LQESolver(GRiccatiDAE2Solver):
    """
    Solves the Linear Quadratic Estimation (LQE) problem for index-2 systems, utilizing the M.E.S.S. library for solving generalized Riccati equations.

    The LQE problem aims to design an optimal estimator (observer) that minimizes the estimation error covariance. This class extends the `GRiccatiDAE2Solver` to handle the estimator Riccati equation and compute the Kalman filter gain.

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
        Function to compute the contribution of the estimator on the H2 norm.

    Methods
    -------
    sensor_noise(alpha)
        Set the magnitude of sensor noises modeled as uncorrelated zero-mean Gaussian white noise.
    disturbance(B, beta=None)
        Set the square root of the random disturbance covariance matrix.
    estimator()
        Compute the estimator (Kalman filter) gain from the solution of the LQE problem.
    solve()
        Solve the LQE problem once and return results.
    iter_solve(num_iter=1, MatQ=None, pid=None, Kr=None, mode=0)
        Solve the LQE problem iteratively using accumulation method.

    Notes
    -----
    - The class supports both low-rank and full-rank solutions, depending on the method used.
    - The computed estimator gain can be applied to design observers for state estimation.
    - Supports iterative solving for large-scale systems where resources are limited.
    """
    def __init__(self, model, method='nm', backend='python'):
        """
        Initialize the LQE solver with a state-space model.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            State-space model or dictionary containing system matrices.
        method : str, optional
            Method used to solve generalized Riccati equations ('nm' or 'radi'). Default is 'nm'.
        backend : str, optional
            Backend used to solve generalized Riccati equations ('python' or 'matlab'). Default is 'python'.

        Notes
        -----
        - Initializes default parameters specific to the LQE problem.
        - Sets up sensor noise with default magnitude.
        """
        self._k0 = BernoulliFeedback(model)
        super().__init__(model, method, backend)
        self._lqe_default_parameters()
        self.sensor_noise(alpha=1)  # initialise Key 'C_pen_mat' in model

    def _lqe_default_parameters(self):
        """
        Set default parameters specific to the LQE problem based on the method and backend.

        Notes
        -----
        - Configures the solver type and equation type appropriate for LQE.
        - For LQE, the solver type is set to 'lqe_solver', and the equation type is 'N' (standard).
        - Pending for further optimization of parameters
        """
        backend = self.param['backend']
        self.param['solver_type'] = 'lqe_solver'
        if backend == 'python':
            self.param['riccati_solver']['type'] = 0
        elif backend == 'matlab':
            self.param['riccati_solver']['eqn']['type'] = 'N'
        else:
            raise ValueError('Backend must be either "python" or "matlab".')

    def _feedback_pencil_deprecated(self, U, V, mode):
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
        - It has been deprecated. Use `self._feedback_pencil` inherited from 'GRiccatiDAE2Solver' instead.
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
        U = -U # negative feedback
        if mode == 0:
            self.eqn['A'] += sp.csr_matrix(U) @ sp.csr_matrix(V.T)
            self.eqn['A'].eliminate_zeros()
            self.eqn['A'].sort_indices()  # sort data indices, prevent unmfpack error -8
        elif mode == 1 and self.param['backend'] == 'matlab':
            self.param['riccati_solver']['eqn'].update({'haveUV': True, 'sizeUV1': 'default'})
            if 'U' in self.eqn and 'V' in self.eqn:
                # In LQE, V should always be measure matrix C
                if V.shape[1] < self.eqn['V'].shape[1] and np.array_equal(V, self.eqn['V'][:, -V.shape[1]:]):
                    # when measure matrix C has already stacked and keep the same, then U only requires addition
                    zeros = np.zeros((U.shape[0], self.eqn['U'].shape[1] - U.shape[1]))
                    U = np.hstack((zeros, U)) + self.eqn['U']
                    self.eqn.update({'U': U})
                else:  # otherwise U, V stack, and dimensions increase
                    U = np.hstack((self.eqn['U'], U))
                    V = np.hstack((self.eqn['V'], V))
                    self.eqn.update({'U': U, 'V': V})
            else: # 1st initialisation
                self.eqn.update({'U': U, 'V': V})
        else:
            raise ValueError('Invalid mode for feedback accumulation.')

    def init_feedback(self):
        """
        Initialize the feedback using Bernoulli feedback.

        Notes
        -----
        - Computes an initial stabilizing feedback gain using the Bernoulli equation.
        - This initial feedback is used to improve convergence of the Riccati solver.
        """
        method = self.param['method']
        backend = self.param['backend']
        k0 = self._k0.solve(transpose=True)  # (k * n)
        if backend == 'python':
            self.param['riccati_solver'][method]['k0'] = k0
        elif backend == 'matlab':
            self.param['riccati_solver'][method]['K0'] = k0

    def sensor_noise(self, alpha):
        """
        Set the magnitude of sensor noises modeled as uncorrelated zero-mean Gaussian white noise.

        The sensor noise affects the measurement covariance matrix, influencing the estimator design.

        Parameters
        ----------
        alpha : float or array-like
            Magnitude of sensor noise. Can be a scalar or an array specifying noise levels for each sensor.

        Raises
        ------
        TypeError
            If the provided `alpha` is of an invalid type.

        Notes
        -----
        - Increases in sensor noise magnitude typically result in decreased estimator gain.
        - The noise weight matrix corresponds to `Q^{-1/2}` in M.E.S.S. or `V^{-1/2}` in literature.
        """
        n = self.eqn['C'].shape[0]
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            mat = 1.0 / alpha * np.identity(n)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(alpha[:n]))
        else:
            raise TypeError('Invalid type for sensor noise.')

        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['C_pen_mat'] = mat
            self.eqn['C'] = mat @ self.eqn['C']
        else:
            self.eqn['Q'] = np.linalg.inv(mat.T @ mat)

    def disturbance(self, B, beta=None):
        """
        Set the square root of the random disturbance covariance matrix.

        Parameters
        ----------
        B : numpy.ndarray
            Disturbance (input) matrix to set. Should have shape compatible with the system.
        beta : float or array-like, optional
            Magnitude of disturbances. Can be a scalar or an array specifying disturbance levels. Default is None.

        Raises
        ------
        ValueError
            If the provided disturbance matrix `B` has an invalid shape.
        TypeError
            If the provided `beta` is of an invalid type.

        Notes
        -----
        - The disturbance affects the process noise covariance, influencing the estimator design.
        - Larger disturbances typically result in increased estimator gain to compensate for process noise.
        - The square root of disturbance covariance matrix corresponds to `R^{-1/2}` in M.E.S.S..
        """
        if B.shape[0] != self.eqn['B'].shape[0]:
            raise ValueError('Invalid shape of input matrix B')

        self.eqn['B'] = B
        n = self.eqn['B'].shape[1]
        if beta is None:
            mat = np.identity(n)
        elif isinstance(beta, (int, np.integer, float, np.floating)):
            mat = beta * np.identity(n)
        elif isinstance(beta, (list, tuple, np.ndarray)):
            mat = np.diag(beta[0:n])
        else:
            raise TypeError('Invalid type for input scale.')

        self.eqn['R'] = mat @ mat.T  # renew due to iterative solve and for consistency
        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['B_pen_mat'] = mat
            self.eqn['B'] = self.eqn['B'] @ mat

    def estimator(self):
        """
        Compute the estimator (Kalman filter) gain from the solution of the LQE problem.

        The estimator gain is computed as:
        ```
        Kf = X * C' * Q^{-1} = Z * Z' * C' * Q^{-1}
        K = (E * Kf)', where K is weighted gain returned by self.iter_solver and M.E.S.S solver
        ```

        Returns
        -------
        Kf : numpy.ndarray
            Kalman filter gain matrix (H2 estimator).

        Notes
        -----
        - The computed gain can be used to design an observer for state estimation.
        - Handles different formulations based on backend and solver options.
        """
        facZ = self._solution_factor()
        if facZ is None:
            print('No solution factor returned since RADI method is used with only initial K0 provided.')
            return np.nan
        # may have issue for Q, R and LDL^T formulation/options
        if self.param['backend'] == 'matlab' and self.param['riccati_solver']['LDL_T']:
            Kf = facZ @ (facZ.T @ self.eqn['C'].T @ np.linalg.inv(self.eqn['Q']))
        else:
            Kf = facZ @ (facZ.T @ self.eqn['C'].T @ self.eqn['C_pen_mat'])
        return Kf

    def solve(self):
        """
        Solve the LQE problem once and return results.

        Returns
        -------
        None

        Notes
        -----
        - Calls `iter_solve` with `num_iter=1`.
        - Suitable for systems where the problem can be solved in a single iteration.
        """

        return self.iter_solve(num_iter=1)

    def iter_solve(self, num_iter=1, MatQ=None, pid=None, Kr=None, mode=0):
        """
        Solve the LQE problem iteratively using accumulation method.

        Parameters
        ----------
        num_iter : int, optional
            Number of iterations to perform. Default is 1.
        MatQ : scipy.sparse matrix, optional
            Square-root/Cholesky factor of the weight matrix used to evaluate the H2 norm. Default is None.
        pid : int, optional
            Number of processes to use for H2 norm computation. Default is None.
        Kr : numpy.ndarray, optional
            Linear Quadratic Regulator (LQR) gain matrix. Default is None.
        mode : int, optional
            Method to perform accumulation method. Default is 0 (i.e., direct assemble).
            If `mode=1` and `backend='matlab'`, utilizes M-M.E.S.S. formulation A = Ã + U*V'.

        Returns
        -------
        output : dict
            Dictionary containing results with keys:
            - 'K': numpy.ndarray, Weighted H2 estimator (Kalman filter) gain.
            - 'status': list, Status of each iteration.
            - 'size': list, Size of solution factor at each iteration.
            - 'alloc_mode': list, Number of disturbance modes allocated to each iteration.
            - 'sqnorm_sys': float, Squared H2 norm of the system.
            - 'sqnorm_lqr': numpy.ndarray, LQR norm, if computed.

        Notes
        -----
        - For large-scale systems, iterative solving can help manage computational resources.
        - Accumulates feedback and performance metrics over iterations.
        - Uses parallel processing for H2 norm computation when `pid` is specified.

        Examples
        --------
        ```python
        # Solve the LQE problem iteratively
        results = lqe_solver.iter_solve(num_iter=5, MatQ=MatQ, pid=4)

        # Access the estimator gain
        estimator_gain = results['K']

        # Access the squared H2 norm
        h2_norm_squared = results['sqnorm_sys']
        ```
        """
        # get original matrices
        B = self.eqn['B'].copy()
        # allocate the number of disturbance modes
        alloc = distribute_numbers(B.shape[1], num_iter)
        if num_iter > 1:
            eng = start_matlab()
        else:
            eng = None
        # initialise results
        status = []
        Zshape = []
        K = np.zeros(self.eqn['C'].T.shape)
        sqnorm_sys = None
        sqnorm_lqr = None

        for i in range(num_iter):
            # allocate disturbance modes
            ind_s = int(np.sum(alloc[:i]))
            ind_e = ind_s + alloc[i]
            self.disturbance(B[:, ind_s:ind_e])
            # now self.eqn['B'] = B[:, ind_s:ind_e]
            self.solve_riccati(engine=eng)

            if num_iter > 1:
                # feedback accumulation
                Kf_sub = self.estimator()
                if Kf_sub is not np.nan:
                    U = self.eqn['M'] @ Kf_sub
                elif self.param['backend'] == 'matlab':
                    U = self.facZ['K'].T
                K += U
                self._feedback_pencil(U, self.eqn['C'].T, mode)
                # get status and size of solution
                status.append(self.status)
                if self.param['backend'] == 'python':
                    Zshape.append(self.facZ.shape)
                elif self.param['backend'] == 'matlab':
                    Zshape.append(np.shape(self.facZ['Z']))
                # accumulate squared norms
                if MatQ is not None:
                    sqnorm_sys = self.sys_energy(MatQ, pid) if i == 0 else sqnorm_sys + self.sys_energy(MatQ, pid)
                if Kr is not None:
                    sqnorm_lqr = self.gain_energy(Kr) if i == 0 else sqnorm_lqr + self.gain_energy(Kr)
                # delete results to save memory
                del self.facZ
                gc.collect()
            else:
                return None

        eng.quit()
        output = {'K': K, 'size': Zshape, 'status': status, 'alloc_mode': alloc, 'sqnorm_sys': sqnorm_sys,
                  'sqnorm_lqr': sqnorm_lqr}
        return output


