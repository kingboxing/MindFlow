#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `LQRSolver` class for solving the Linear Quadratic Regulator (LQR) problem for index-2 systems,
utilizing the M.E.S.S. (Matrix Equation Sparse Solver) library to solve generalized Riccati equations.

The LQR problem involves designing an optimal state feedback controller that minimizes a quadratic cost function,
typically balancing control effort against state regulation performance. The `LQRSolver` class extends the
`GRiccatiDAE2Solver` to solve the regulator Riccati equation and compute the optimal control gain.

Classes
-------
- LQRSolver:
    Solves the LQR problem for index-2 systems and computes the optimal control gain.

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
Typical usage involves creating an instance of `LQRSolver` with a state-space model, setting control penalty and measurement parameters,
solving the LQR problem, and computing the regulator gain.

```python
from FERePack.OptimControl.LQRSolver import LQRSolver
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

# Initialize the LQR solver with the model
lqr_solver = LQRSolver(model, method='nm', backend='python')

# Set control penalty parameters
lqr_solver.control_penalty(alpha=1.0)

# Set measurement parameters
measurement_matrix = ...  # Define your measurement matrix
lqr_solver.measurement(C=measurement_matrix, beta=0.1)

# Solve the LQR problem
lqr_solver.solve()

# Compute the regulator (optimal control) gain
regulator_gain = lqr_solver.regulator()
```

Notes
--------

- The LQRSolver class solves the regulator Riccati equation to compute the optimal state feedback control gain.
- It supports iterative solving using accumulation methods for large-scale systems.
- The class can interface with both Python and MATLAB versions of the M.E.S.S. library.
- Control penalty and measurement parameters can be configured to balance control effort and performance.

"""

from ..Deps import *
from ..OptimControl.RiccatiSolver import GRiccatiDAE2Solver
from ..OptimControl.BernoulliSolver import BernoulliFeedback
from ..LinAlg.Utils import distribute_numbers
from ..Params.Params import DefaultParameters
from ..Interface.Py2Mat import start_matlab


class LQRSolver(GRiccatiDAE2Solver):
    """
    Solves the Linear Quadratic Regulator (LQR) problem for index-2 systems, utilizing the M.E.S.S. library to solve generalized Riccati equations.

    The LQR problem involves designing an optimal state feedback controller that minimizes a quadratic cost function,
    balancing control effort against state regulation performance. This class extends the `GRiccatiDAE2Solver` to
    handle the regulator Riccati equation and compute the optimal control gain.

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
        Function to compute the contribution of the regulator on the H2 norm.

    Methods
    -------
    control_penalty(alpha)
        Set the magnitude of control penalty in the LQR formulation.
    measurement(C, beta=None)
        Set the square root of the response weight matrix for the LQR formulation.
    regulator()
        Compute the optimal control gain from the solution of the LQR problem.
    solve()
        Solve the LQR problem once and return results.
    iter_solve(num_iter=1, MatQ=None, pid=None, Kf=None, mode=0)
        Solve the LQR problem iteratively using accumulation method.

    Notes
    -----
    - The class supports both low-rank and full-rank solutions, depending on the method used.
    - The computed regulator gain can be applied to design controllers for state regulation.
    - Supports iterative solving for large-scale systems where resources are limited.
    """
    def __init__(self, model, method='nm', backend='python'):
        """
        Initialize the LQR solver with a state-space model.

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
        - Initializes default parameters specific to the LQR problem.
        - Sets up control penalty with default magnitude.
        """
        self._k0 = BernoulliFeedback(model)
        super().__init__(model, method, backend)
        self._lqr_default_parameters()
        self.control_penalty(alpha=1)  # initialise Key 'B_pen_mat' in model

    def _lqr_default_parameters(self):
        """
        Set default parameters specific to the LQR problem based on the method and backend.

        Notes
        -----
        - Configures the solver type and equation type appropriate for LQR.
        - For LQR, the solver type is set to 'lqr_solver', and the equation type is 'T' (transposed).
        - Pending for further optimization of parameters
        """
        backend = self.param['backend']
        self.param['solver_type'] = 'lqr_solver'
        if backend == 'python':
            self.param['riccati_solver']['type'] = 1
        elif backend == 'matlab':
            self.param['riccati_solver']['eqn']['type'] = 'T'
        else:
            raise ValueError('Backend must be either "python" or "matlab"')

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
        V = -V  # negative feedback
        if mode == 0:
            self.eqn['A'] += sp.csr_matrix(U) @ sp.csr_matrix(V.T)
            self.eqn['A'].eliminate_zeros()
            self.eqn['A'].sort_indices()  # sort data indices, prevent unmfpack error -8
        elif mode == 1 and self.param['backend'] == 'matlab':
            self.param['riccati_solver']['eqn'].update({'haveUV': True, 'sizeUV1': 'default'})
            if 'U' in self.eqn and 'V' in self.eqn:
                # In LQR, U should always be actuation matrix B
                if U.shape[1] < self.eqn['U'].shape[1] and np.array_equal(U, self.eqn['U'][:, -U.shape[1]:]):
                    # when actuation matrix B has already stacked and keep the same, then V only requires addition
                    zeros = np.zeros((V.shape[0], self.eqn['V'].shape[1] - V.shape[1]))
                    V = np.hstack((zeros, V)) + self.eqn['V']
                    self.eqn.update({'V': V})
                else:  # otherwise U, V stack, and dimensions increase
                    U = np.hstack((self.eqn['U'], U))
                    V = np.hstack((self.eqn['V'], V))
                    self.eqn.update({'U': U, 'V': V})
            else:  # 1st initialisation
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
        k0 = self._k0.solve(transpose=False)  # (k * n)
        if backend == 'python':
            self.param['riccati_solver'][method]['k0'] = k0
        elif backend == 'matlab':
            self.param['riccati_solver'][method]['K0'] = k0

    def control_penalty(self, alpha):
        """
        Set the magnitude of control penalty in the LQR formulation.

        The control penalty affects the weighting of control effort in the cost function.

        Parameters
        ----------
        alpha : float or array-like
            Magnitude of control penalty. Can be a scalar or an array specifying penalties for each control input.

        Raises
        ------
        TypeError
            If the provided `alpha` is of an invalid type.

        Notes
        -----
        - A higher control penalty reduces control effort but may result in poorer state regulation.
        - The control penalty matrix corresponds to `R^{-1/2}` in M.E.S.S..
        """

        n = self.eqn['B'].shape[1]
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            mat = 1.0 / alpha * np.identity(n)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            mat = np.diag(np.reciprocal(alpha[:n]))
        else:
            raise TypeError('Invalid type for control penalty.')

        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['B_pen_mat'] = mat
            self.eqn['B'] = self.eqn['B'] @ mat
        else:
            self.eqn['R'] = np.linalg.inv(mat @ mat.T)

    def measurement(self, C, beta=None):
        """
        Set the square root of the response weight matrix for the LQR formulation.

        The measurement affects the weighting of state deviations in the cost function.

        Parameters
        ----------
        C : numpy.ndarray
            Measurement (output) matrix that represents the response weight.
        beta : float or array-like, optional
            Magnitude of response weight. Can be a scalar or an array specifying weights for each state. Default is None.

        Raises
        ------
        ValueError
            If the shape of the provided `C` does not match the shape of the existing `C` matrix.
        TypeError
            If the provided `beta` is of an invalid type.

        Notes
        -----
        - Increasing the response weight places more emphasis on state regulation in the cost function.
        - The measurement matrix corresponds to `Q^{-1/2}` in M.E.S.S..
        """
        if C.shape[1] != self.eqn['C'].shape[1]:
            raise ValueError('Invalid shape of output matrix C.')

        self.eqn['C'] = C
        n = self.eqn['C'].shape[0]
        if beta is None:
            mat = np.identity(n)
        elif isinstance(beta, (int, np.integer, float, np.floating)):
            mat = beta * np.identity(n)
        elif isinstance(beta, (list, tuple, np.ndarray)):
            mat = np.diag(beta[0:n])
        else:
            raise TypeError('Invalid type for output scale.')

        self.eqn['Q'] = mat.T @ mat  # renew due to iterative solve and for consistency
        if self.param['backend'] == 'python' or not self.param['riccati_solver']['LDL_T']:
            self.eqn['C_pen_mat'] = mat
            self.eqn['C'] = mat @ self.eqn['C']

    def regulator(self):
        """
        Compute the optimal control gain (regulator) from the solution of the LQR problem.

        The regulator gain is computed as:
        ```
        Kr = R^{-1} * B^T * X = R^{-1} * B^T * Z * Z^T
        K = Kr * E, where K is weighted gain returned by self.iter_solver and M.E.S.S solver
        ```

        Returns
        -------
        Kr : numpy.ndarray
            Linear Quadratic Regulator (optimal control) gain matrix.

        Notes
        -----
        - The computed gain can be used to design a state feedback controller.
        - Handles different formulations based on backend and solver options.
        """
        facZ = self._solution_factor()
        if facZ is None:
            print('No solution factor returned since RADI method is used with only initial K0 provided.')
            return np.nan

        # may have issue for Q, R and LDL^T formulation/options
        if self.param['backend'] == 'matlab' and self.param['riccati_solver']['LDL_T']:
            Kr = np.linalg.inv(self.eqn['R']) @ self.eqn['B'].T @ facZ @ facZ.T
        else:
            Kr = self.eqn['B_pen_mat'] @ self.eqn['B'].T @ facZ @ facZ.T
        return Kr

    def solve(self):
        """
        Solve the LQR problem once and return results.

        Returns
        -------
        None

        Notes
        -----
        - Calls `iter_solve` with `num_iter=1`.
        - Suitable for systems where the problem can be solved in a single iteration.
        """

        return self.iter_solve(num_iter=1)

    def iter_solve(self, num_iter=1, MatQ=None, pid=None, Kf=None, mode=0):
        """
        Solve the LQR problem iteratively using accumulation method.

        Parameters
        ----------
        num_iter : int, optional
            Number of iterations to perform. Default is 1.
        MatQ : scipy.sparse matrix, optional
            Square-root/Cholesky factor of the weight matrix used to evaluate the H2 norm. Default is None.
        pid : int, optional
            Number of processes to use for H2 norm computation. Default is None.
        Kf : numpy.ndarray, optional
            Kalman Filter gain matrix (from LQE). Default is None.
        mode : int, optional
            Method to perform accumulation method. Default is 0 (i.e., direct assemble).
            If `mode=1` and `backend='matlab'`, utilizes M-M.E.S.S. formulation A = Ã + U*V'.

        Returns
        -------
        output : dict
            Dictionary containing results with keys:
            - 'K': numpy.ndarray, Weighted optimal control gain.
            - 'status': list, Status of each iteration.
            - 'size': list, Size of solution factor at each iteration.
            - 'alloc_mode': list, Number of measurement modes allocated to each iteration.
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
        # Solve the LQR problem iteratively
        results = lqr_solver.iter_solve(num_iter=5, MatQ=MatQ, pid=4)

        # Access the regulator gain
        regulator_gain = results['K']

        # Access the squared H2 norm
        h2_norm_squared = results['sqnorm_sys']
        ```
        """

        # get original matrices
        C = self.eqn['C'].copy()
        # allocate the number of disturbance modes
        alloc = distribute_numbers(C.shape[0], num_iter)
        if num_iter > 1:
            eng = start_matlab()
        else:
            eng = None
        # initialise results
        status = []
        Zshape = []
        K = np.zeros(self.eqn['B'].T.shape) # k * n
        sqnorm_sys = None
        sqnorm_lqe = None

        for i in range(num_iter):
            # allocate disturbance modes
            ind_s = int(np.sum(alloc[:i]))
            ind_e = ind_s + alloc[i]
            self.measurement(C[ind_s:ind_e, :])
            # now self.eqn['B'] = B[:, ind_s:ind_e]
            self.solve_riccati(engine=eng)

            if num_iter > 1:
                # feedback accumulation
                Kr_sub = self.regulator()
                if Kr_sub is not np.nan:
                    V = (Kr_sub @ self.eqn['M'])
                elif self.param['backend'] == 'matlab':
                    V = self.facZ['K']
                K += V
                self._feedback_pencil(self.eqn['B'], V.T, mode)
                # get status and size of solution
                status.append(self.status)
                if self.param['backend'] == 'python':
                    Zshape.append(self.facZ.shape)
                elif self.param['backend'] == 'matlab':
                    Zshape.append(np.shape(self.facZ['Z']))
                # accumulate squared norms
                if MatQ is not None:
                    sqnorm_sys = self.sys_energy(MatQ, pid) if i == 0 else sqnorm_sys + self.sys_energy(MatQ, pid)
                if Kf is not None:
                    sqnorm_lqe = self.gain_energy(Kf) if i == 0 else sqnorm_lqe + self.gain_energy(Kf)
                # delete results to save memory
                del self.facZ
                gc.collect()
            else:
                return None

        eng.quit()
        output = {'K': K, 'size': Zshape, 'status': status, 'alloc_mode': alloc, 'sqnorm_sys': sqnorm_sys,
                  'sqnorm_lqr': sqnorm_lqe}
        return output

