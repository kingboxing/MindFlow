#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `BernoulliFeedback` class for computing feedback control using a generalized algebraic Bernoulli equation and eigen-decomposition techniques.

The class is designed to compute feedback gains that stabilize a dynamical system by shifting unstable eigenvalues to
the left half-plane. It is particularly useful as an initial feedback for control design in unstable fluid flow
systems modeled by linearized Navier-Stokes equations.

Classes
-------
- BernoulliFeedback:
    Computes feedback control to stabilize a dynamical system by solving a generalized Bernoulli equation.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `BernoulliFeedback` with a state-space model, solving for the feedback gain, and then applying the gain to stabilize the system.

```python
from FERePack.OptimControl.BernoulliSolver import BernoulliFeedback
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

# Initialize the Bernoulli feedback solver with the model
bernoulli_solver = BernoulliFeedback(model)

# Solve for the feedback gain
feedback_gain = bernoulli_solver.solve()

# Validate the eigenvalues of the closed-loop system
eigenvalues, eigenvectors = bernoulli_solver.validate_eigs(k0=feedback_gain, k=5)
```

Notes
--------

- The BernoulliFeedback class solves the generalized algebraic Bernoulli equation to compute feedback gains.
- It uses eigen-decomposition to identify unstable eigenvalues and forms projection bases for the computation.
- The computed feedback can be used to stabilize the system by applying state feedback control.
- The class supports solving both the standard problem (e.g. for regulator design) and the transposed problem (e.g. for estimator design).
"""
from ..Deps import *
from ..LinAlg.Utils import eigen_decompose, sort_complex, assemble_dae2, assemble_sparse
from ..OptimControl.SystemModel import StateSpaceDAE2
from ..Params.Params import DefaultParameters


class BernoulliFeedback:
    """
    Computes feedback control to stabilize a dynamical system by solving a generalized Bernoulli equation.

    Parameters
    ----------
    model : dict or StateSpaceDAE2
        The state-space model of the system. Can be provided as a dictionary containing the system matrices or as an instance of `StateSpaceDAE2`.

    Attributes
    ----------
    model : dict
        The state-space model containing system matrices and other relevant data.
    param : dict
        Default parameters for the Bernoulli feedback solver.

    Methods
    -------
    solve(transpose=False)
        Solve the Bernoulli equation and compute the feedback vector to stabilize the system.
    validate_eigs(k0, k=3, sigma=0.0, param=None, transpose=False)
        Perform eigen-decomposition on the state-space model with feedback applied to validate stability.

    Notes
    -----
    - The class can solve both the standard problem (e.g. for regulator design) and the transposed problem (e.g. for estimator design).
    - It uses projection methods to reduce the size of the problem and make computations tractable.
    - The feedback gain can be applied to the system to achieve stabilization.
    """

    def __init__(self, model):
        """
        Initialize the BernoulliFeedback solver with the state-space model.

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            The state-space model of the system.

        Raises
        ------
        TypeError
            If the input `model` is not a valid state-space model.
        """
        self._assign_model(model)
        self.param = DefaultParameters().parameters['bernoulli_feedback']

    def _assign_model(self, model):
        """
        Assign the state-space model to the solver.

        This method processes the provided model, ensuring that it contains the necessary matrices for the Bernoulli feedback computation.

        Mass = E_full = | M   0 |      State = A_full = | A   G  |
                        | 0   0 |                       | GT  Z=0|

        Parameters
        ----------
        model : dict or StateSpaceDAE2
            The state-space model.

        Raises
        ------
        TypeError
            If the input is not a valid state-space model.

        Notes
        -----
        - If the model is an instance of `StateSpaceDAE2`, it extracts the internal model dictionary.
        - If the full system matrices `A_full` and `E_full` are unavailable, assemble them from model
        """
        if isinstance(model, StateSpaceDAE2):
            model = model.model

        if isinstance(model, dict):
            self.model = model
            if 'E_full' not in model or 'A_full' not in model:
                self.A_full, self.E_full = assemble_dae2(self.model)
            else:
                self.E_full = self.model['E_full']
                self.A_full = self.model['A_full']
        else:
            raise TypeError('Invalid type for state-space model.')

    def _real_projection_basis(self, vals, vecs):
        """
        Form a real-valued projection basis from complex eigenvalues and eigenvectors.

        This method constructs a real-valued basis for projection by combining the real and imaginary parts of the eigenvalues and eigenvectors, ensuring uniqueness.

        Parameters
        ----------
        vals : numpy.ndarray
            Array of complex eigenvalues.
        vecs : numpy.ndarray
            Array of complex eigenvectors.

        Returns
        -------
        Basis : numpy.ndarray
            Real-valued projection basis.

        Notes
        -----
        - The method concatenates the real and absolute imaginary parts of the eigenvalues and eigenvectors.
        - It ensures that the basis vectors are unique and real-valued.
        """

        arr = np.concatenate((np.real(vals), np.abs(np.imag(vals))))
        mat = np.concatenate((np.real(vecs), np.imag(vecs)), axis=1)
        _, indices_unique = np.unique(arr, return_index=True)

        Basis = mat[:, indices_unique]
        return Basis

    def _bi_eigen_decompose(self):
        """
        Compute right and left eigenvectors for the unstable eigenvalues and form projection bases.

        This method performs eigen-decomposition on the system and its transpose to obtain the right and left
        eigenvectors corresponding to unstable eigenvalues (eigenvalues with positive real parts).

        Returns
        -------
        HL : numpy.ndarray
            Real projection basis for the left eigenvectors.
        HR : numpy.ndarray
            Real projection basis for the right eigenvectors.

        Raises
        ------
        ValueError
            If the left and right eigenvalues are not equal within a tolerance.

        Notes
        -----
        - The method uses parameters from `self.param['bernoulli_feedback']` to determine the number of eigenvalues to compute.
        - Eigenvalues are sorted and compared to ensure consistency between left and right eigenvalues.
        - Projection bases are formed only from the unstable eigenvalues.
        """
        param = self.param['bernoulli_feedback']
        vals_r, vecs_r = eigen_decompose(self.A_full, M=self.E_full, k=param['k'], sigma=param['sigma'],
                                         solver_params=param)
        vals_l, vecs_l = eigen_decompose(self.A_full.transpose(), M=self.E_full.transpose(), k=param['k'],
                                         sigma=param['sigma'], solver_params=param)

        # Sort eigenvalues and eigenvectors
        vals_rs, ind_r = sort_complex(vals_r)
        vals_ls, ind_l = sort_complex(vals_l)
        vecs_rs = vecs_r[:, ind_r]
        vecs_ls = vecs_l[:, ind_l]

        if np.sum(np.abs(vals_rs - vals_ls)) > 1e-10:
            raise ValueError('Left and right eigenvalues are not equal.')
        # Count the number of elements where the real part is greater than 0
        count = np.sum(np.real(vals_rs) > 0)
        info(f'Note: {count} unstable eigenvalues and {vals_r.size - count} stable eigenvalues are computed.')

        # Form projection basis for unstable eigenvalues
        ind_us = np.real(vals_rs) > 0  # indices of unstable eigenvalues
        HL = self._real_projection_basis(vals_ls[ind_us], vecs_ls[:, ind_us])  # left
        HR = self._real_projection_basis(vals_rs[ind_us], vecs_rs[:, ind_us])  # right

        return HL, HR

    def _bernoulli_solver(self, transpose):
        """
        Solve the generalized algebraic Bernoulli equation and form the feedback vector.

        Parameters
        ----------
        transpose : bool, optional
            If True, solve the transposed system (useful for estimator/observer design). Default is False.

        Returns
        -------
        feedback : numpy.ndarray
            Feedback vector of shape (n, k), where `n` is the number of states and `k` is the number of inputs or outputs.

        Notes
        -----
        - The method computes reduced-order system matrices using projection.
        - Solves the continuous-time algebraic Riccati equation (ARE) to obtain the feedback gain.
        - Constructs the full feedback vector by projecting back to the current state space.
        """
        if not transpose:
            B = self.model['B']
        else:
            B = self.model['C'].T

        HL, HR = self._bi_eigen_decompose()

        # Compute reduced system matrices
        M_tilda = HL.T.conj() @ self.E_full @ HR
        A_tilda = HL.T.conj() @ self.A_full @ HR
        B_tilda = HL.T.conj() @ np.pad(B, ((0, self.E_full.shape[0] - B.shape[0]), (0, 0)), 'constant',
                                       constant_values=0)

        if transpose:  # LQE problem
            M_tilda = M_tilda.T
            A_tilda = A_tilda.T
            B_tilda = HR.T.conj() @ np.pad(B, ((0, self.E_full.shape[0] - B.shape[0]), (0, 0)), 'constant',
                                           constant_values=0)

        Xare = sla.solve_continuous_are(A_tilda, B_tilda, np.zeros_like(A_tilda), np.identity(B_tilda.shape[1]),
                                        e=M_tilda, s=None, balanced=True)

        if transpose:  # LQE problem
            return self.E_full.T @ HR @ Xare @ B_tilda

        return self.E_full @ HL @ Xare @ B_tilda

    def solve(self, transpose=False):
        """
        Solve the Bernoulli equation and compute the feedback vector to stabilize the system.

        Parameters
        ----------
        transpose : bool, optional
            If True, solve the transposed problem (useful for estimator/observer design). Default is False.

        Returns
        -------
        feedback : numpy.ndarray
            Feedback vector of shape (k, m), where `k` is the number of inputs or outputs, and `m` is the number of input/output states.

        Notes
        -----
        - The computed feedback can be used to modify the system dynamics and achieve stabilization.
        - If `transpose` is True, the method computes the observer gain instead.

        Examples
        --------
        ```python
        # Solve for the state feedback gain
        feedback_gain = bernoulli_solver.solve()

        # Solve for the observer gain
        observer_gain = bernoulli_solver.solve(transpose=True)
        ```
        """
        feedback = self._bernoulli_solver(transpose)
        info('Bernoulli Feedback Solver Finished.')
        if transpose:
            return feedback[:self.model['C'].shape[1], :].T

        return feedback[:self.model['B'].shape[0], :].T

    def validate_eigs(self, k0, k=3, sigma=0.0, param=None, transpose=False):
        """
        Perform eigen-decomposition on the state-space model with feedback applied to validate stability.

        Parameters
        ----------
        k0 : numpy.ndarray
            Feedback vector computed using `self.solve()`.
        k : int, optional
            Number of eigenvalues to compute. Default is 3.
        sigma : float or complex, optional
            Shift-invert parameter to target eigenvalues near `sigma`. Default is 0.0.
        param : dict, optional
            Additional parameters for the eigenvalue solver. Default is None.
        transpose : bool, optional
            If True, perform validation on the transposed problem. Default is False.

        Returns
        -------
        vals : numpy.ndarray
            Computed eigenvalues of the closed-loop system.
        vecs : numpy.ndarray
            Corresponding eigenvectors.

        Notes
        -----
        - This method applies the computed feedback to the system and computes the eigenvalues of the closed-loop system.
        - It is used to validate whether the unstable eigenvalues have been shifted to the left half-plane.

        Examples
        --------
        ```python
        # Validate eigenvalues after applying state feedback
        eigenvalues, eigenvectors = bernoulli_solver.validate_eigs(k0=feedback_gain, k=5)

        # Validate eigenvalues after applying observer gain
        eigenvalues_obs, eigenvectors_obs = bernoulli_solver.validate_eigs(k0=observer_gain, k=5, transpose=True)
        ```
        """
        if param is None:
            param = {}
        # Set up matrices
        if not transpose:
            M = self.E_full
            A = self.A_full
            BC = self.model['B']
        else:
            M = self.E_full.T
            A = self.A_full.T
            BC = self.model['C'].T

        if k0 is not None:
            # Pad matrices to match dimensions
            B_sp = sp.csr_matrix(np.pad(BC, ((0, M.shape[0] - BC.shape[0]), (0, 0)), 'constant', constant_values=0))
            K_sp = sp.csr_matrix(np.pad(k0, ((0, 0), (0, M.shape[0] - k0.shape[1])), 'constant', constant_values=0))
            B_sp.eliminate_zeros()
            K_sp.eliminate_zeros()
            # Compute the feedback matrix
            Mat = B_sp @ K_sp
            # Compute eigenvalues of the closed-loop system
            # pending for sparse representation A + U * V'
            return eigen_decompose(A - Mat, M, k=k, sigma=sigma, solver_params=param)
        else:
            # Compute eigenvalues of the open-loop system
            return eigen_decompose(A, M, k=k, sigma=sigma, solver_params=param)
