#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `FrequencyResponse` class for computing the frequency response of linearized Navier-Stokes systems.

The class extends `FrequencySolverBase` and is designed to solve for the system's response to given inputs at
specified frequencies, facilitating analysis such as transfer function computation, control system design,
and stability assessment in the frequency domain.

Classes
-------
- FrequencyResponse:
    Solves the frequency response of a linearized Navier-Stokes system with specified input and output vectors.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves creating an instance of `FrequencyResponse`, setting up the problem domain, boundary conditions, base flow, and solving for the frequency response at specified frequencies.

```python
from FERePack.FreqAnalys.FreqSolver import FrequencyResponse
import numpy as np

# Define mesh and parameters
mesh = ...  # Define your mesh
Re = 100.0
frequency = 1.0  # Frequency in Hz
omega = 2 * np.pi * frequency  # Angular frequency

# Initialize the frequency response solver
solver = FrequencyResponse(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set base flow
solver.set_baseflow(ic=base_flow_function)

# Define input and output vectors
input_vec = ...  # Define your input vector
output_vec = ...  # Define your output vector

# Solve for the frequency response at the specified frequency
s = 1j * omega
solver.solve(s=s, input_vec=input_vec, output_vec=output_vec)

# Access the computed gain (frequency response)
gain = solver.gain
```

Notes
--------
- The FrequencyResponse class relies on an LU decomposition or Krylov subspace methods for solving the linear system.
- The input and output vectors represent how the system is actuated and measured, respectively.
- The solver can handle feedback matrices, spatial frequency parameters for quasi-analysis, and can reuse previous computations for efficiency.
"""

from ..Deps import *

from ..FreqAnalys.FreqSolverBase import FreqencySolverBase
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, ConvertVector, ConvertMatrix, InverseMatrixOperator
from ..LinAlg.Utils import allclose_spmat, get_subspace_info, find_subspace_index


#%%
class FrequencyResponse(FreqencySolverBase):
    """
    Solves the frequency response of a linearized Navier-Stokes system with given input and output vectors.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The computational mesh of the flow domain.
    Re : float, optional
        The Reynolds number of the flow. Default is None.
    order : tuple of int, optional
        The polynomial orders of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
    dim : int, optional
        Dimension of the flow field. Default is 2.
    constrained_domain : dolfin.SubDomain, optional
        A constrained domain for applying periodic boundary conditions or other constraints. Default is None.

    Attributes
    ----------
    gain : numpy.ndarray
        The computed frequency response (gain) of the system.
    state : numpy.ndarray
        The state vector obtained from solving the linear system.

    Methods
    -------
    solve(s=None, input_vec=None, output_vec=None, Re=None, Mat=None, sz=None, reuse=False)
        Solve for the frequency response of the linearized Navier-Stokes system.

    Notes
    -----
    - The class supports both LU decomposition and Krylov subspace methods for solving the linear system.
    - Input and output vectors should be provided as 1-D numpy arrays.
    - The solver can handle feedback matrices and spatial frequencies for quasi-analysis.
    """

    def __init__(self, mesh, Re=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the FrequencyResponse solver.

        Parameters
        ----------
        mesh : dolfin.Mesh
            The computational mesh of the flow domain.
        Re : float, optional
            The Reynolds number of the flow. Default is None.
        order : tuple of int, optional
            The polynomial orders of the finite element spaces for velocity and pressure, respectively. Default is (2, 1).
        dim : int, optional
            Dimension of the flow field. Default is 2.
        constrained_domain : dolfin.SubDomain, optional
            A constrained domain for applying periodic boundary conditions or other constraints. Default is None.
        """
        super().__init__(mesh, Re, order, dim, constrained_domain)
        self.param = self.param['frequency_response']

    def _initialize_solver(self):
        """
        Initialize the solver for the frequency response computation.

        This method sets up the inverse matrix operator using LU decomposition or Krylov methods (pending), based on the solver parameters.
        """
        param = self.param[self.param['solver_type']]
        self.LHS = self.pencil[0] + self.pencil[1].multiply(1j)
        Mat = self.param['feedback_pencil']
        if self.element.dim > self.element.mesh.topology().dim():  #quasi-analysis
            self.LHS += self.pencil[2].multiply(1j)

        if param['method'] == 'lu':
            info(f"LU decomposition using {param['lusolver'].upper()} solver...")
            self.Linv = InverseMatrixOperator(self.LHS, Mat=Mat, lusolver=param['lusolver'], echo=param['echo'])
            info('Done.')

        elif param['method'] == 'krylov':
            #self.Minv=precondition_jacobi(L, useUmfpack=useUmfpack)
            raise NotImplementedError("The solve method has not yet been implemented.")
            # Krylov solver implementation pending

    def _check_vec(self, b):
        """
        Validate and flatten the input or output vector.

        Parameters
        ----------
        b : numpy.ndarray
            The vector to be validated. Should be a 1-D numpy array.

        Raises
        ------
        ValueError
            If the input is not a 1-D array.

        Returns
        -------
        b : numpy.ndarray
            Flattened 1-D array suitable for computations.
        """

        if np.size(b.shape) == 1 or np.min(b.shape) == 1:
            b = np.asarray(b).flatten()
        else:
            raise ValueError('Please give 1D input/output array')

        return b

    def _solve_linear_system(self, input_vec, output_vec=None):
        """
        Solve the linearized system for the given input and compute the output.

        Parameters
        ----------
        input_vec : numpy.ndarray
            Input vector for the system (actuation).
        output_vec : numpy.ndarray, optional
            Output vector for the system (measurement). If None, returns the state vector.

        Returns
        -------
        gain : numpy.ndarray
            The frequency response (gain) computed from the input and output vectors.

        Notes
        -----
        - If `output_vec` is None, the method returns the state vector (flow field response).
        - Otherwise, it computes the gain as the inner product of the output vector and the state vector.
        """
        iv = self._check_vec(input_vec)
        self.state = self.Linv.matvec(iv)

        if output_vec is None:
            return self.state.reshape(-1, 1)  # Reshape to column vector
        else:
            ov = self._check_vec(output_vec).reshape(1, -1)  # Reshape to row vector
            return ov @ self.state  # Matrix multiplication (equivalent to * and np.matrix)

    def solve(self, s=None, input_vec=None, output_vec=None, Re=None, Mat=None, sz=None, reuse=False):
        """
        Solve for the frequency response of the linearized Navier-Stokes system.

        Parameters
        ----------
        s : complex, optional
            The Laplace variable `s`, typically `s = sigma + i*omega`, where `omega` is the angular frequency.
        input_vec : numpy.ndarray, optional
            1-D input vector representing the actuation. Default is None.
        output_vec : numpy.ndarray, optional
            1-D output vector representing the measurement. Default is None.
        Re : float, optional
            Reynolds number. If provided, updates the Reynolds number before solving. Default is None.
        Mat : scipy.sparse matrix or dict with keys 'U' and 'V', optional
            Feedback matrix `Mat = U * V.T`. Can be provided as a sparse matrix or a dictionary containing 'U' and 'V'. Default is None.
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field. Default is None.
        reuse : bool, optional
            If True, reuses previous computations (e.g., LU factorization) if available. Default is False.

        Returns
        -------
        None

        Notes
        -----
        - The method assembles the system matrices and initializes the solver if `reuse` is False.
        - After solving, the computed gain (frequency response) is stored in `self.gain`.
        - The state vector (flow field response) is stored in `self.state`.
        """
        param = self.param[self.param['solver_type']]

        if Re is not None:
            self.eqn.Re = Re

        if not reuse:
            self._form_LNS_equations(s, sz)
            if Mat is None or sp.issparse(Mat):
                self._assemble_pencil(Mat=Mat, symmetry=param['symmetry'], BCpart=param['BCpart'])
            elif isinstance(Mat, dict) and 'U' in Mat and 'V' in Mat:
                self.param['feedback_pencil'] = Mat
                self._assemble_pencil(Mat=None, symmetry=param['symmetry'], BCpart=param['BCpart'])
            else:
                raise ValueError(
                    'Invalid Type of feedback matrix Mat (Can be a sparse matrix or a dict containing U and V).')
            self._initialize_solver()

        self.gain = self._solve_linear_system(input_vec, output_vec)
        gc.collect()
