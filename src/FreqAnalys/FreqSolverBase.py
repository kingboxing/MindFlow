#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the `FrequencySolverBase` class, which serves as a base class for performing frequency domain
analysis on fluid flow problems using the finite element method with the FEniCS library.

The class extends `NSolverBase` and is designed to handle the linearization of the Navier-Stokes equations around a
base flow, assemble the system matrices (also known as the matrix pencil), and set up the necessary components for
frequency response analysis, stability analysis, and control applications.

Classes
-------
- FrequencySolverBase:
    Base class for frequency domain solvers for linearized Navier-Stokes equations.

Dependencies
------------
- FEniCS
- NumPy
- SciPy

Ensure that all dependencies are installed and properly configured.

Examples
--------
Typical usage involves subclassing `FrequencySolverBase` to implement specific frequency domain solvers for fluid flow problems.

```python
from FERePack.FreqAnalys.FreqSolverBase import FrequencySolverBase

# Define mesh and parameters
mesh = ...
Re = 100.0

# Initialize the frequency solver base
solver = FrequencySolverBase(mesh, Re=Re, order=(2, 1), dim=2)

# Set boundary conditions
solver.set_boundary(bc_list)
solver.set_boundarycondition(bc_list)

# Set base flow
solver.set_baseflow(ic=base_flow_function)

# Form the linearized Navier-Stokes equations at a specific frequency
s = 1j * omega  # Laplace variable for frequency omega
solver._form_LNS_equations(s)

# Assemble the system matrices (matrix pencil)
solver._assemble_pencil()
```

Notes
-------
- This class is intended to be subclassed, with methods implemented to perform specific analyses such as frequency response computation, eigenvalue analysis, or control design.
- The class handles the assembly of complex matrices representing the linearized system, taking into account boundary conditions and possible feedback terms.
"""

from ..Deps import *

from ..NSolver.SolverBase import NSolverBase
from ..BasicFunc.ElementFunc import TaylorHood
from ..BasicFunc.Boundary import SetBoundary, SetBoundaryCondition
from ..BasicFunc.InitialCondition import SetInitialCondition
from ..LinAlg.MatrixOps import AssembleMatrix, AssembleVector, ConvertVector, ConvertMatrix, AssembleSystem, \
    TransposePETScMat, InverseMatrixOperator, SparseLUSolver
from ..Params.Params import DefaultParameters


#%%
class FreqencySolverBase(NSolverBase):
    """
    Base class for frequency domain solvers for linearized Navier-Stokes equations.

    This class sets up the necessary finite element spaces, boundary conditions, and equations for performing frequency domain analysis, such as frequency response and stability analysis, of incompressible fluid flows. It linearizes the Navier-Stokes equations around a given base flow and assembles the system matrices (matrix pencil) required for solving the linearized system in the frequency domain.

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
        A constrained subdomain for applying periodic boundary conditions or other constraints. Default is None.

    Attributes
    ----------
    element : object
        The finite element object defining the function spaces.
    boundary_condition : SetBoundaryCondition
        Object for handling boundary conditions.
    param : dict
        Dictionary of solver parameters.

    Methods
    -------
    set_baseflow(ic, timestamp=0.0)
        Set the base flow around which the Navier-Stokes equations are linearized.
    _form_LNS_equations(s, sz=None)
        Formulate the linearized Navier-Stokes equations in the frequency domain.
    _assemble_pencil(Mat=None, symmetry=False, BCpart=None)
        Assemble the matrix pencil (system matrices) of the linearized system.

    Notes
    -----
    - The class uses Taylor-Hood elements by default for the finite element discretization.
    - Boundary conditions and base flow must be specified before assembling the system matrices.
    """
    def __init__(self, mesh, Re=None, order=(2, 1), dim=2, constrained_domain=None):
        """
        Initialize the FrequencySolverBase.

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

        element = TaylorHood(mesh=mesh, order=order, dim=dim,
                             constrained_domain=constrained_domain)  # initialise finite element space
        super().__init__(mesh, element, Re, None, None)

        # boundary condition
        self.boundary_condition = SetBoundaryCondition(self.element.functionspace, self.boundary)
        # init param
        self.param = DefaultParameters().parameters

    def set_baseflow(self, ic, timestamp=0.0):
        """
        Set the base flow around which the Navier-Stokes equations are linearized.

        Parameters
        ----------
        ic : str or dolfin.Function
            The initial condition representing the base flow. Can be a file path to stored data or a FEniCS function.
        timestamp : float, optional
            The timestamp for retrieving the base flow from a time series file if `ic` is a file path. Default is 0.0.

        Returns
        -------
        None

        Notes
        -----
        - The base flow is essential for linearizing the Navier-Stokes equations and must be set before forming the linearized equations.
        """

        SetInitialCondition(0, ic=ic, fw=self.eqn.fw[0], timestamp=timestamp)

    def _form_LNS_equations(self, s, sz=None):
        """
        Formulate the linearized Navier-Stokes equations in the frequency domain.

        This method assembles the UFL expressions representing the linearized Navier-Stokes equations around the base flow, including frequency-dependent terms.

        Parameters
        ----------
        s : complex
            The Laplace variable `s`, typically `s = sigma + i*omega`, where `omega` is the angular frequency.
        sz : complex or tuple/list of complex, optional
            Spatial frequency parameters for quasi-analysis of the flow field (e.g., for three-dimensional perturbations in a two-dimensional base flow). Default is None.

        Returns
        -------
        None

        Notes
        -----
        - The linearized equations are stored in `self.LNS` as UFL expressions.
        - The method handles both standard linearization and quasi-analysis (if spatial frequencies `sz` are provided).
        """
        if self.element.dim == self.element.mesh.topology().dim():
            # form Steady Linearised Incompressible Navier-Stokes Equations
            leqn = self.eqn.SteadyLinear()
            feqn = self.eqn.Frequency(s)

            for key in self.has_traction_bc.keys():
                leqn += self.BoundaryTraction(self.eqn.tp, self.eqn.tu, self.eqn.nu, mark=self.has_traction_bc[key][0],
                                              mode=self.has_traction_bc[key][1])

            self.LNS = (leqn + feqn[0], feqn[1])  # (real part, imag part)

        elif self.element.dim > self.element.mesh.topology().dim():  # quasi-analysis
            # form quasi-Steady Linearised Incompressible Navier-Stokes Equations
            leqn_r, leqn_i = self.eqn.QuasiSteadyLinear(sz)
            feqn = self.eqn.Frequency(s)

            for key in self.has_traction_bc.keys():
                leqn_r += self.BoundaryTraction(self.eqn.tp, self.eqn.tu, self.eqn.nu,
                                                mark=self.has_traction_bc[key][0], mode=self.has_traction_bc[key][1])

            self.LNS = (leqn_r + feqn[0], feqn[1], leqn_i)

    def _assemble_pencil(self, Mat=None, symmetry=False, BCpart=None):
        """
        Assemble the matrix pencil (system matrices sM+NS) of the linearized system.

        u=(sM+NS)^-1*f where RHS=f, Resp=u, sM = Ai, NS = Ar

        Boundary Condition: u = value/0.0

        The method assembles the system matrices (also known as the matrix pencil) representing the linearized
        Navier-Stokes equations in the frequency domain. It handles the application of boundary conditions and can
        incorporate feedback matrices for control applications.

        Parameters
        ----------
        Mat : scipy.sparse matrix, optional
            Feedback matrix to be added to the system matrix (e.g., for control applications: sM+NS+Mat). Default is None.
        symmetry : bool, optional
            If True, assemble matrices in a symmetric fashion. Default is False.
        BCpart : str, optional
            Specifies where to apply homogeneous boundary conditions:
            - 'r' : Apply to the real part of the system matrix (matrix NS).
            - 'i' : Apply to the imaginary part of the system matrix (matrix M).
            Default is None, which applies to the imaginary part.

        Returns
        -------
        None

        Notes
        -----
        - The assembled matrices are stored in `self.pencil` as a tuple of sparse matrices.
        - The matrices are converted to CSC (Compressed Sparse Column) format for efficient numerical operations.
        - Boundary conditions are incorporated into the matrices by modifying the relevant rows and columns.
        """
        if symmetry:  # for homogeneous bcs, e.g. eigen analysis
            dummy_rhs = inner(Constant((0.0,) * self.eqn.dim), self.eqn.v) * dx
            Ar = AssembleSystem(self.LNS[0], dummy_rhs, self.boundary_condition.bc_list)[0]
            Ai = AssembleSystem(self.LNS[1], dummy_rhs, self.boundary_condition.bc_list)[0]
            I_bc = AssembleSystem(Constant(0.0) * self.LNS[1], dummy_rhs, self.boundary_condition.bc_list)[0]

        else:
            Ar = AssembleMatrix(self.LNS[0], self.boundary_condition.bc_list)  # assemble the real part
            Ai = AssembleMatrix(self.LNS[1], self.boundary_condition.bc_list)  # assemble the imag part
            I_bc = AssembleMatrix(Constant(0.0) * self.LNS[1],
                                  self.boundary_condition.bc_list)  # matrix has only ones in diagonal for rows specified by the boundary condition

        if Mat is not None:  # feedback loop
            Ar += Mat  # Mat need to be BC applied
            temp_mat = ConvertMatrix(Ar, flag='Mat2PETSc')
            [bc.apply(temp_mat) for bc in self.boundary_condition.bc_list]

            if symmetry:
                temp_mat = TransposePETScMat(temp_mat)
                [bc.apply(temp_mat) for bc in self.boundary_condition.bc_list]
                temp_mat = TransposePETScMat(temp_mat)

            Ar = ConvertMatrix(temp_mat, flag='PETSc2Mat')

        if BCpart is None or BCpart.lower() == 'i':
            Ai = Ai - I_bc
        elif BCpart.lower() == 'r':
            Ar = Ar - I_bc
        else:
            raise ValueError("BCpart must be one of ('r','i')")

        # Complex matrix with boundary conditions
        pencil = (Ar.tocsc(), Ai.tocsc())

        if self.element.dim > self.element.mesh.topology().dim():
            if symmetry:
                Ai_quasi = AssembleSystem(self.LNS[2], dummy_rhs, self.boundary_condition.bc_list)[0] - I_bc
            else:
                Ai_quasi = AssembleMatrix(self.LNS[2], self.boundary_condition.bc_list) - I_bc

            pencil += (Ai_quasi.tocsc(),)
        # Convert to CSC format for efficient LU decomposition
        self.pencil = pencil
