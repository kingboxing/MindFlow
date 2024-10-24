#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides utility functions for constructing prolongation matrices, mass/weight matrices,
and identity matrices with boundary or subdomain conditions for finite element analysis in FEniCS.

Functions
---------
- IdentMatProl(element, index=None):
    Construct the prolongation matrix that excludes specified subspaces.
- MatWgt(element):
    Assemble the mass (weight) matrix for the entire function space.
- IdentMatBC(element, bcs=[]):
    Construct an identity matrix with zeros at DOFs corresponding to boundary conditions.
- IdentMatSub(element, bound=None):
    Construct an identity matrix with zeros outside a specified subdomain.
- MatP(element):
    Construct a prolongation matrix for resolvent analysis, excluding pressure subspaces.
- MatM(element, bcs=[]):
    Assemble the mass matrix restricted to the velocity subspace for resolvent analysis.
- MatQ(element):
    Assemble the mass matrix restricted to the velocity subspace without boundary conditions.
- MatD(element, bound):
    Construct a prolongation matrix for a specified subdomain.

Notes
-----
These functions are particularly useful in advanced finite element methods where manipulation of the
function space and associated matrices is required, such as in resolvent analysis or model reduction techniques.

Dependencies
------------
- NumPy
- SciPy
- FEniCS

Ensure that all dependencies are installed and properly configured.
"""

from ..Deps import *
from ..LinAlg.MatrixOps import AssembleMatrix, ConvertMatrix
from ..LinAlg.Utils import get_subspace_info, find_subspace_index


def IdentMatProl(element, index=None):
    """
    Construct the prolongation matrix P, which excludes the subspaces specified by `index` (remove columns in the matrix).
    The transpose of P (P^T) acts as a restriction operator, effectively removing the specified subspaces.

    Parameters
    ----------
    element : object
        The finite element object containing the function space (e.g., an instance of TaylorHood).
    index : int, list, or tuple, optional
        The index or indices of the subspaces to include. If None, the last subspace is included.
        Default is None.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Prolongation matrix of size (n x k), where n is the dimension of the full function space,
        and k is the dimension after excluding the specified subspaces.

    Raises
    ------
    TypeError
        If the `index` parameter is not an integer, list, or tuple.

    Notes
    -----
    - This function is useful when you need to work with a subset of the function space,
      such as excluding pressure components in velocity-pressure formulations.
    - The function constructs a diagonal matrix with ones corresponding to the DOFs to keep
      and zeros corresponding to the DOFs to exclude.

    Examples
    --------
    Exclude the last subspace (e.g., pressure) from the function space:

        P = IdentMatProl(element)

    Exclude multiple subspaces:

        P = IdentMatProl(element, index=[0, 2])
    """
    diag = np.ones(element.functionspace.dim())
    sub_spaces = get_subspace_info(element.functionspace)
    if index is None:  # if None, delete the last subspace
        index = np.sum(sub_spaces) - 1

    if isinstance(index, (int, np.integer)):  # make a list to iterate
        index = [index]
    elif not isinstance(index, (list, tuple)):
        raise TypeError('Wrong type of subspace index specified')

    for ind in index:  # set zeros in diagonal for multiple subspaces
        sub_index = find_subspace_index(ind, sub_spaces)
        # get subsapce to apply
        subfunc = element.functionspace
        for i in sub_index:
            subfunc = subfunc.sub(i)
        # get indices of subspace in global dofs
        subdofs_index = subfunc.dofmap().dofs()
        # set identity array with zeros for the specified subspace
        diag[subdofs_index] = 0

    A = sp.diags(diag).tocsr()
    # Remove zero columns and return
    indices = A.nonzero()
    return A[:, indices[1]]


def MatWgt(element):
    """
    Assemble the mass (weight) matrix for the entire function space.

    Parameters
    ----------
    element : object
        The finite element object containing the function space.

    Returns
    -------
    M : scipy.sparse.csr_matrix
        The assembled mass matrix in sparse CSR format.

    Notes
    -----
    - The mass matrix is assembled using the inner product of the trial and test functions
      over the entire domain.
    - This matrix is often used in computations involving norms or inner products.

    Examples
    --------
    Assemble the mass matrix for a given finite element:

        M = MatWgt(element)
    """

    expr = inner(element.tw, element.tew) * dx
    M = AssembleMatrix(expr)
    return M


def IdentMatBC(element, bcs=[]):
    """
    Construct an identity matrix with zeros at diagonal matrix elements corresponding to boundary conditions.

    Parameters
    ----------
    element : object
        The finite element object containing the function space.
    bcs : list of dolfin.DirichletBC, optional
        List of boundary conditions to apply. Default is an empty list.

    Returns
    -------
    I_bc : scipy.sparse.csr_matrix
        Identity matrix with zeros corresponding to DOFs constrained by boundary conditions.

    Notes
    -----
    - This matrix can be used to modify system matrices or vectors to account for boundary conditions.
    - Applying this matrix effectively eliminates the contributions from the DOFs that are constrained.

    Examples
    --------
    Create an identity matrix considering boundary conditions:

        I_bc = IdentMatBC(element, bcs)
    """

    I = sp.identity(element.functionspace.dim())

    expr = Constant(0.0) * inner(element.tw, element.tew) * dx(99)
    Z = AssembleMatrix(expr, bcs=bcs)
    Z.eliminate_zeros()

    return I - Z


def IdentMatSub(element, bound=None):
    """
    Construct an identity matrix with zeros outside a specified subdomain.

    Parameters
    ----------
    element : object
        The finite element object containing the function space.
    bound : str, optional
        A string expression specifying the condition for the subdomain (e.g., "x[0] < 0.5").
        Default is None, which results in a full identity matrix.

    Returns
    -------
    I_sub : scipy.sparse.csr_matrix
        Identity matrix with zeros for DOFs outside the specified subdomain.

    Raises
    ------
    TypeError
        If `bound` is not a string or None.

    Notes
    -----
    - Useful for operations that are restricted to a certain region of the domain.
    - The function constructs a subdomain and applies Dirichlet conditions to zero out DOFs outside it.

    Examples
    --------
    Construct an identity matrix for a subdomain:

        I_sub = IdentMatSub(element, bound="x[0] < 0.5")
    """

    if bound is None:
        I = sp.identity(element.functionspace.dim())
        return I
    elif isinstance(bound, str):
        Z = PETScMatrix().zero()
        sub_spaces = get_subspace_info(element.functionspace)
        num_subspaces = np.sum(sub_spaces)

        class Omega_1(SubDomain):
            def inside(self, x, on_boundary):
                return eval(bound)

        subdomain1 = Omega_1()
        bc = DirichletBC(element.functionspace, Constant((0,) * num_subspaces), subdomain1)
        bc.apply(Z)
        return ConvertMatrix(Z)
    else:
        raise TypeError('Subdomain condition `bound` must be a string or None.')


def MatP(element):
    """
    Construct a prolongation matrix for resolvent analysis, excluding the pressure subspace.

    Parameters
    ----------
    element : object
        The finite element object containing the function space.

    Returns
    -------
    P : scipy.sparse.csr_matrix
        Prolongation matrix that excludes the pressure subspace.

    Notes
    -----
    - Specifically tailored for resolvent analysis where only the velocity components are of interest.
    - Utilizes `IdentMatProl` to exclude the last subspace, which is typically the pressure.

    Examples
    --------
    Generate the prolongation matrix:

        P = MatP(element)
    """
    P = IdentMatProl(element)
    return P  # size k x k


def MatM(element, bcs=[]):
    """
    Assemble the mass matrix restricted to the velocity subspace for resolvent analysis.

    Parameters
    ----------
    element : object
        The finite element object containing the function space.
    bcs : list of dolfin.DirichletBC, optional
        List of boundary conditions to apply. Default is an empty list.

    Returns
    -------
    M_v : scipy.sparse.csr_matrix
        The mass matrix restricted to the velocity subspace, incorporating boundary conditions.

    Notes
    -----
    - This matrix is useful in resolvent analysis where the focus is on the velocity field.
    - Boundary conditions are applied in an asymmetric way to the mass matrix.

    Examples
    --------
    Assemble the velocity mass matrix with boundary conditions:

        M_v = MatM(element, bcs)
    """

    P = MatP(element)
    M = MatWgt(element)
    Ibc = IdentMatBC(element, bcs)

    return P.transpose() * Ibc * M * P


def MatQ(element):
    """
    Assemble the mass matrix restricted to the velocity subspace without applying boundary conditions.

    Parameters
    ----------
    element : object
        The finite element object containing the function space.

    Returns
    -------
    Q : scipy.sparse.csr_matrix
        The mass matrix restricted to the velocity subspace.

    Notes
    -----
    - Similar to `MatM` but without the application of boundary conditions.
    - Useful when boundary conditions are handled separately or not required.

    Examples
    --------
    Assemble the velocity mass matrix without boundary conditions:

        Q = MatQ(element)
    """

    P = MatP(element)
    M = MatWgt(element)

    return P.transpose() * M * P


def MatD(element, bound):
    """
    Construct a prolongation matrix excluding the pressure subspace for the specified subdomain.

    Parameters
    ----------
    element : object
        The finite element object containing the function space.
    bound : str
        A string expression specifying the subdomain condition (e.g., "x[0] < 0.5").

    Returns
    -------
    D_p : scipy.sparse.csr_matrix of shape k x m.
        Prolongation matrix corresponding to the specified subdomain.

    Notes
    -----
    - If `bound` is None, the identity matrix of size k is returned.
    - Useful when working with a specific subdomain in the analysis.

    Examples
    --------
    Create a prolongation matrix for a subdomain:

        D_p = MatD(element, bound="x[0] < 0.5")
    """

    P = IdentMatProl(element)
    D = IdentMatSub(element, bound)
    Dp = P.transpose() * D * P
    return Dp[:, Dp.nonzero()[1]]  # size k x m
