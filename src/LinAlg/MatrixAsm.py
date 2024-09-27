#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:30:14 2024

@author: bojin

MatrixAsm Module

This module provides utility functions for constructing prolongation matrices, mass/weight matrices, 
and identity matrices with boundary or subdomain conditions for finite element analysis.
"""

from ..Deps import *
from ..LinAlg.MatrixOps import AssembleMatrix, ConvertMatrix
from ..LinAlg.Utils import get_subspace_info, find_subspace_index


def IdentMatProl(element, index=None):
    """
    Construct the prolongation matrix P, which adds the subspace specified by index.
    P^T is the restriction operator, which removes the subspace specified by index.

    Parameters
    ----------
    element : Element
        The finite element containing the function space.
    index : int, list, tuple optional
        The index of the subspace to prolong/restrict. If None, the last subspace is prolonged/removed. Default is None.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Prolongation matrix with size of n x k, where k is the dim after delete the specified subspace. 
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
    Assemble the mass/weight matrix for the entire function space.

    Parameters
    ----------
    element : Element
        The finite element containing the function space.

    Returns
    -------
    M : scipy.sparse.csr_matrix
        Assembled mass/weight matrix.
    """

    expr = inner(element.tw, element.tew) * dx
    M = AssembleMatrix(expr)
    return M


def IdentMatBC(element, bcs=[]):
    """
    Construct an identity matrix with zeros at diagonal matrix elements corresponding to boundary conditions.

    Parameters
    ----------
    element : Element
        The finite element containing the function space.
    bcs : list of DirichletBC, optional
        List of boundary conditions to apply. Default is an empty list.

    Returns
    -------
    scipy.sparse.csr_matrix
        Identity matrix with zeros for boundary condition rows.
    """

    I = sp.identity(element.functionspace.dim())

    expr = Constant(0.0) * inner(element.tw, element.tew) * dx(99)
    Z = AssembleMatrix(expr, bcs=bcs)
    Z.eliminate_zeros()

    return I - Z


def IdentMatSub(element, bound=None):
    """
    Construct an identity matrix for the function space with zeros outside a specified subdomain.

    Parameters
    ----------
    element : Element
        The finite element containing the function space.
    bound : str, optional
        String specifying the condition for the subdomain. Default is None, which applies the identity matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Identity matrix of size nxn with zeros for elements outside the subdomain.
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
        raise TypeError('Wrong type of subdomain condition specified.')


def MatP(element):
    """
    Prolongation matrix for resolvent analysis.
    Prolong/restrict the last subspace (i.e. pressure) and the region outside the specified subdomain if bound is not None
    
    Parameters
    ----------
    element : Element
        The finite element containing the function space.
    bound : str, optional
        String specifying the subdomain condition. Default is None.
    Returns
    -------
    scipy.sparse.csr_matrix
        Prolongation matrix.
    """
    P = IdentMatProl(element)
    return P  # size k x k


def MatM(element, bcs=[]):
    """
    Mass/weight matrix with only the velocity subspace for resolvent analysis.
    Boundary condition applied in the asymmetric way
    

    Parameters
    ----------
    element : Element
        The finite element containing the function space.
    bcs : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    scipy.sparse.csr_matrix
        The mass/weight matrix of size k which is restricted to the velocity subspace.
    """

    P = MatP(element)
    M = MatWgt(element)
    Ibc = IdentMatBC(element, bcs)

    return P.transpose() * Ibc * M * P


def MatQ(element):
    """
    Mass/weight matrix with only the velocity subspace for resolvent analysis.

    Parameters
    ----------
    element : Element
        The finite element containing the function space.

    Returns
    -------
    scipy.sparse.csr_matrix
        The mass/weight matrix of size k which is restricted to the velocity subspace.
    """

    P = MatP(element)
    M = MatWgt(element)

    return P.transpose() * M * P


def MatD(element, bound):
    """
    Prolongation matrix for the specified subdomain if bound is not None
    if bound is None, return identity matrix of size k

    Parameters
    ----------
    element : Element
        The finite element containing the function space.
    bound : str, optional
        String specifying the subdomain condition. Default is None.
    Returns
    -------
    scipy.sparse.csr_matrix
        Prolongation matrix of size k x m.

    """

    P = IdentMatProl(element)
    D = IdentMatSub(element, bound)
    Dp = P.transpose() * D * P
    return Dp[:, Dp.nonzero()[1]]  # size k x m
