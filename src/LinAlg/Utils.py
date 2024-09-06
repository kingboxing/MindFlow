#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:48:11 2024

@author: bojin
"""
from src.Deps import *

def assign2(receiving_func, assigning_func):
    """
    Assigns a NumPy array or another FEniCS function to a target FEniCS function.

    Parameters
    ----------
    receiving_func : function.function.Function
        The recieving function. The target FEniCS function to which the values will be assigned.
    assigning_func : np.ndarray, function.function.Function
        The assigning function. The source of the values to be assigned. This can be either:
          - A NumPy array with a size matching the number of DOFs in the target FEniCS function.
          - Another FEniCS function defined on the same function space.

    Raises:
    - ValueError: If the source type is not supported or the sizes do not match.

    """
    if isinstance(assigning_func, np.ndarray):
        # Check if the size of the NumPy array matches the number of DOFs in the FEniCS function
        if receiving_func.vector().size() != assigning_func.size:
            raise ValueError(f"Size mismatch: FEniCS function has {fenics_function.vector().size()} DOFs, "
                             f"but the NumPy array has {source.size} elements.")
        # Assign the NumPy array values to the FEniCS function
        receiving_func.vector()[:]=np.ascontiguousarray(assigning_func)
    elif isinstance(assigning_func, function.function.Function):
        # Check if the function spaces match
        if assigning_func.function_space() != receiving_func.function_space():
            raise ValueError("Function spaces do not match.")
        receiving_func.assign(assigning_func)
    else:
        raise ValueError("Unsupported source type. Must be a NumPy array or a FEniCS function.")

def allclose_spmat(A, B, tol=1e-12):
    """
    Check if two sparse matrices A and B are identical within a tolerance.

    Parameters:
    - A, B: scipy.sparse matrices
    - tol: tolerance value

    Returns:
    - True if matrices are identical within the specified tolerance, otherwise False.
    """
    if A.shape != B.shape:
        return False

    # Compute the difference between the matrices
    diff = A - B

    # Compute the norm of the difference
    diff_norm = spla.norm(diff, ord='fro')

    # Check if the norm is within the tolerance
    return diff_norm <= tol


def get_subspace_info(function_space):
    """
    Get the number of top-level sub-elements and the total number of scalar subspaces 
    in a given two-level function space.

    Parameters
    ----------
    function_space : FunctionSpace
        DESCRIPTION.

    Returns
    -------
    sub_spaces : tuple
        The number of scalar subspaces under each subspace.

    """
    sub_spaces = ()
    num_sub_spaces = function_space.num_sub_spaces()
    for i in range(num_sub_spaces):
        num_space = function_space.sub(i).num_sub_spaces()
        if num_space == 0:
            sub_spaces += (1, )
            #total_scalar_subspaces += 1
        else:
            sub_spaces += (num_space, )
            #total_scalar_subspaces += num_space
    
    return sub_spaces

def find_subspace_index(index, sub_spaces):
    """
    Find the top-level subspace and sub-level subspace indices for a given scalar subspace index.

    Parameters
    ----------
    index : int
        The index of the scalar subspace (0-based index)..
    sub_spaces : tuple
        number scalar subspaces per top level subspace. 
        A list where each entry corresponds to the number of scalar subspaces in a top-level subspace.
        obtained from 'get_subspace_info(function_space)'

    Raises
    ------
    ValueError
        Index out of bounds.

    Returns
    -------
    (top_level_index, sub_level_index) : tuple
        A tuple with the top-level subspace index and the sub-level subspace index (both 0-based)..

    """
    cumulative_sum = 0
    
    for i, num_scalar_subspaces in enumerate(sub_spaces):
        previous_sum = cumulative_sum
        cumulative_sum += num_scalar_subspaces
        if index < cumulative_sum:
            top_level_index = i
            sub_level_index = index - previous_sum
            if sub_spaces[top_level_index]>1:
                return (top_level_index, sub_level_index)
            else: # the number of scalar subspaces in a top-level subspace = 1
                return (top_level_index, )
                
    raise ValueError("Index out of bounds")
    
def is_symmetric(m):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


    
    
    
    
    
    